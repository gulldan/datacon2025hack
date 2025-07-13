"""Ligand preparation for AutoDock Vina.

Reads `generated_molecules.parquet`, generates 3D conformers with RDKit ETKDG,
adds hydrogens and converts to PDBQT via OpenBabel CLI.
"""
from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path

import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER


def smiles_to_3d_pdb(smiles: str, out_path: Path) -> bool:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return False
    mol = Chem.AddHs(mol)  # type: ignore[attr-defined]
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # type: ignore[attr-defined]
        AllChem.UFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    except Exception:
        return False
    Chem.MolToPDBFile(mol, str(out_path))  # type: ignore[attr-defined]
    return True


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path), "--partialcharge", "gasteiger"]
    subprocess.run(cmd, check=True)


def main() -> None:
    src = config.GENERATED_MOLECULES_PATH
    if not src.exists():
        LOGGER.error("Generated molecules parquet not found: %s", src)
        return

    df = pl.read_parquet(src)
    LOGGER.info("Preparing %d ligands for docking…", len(df))

    for idx, smi in enumerate(df["smiles"]):
        pdb_path = config.LIGAND_PDBQT_DIR / f"lig_{idx}.pdb"
        pdbqt_path = pdb_path.with_suffix(".pdbqt")
        if pdbqt_path.exists():
            continue
        if not smiles_to_3d_pdb(smi, pdb_path):
            LOGGER.warning("Failed to generate 3D for %s", smi)
            continue
        pdb_to_pdbqt(pdb_path, pdbqt_path)
    LOGGER.info("Ligand preparation finished. Files in %s", config.LIGAND_PDBQT_DIR)


if __name__ == "__main__":
    main()
