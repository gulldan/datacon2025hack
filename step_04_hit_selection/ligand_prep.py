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
    """Convert PDB to PDBQT via OpenBabel with post-processing."""
    cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path), "--partialcharge", "gasteiger"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)

        # Post-process to fix multiple ROOT sections
        if pdbqt_path.exists():
            fix_pdbqt_multiple_roots(pdbqt_path)

    except subprocess.CalledProcessError as e:
        LOGGER.warning(f"OpenBabel conversion failed: {e}")


def fix_pdbqt_multiple_roots(pdbqt_path: Path) -> None:
    """Fix PDBQT files with multiple ROOT sections by keeping only the first one."""
    try:
        with open(pdbqt_path) as f:
            lines = f.readlines()

        # Find the first ROOT section and keep only that
        new_lines = []
        in_first_root = False
        found_first_root = False

        for line in lines:
            if line.strip() == "ROOT":
                if not found_first_root:
                    found_first_root = True
                    in_first_root = True
                    new_lines.append(line)
                else:
                    # Skip subsequent ROOT sections
                    break
            elif line.strip() == "ENDROOT":
                if in_first_root:
                    new_lines.append(line)
                    in_first_root = False
                # Skip subsequent ENDROOT
            elif in_first_root or not found_first_root:
                new_lines.append(line)

        # Add final TORSDOF if not present
        if new_lines and not any("TORSDOF" in line for line in new_lines):
            new_lines.append("TORSDOF 0\n")

        # Write back the cleaned file
        with open(pdbqt_path, "w") as f:
            f.writelines(new_lines)

    except Exception as e:
        LOGGER.warning(f"Failed to fix PDBQT file {pdbqt_path}: {e}")


def is_valid_pdbqt(pdbqt_path: Path) -> bool:
    """Return True if PDBQT contains at least one ATOM/HETATM line."""
    try:
        for line in pdbqt_path.open():
            if line.startswith(("ATOM", "HETATM")):
                return True
    except FileNotFoundError:
        return False
    return False


def main() -> None:
    src = config.GENERATED_MOLECULES_PATH
    if not src.exists():
        LOGGER.error(f"Generated molecules parquet not found: {src}")
        return

    df = pl.read_parquet(src)
    LOGGER.info(f"Preparing {len(df)} ligands for docking…")

    for idx, smi in enumerate(df["smiles"]):
        pdb_path = config.LIGAND_PDBQT_DIR / f"lig_{idx}.pdb"
        pdbqt_path = pdb_path.with_suffix(".pdbqt")
        if pdbqt_path.exists():
            continue
        if not smiles_to_3d_pdb(smi, pdb_path):
            LOGGER.warning(f"Failed to generate 3D for {smi}")
            continue
        pdb_to_pdbqt(pdb_path, pdbqt_path)
        # Validate conversion – OpenBabel sometimes produces empty PDBQT if unsupported atoms
        if not is_valid_pdbqt(pdbqt_path):
            LOGGER.warning(f"OpenBabel failed to create valid PDBQT for {smi} (idx {idx}). Skipping ligand.")
            pdbqt_path.unlink(missing_ok=True)
            pdb_path.unlink(missing_ok=True)
            continue
    LOGGER.info(f"Ligand preparation finished. Files in {config.LIGAND_PDBQT_DIR}")


if __name__ == "__main__":
    main()
