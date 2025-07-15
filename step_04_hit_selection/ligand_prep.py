"""Ligand preparation for AutoDock Vina with ID mapping.

Reads `generated_molecules.parquet`, assigns a unique `id_ligand` (e.g. ``lig_0``) to each
row, generates 3D conformers with RDKit ETKDG, adds hydrogens and converts to PDBQT via
OpenBabel CLI. The mapping between `id_ligand` and original SMILES is stored in
``ligand_id_map.parquet`` for downstream docking result alignment.

This optimized version processes ligands in parallel to significantly speed up execution.
"""

from __future__ import annotations

import subprocess
import sys as _sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration (expected to be defined in config.py) ----------------------

# Add project root to import path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

# Try to import user‑level config and logger; fall back to mock versions
try:
    import config  # type: ignore
    from utils.logger import LOGGER  # type: ignore
except ImportError:
    import logging

    class MockConfig:
        GENERATED_MOLECULES_PATH = Path("generated_molecules.parquet")
        LIGAND_PDBQT_DIR = Path("ligands_pdbqt")
        # <‑‑ new path to save the ID → SMILES mapping
        LIGAND_MAPPING_PATH = Path("ligand_id_map.parquet")

    config = MockConfig()  # type: ignore
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Existing helper functions (signatures unchanged)
# -----------------------------------------------------------------------------


def smiles_to_3d_pdb(smiles: str, out_path: Path) -> bool:
    """Generate a 3D conformer from SMILES and save it as PDB.

    Returns ``True`` on success, ``False`` otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        LOGGER.warning("RDKit failed to read SMILES: %s", smiles)
        return False

    mol = Chem.AddHs(mol)

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # reproducibility
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result == -1:
            LOGGER.error("Failed to embed conformer for %s — molecule is too strained", smiles)
            return False
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error(f"Unexpected error while generating 3D for {smiles}: {exc}")
        return False

    Chem.MolToPDBFile(mol, str(out_path))
    return True


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    """Convert PDB to PDBQT using OpenBabel and post‑process."""
    cmd = [
        "obabel",
        str(pdb_path),
        "-O",
        str(pdbqt_path),
        "--partialcharge",
        "gasteiger",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if pdbqt_path.exists():
            fix_pdbqt_multiple_roots(pdbqt_path)
    except subprocess.CalledProcessError as exc:
        LOGGER.warning(f"OpenBabel conversion failed for {pdb_path.name}: {exc.stderr.strip()}")


def fix_pdbqt_multiple_roots(pdbqt_path: Path) -> None:
    """Keep only the first ROOT / ENDROOT block and ensure TORSDOF is present."""
    try:
        with open(pdbqt_path, encoding="utf‑8") as handle:
            lines = handle.readlines()

        root_indices = [i for i, line in enumerate(lines) if line.strip() == "ROOT"]
        if len(root_indices) > 1:
            try:
                endroot_index = lines.index("ENDROOT\n", root_indices[0])
                lines = lines[: endroot_index + 1]
            except ValueError:
                LOGGER.warning(f"Multiple ROOTs but no ENDROOT in {pdbqt_path} — skipping fix")
                return

        if not any("TORSDOF" in line for line in lines):
            insert_pos = -1 if lines and lines[-1].strip().startswith(("ENDROOT", "ENDBRANCH")) else len(lines)
            lines.insert(insert_pos, "TORSDOF 0\n")

        with open(pdbqt_path, "w", encoding="utf‑8") as handle:
            handle.writelines(lines)

    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(f"Failed to fix PDBQT {pdbqt_path}: {exc}")


def is_valid_pdbqt(pdbqt_path: Path) -> bool:
    """Return ``True`` iff file exists & contains at least one ATOM/HETATM line."""
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
    try:
        with open(pdbqt_path, encoding="utf‑8") as handle:
            return any(line.startswith(("ATOM", "HETATM")) for line in handle)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error(f"Error reading {pdbqt_path}: {exc}")
        return False


# -----------------------------------------------------------------------------
# Worker function for multiprocessing
# -----------------------------------------------------------------------------


def _prepare_single(args: tuple[int, str], ligand_dir: Path) -> bool:
    """End‑to‑end preparation for one ligand (internal helper for Pool)."""
    idx, smiles = args
    ligand_id = f"lig_{idx}"

    pdbqt_path = ligand_dir / f"{ligand_id}.pdbqt"
    if pdbqt_path.exists():
        return True  # already processed

    pdb_path = ligand_dir / f"{ligand_id}.pdb"

    if not smiles_to_3d_pdb(smiles, pdb_path):
        LOGGER.warning(f"[{ligand_id}] Skipping ligand — 3D generation failed")
        pdb_path.unlink(missing_ok=True)
        return False

    pdb_to_pdbqt(pdb_path, pdbqt_path)

    if not is_valid_pdbqt(pdbqt_path):
        LOGGER.warning(f"[{ligand_id}] Invalid PDBQT produced — skipping")
        pdbqt_path.unlink(missing_ok=True)
        pdb_path.unlink(missing_ok=True)
        return False

    pdb_path.unlink(missing_ok=True)
    return True


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main() -> None:
    """Prepare ligands in parallel and persist SMILES ↔ ID mapping."""
    src = config.GENERATED_MOLECULES_PATH
    if not src.exists():
        LOGGER.error(f"Molecule file not found: {src}")
        return

    config.LIGAND_PDBQT_DIR.mkdir(exist_ok=True)

    df = pl.read_parquet(src)

    # Add id_ligand column (lig_0, lig_1, …)
    df = df.with_columns(pl.Series("id_ligand", [f"lig_{i}" for i in range(len(df))]))

    # Persist mapping for downstream docking alignment
    mapping_path = config.LIGAND_MAPPING_PATH
    df.select(["id_ligand", "smiles"]).write_parquet(mapping_path)
    LOGGER.info(f"ID → SMILES mapping saved to: {mapping_path}")

    smiles_list = df["smiles"].to_list()
    total = len(smiles_list)
    LOGGER.info("Preparing %d ligands for docking…", total)

    tasks = list(enumerate(smiles_list))
    worker = partial(_prepare_single, ligand_dir=config.LIGAND_PDBQT_DIR)

    with Pool() as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=total, desc="Ligand prep"))

    success = sum(results)
    LOGGER.info("--- Preparation statistics ---")
    LOGGER.info(f"Total ligands:      {total}")
    LOGGER.info(f"Successfully ready: {success}")
    LOGGER.info(f"Failed:            {total - success}")
    LOGGER.info(f"PDBQT directory:    {config.LIGAND_PDBQT_DIR}")


if __name__ == "__main__":
    # Create a small test dataset when run standalone
    if not config.GENERATED_MOLECULES_PATH.exists():
        LOGGER.info("Creating demo generated_molecules.parquet …")
        test_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "C#CCCN1C[C@@H]2C(NC(=O)c3ccc4c(c3)C(C)(C)CO4)[C@H]2C1",  # Strained example
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Caffeine
        ]
        pl.DataFrame({"smiles": test_smiles}).write_parquet(config.GENERATED_MOLECULES_PATH)

    main()
