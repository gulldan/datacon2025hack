"""Protein preparation for AutoDock Vina.

Downloads PDB file specified by ``config.CHOSEN_PDB_ID`` (if not present),
removes hetero atoms / waters, adds hydrogens and converts to PDBQT using
OpenBabel.

Requirements: biopython, openbabel (CLI `obabel`).
Usage:
    uv run python step_04_hit_selection/protein_prep.py
"""
from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path

from Bio.PDB import PDBList  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER


def download_pdb(pdb_id: str, dst: Path) -> None:
    LOGGER.info("Downloading PDB %s…", pdb_id)
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=str(dst.parent), file_format="pdb")
    downloaded = dst.parent / f"pdb{pdb_id.lower()}.ent"
    downloaded.rename(dst)


def clean_protein(src: Path, out: Path) -> None:
    LOGGER.info("Cleaning protein – removing waters/hetero atoms…")
    from Bio.PDB import PDBIO, PDBParser  # type: ignore

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(src))

    # Remove all HETATM except co-crystallised ligand if desired (kept none)
    io = PDBIO()

    class SelectBackbone:
        def accept_residue(self, residue):  # type: ignore[override]
            return residue.id[0] == " "  # keep only standard residues

    io.set_structure(structure)
    io.save(str(out), select=SelectBackbone())


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    LOGGER.info("Converting to PDBQT via OpenBabel…")
    cmd = [
        "obabel",
        str(pdb_path),
        "-O",
        str(pdbqt_path),
        "--partialcharge",
        "gasteiger",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    pdb = config.PROTEIN_PDB_PATH
    pdbqt = config.PROTEIN_PDBQT_PATH

    if not pdb.exists():
        download_pdb(config.CHOSEN_PDB_ID, pdb)
    clean_path = pdb.with_suffix("_clean.pdb")
    clean_protein(pdb, clean_path)
    pdb_to_pdbqt(clean_path, pdbqt)
    LOGGER.info("Receptor prepared: %s", pdbqt)


if __name__ == "__main__":
    main()
