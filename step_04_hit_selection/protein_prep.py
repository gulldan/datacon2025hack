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
    LOGGER.info(f"Downloading PDB {pdb_id}…")
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
        def accept_model(self, model):  # type: ignore[override]
            return True

        def accept_chain(self, chain):  # type: ignore[override]
            return True

        def accept_residue(self, residue):  # type: ignore[override]
            return residue.id[0] == " "  # keep only standard residues

        def accept_atom(self, atom):  # type: ignore[override]
            return True

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


def strip_torsion_tree(pdbqt_path: Path) -> None:
    """Remove ROOT/BRANCH sections written by OpenBabel (ligand torsion tree).

    AutoDock Vina expects rigid receptor files to contain only ATOM/HETATM/TER etc.
    Tags like ROOT, ENDROOT, BRANCH, ENDBRANCH and TORSDOF cause a parse error.
    """
    allowed_prefixes = {"ATOM", "HETATM", "TER", "REMARK", "ENDMDL", "MODEL", "END"}
    lines = pdbqt_path.read_text().splitlines()
    filtered = [ln for ln in lines if ln[:6].strip() in allowed_prefixes]
    pdbqt_path.write_text("\n".join(filtered) + "\n")


def main() -> None:
    pdb = config.PROTEIN_PDB_PATH
    pdbqt = config.PROTEIN_PDBQT_PATH

    if not pdb.exists():
        download_pdb(config.CHOSEN_PDB_ID, pdb)
    clean_path = pdb.with_name(pdb.stem + "_clean.pdb")
    clean_protein(pdb, clean_path)
    pdb_to_pdbqt(clean_path, pdbqt)
    strip_torsion_tree(pdbqt)
    LOGGER.info(f"Receptor prepared: {pdbqt}")


if __name__ == "__main__":
    main()
