"""Verify that every headline number in README.md still matches the data.

The README quotes exact counts for the Lipinski and CNS filter panels, the
docking score distribution and the docking success rate. Those numbers were
computed once, by hand, from
``step_04_hit_selection/results/panels/ligands_descriptors.parquet``.

This test recomputes them from that same file, so the README cannot silently
drift away from the artefacts it describes. If a pipeline rerun changes the
numbers, this test fails and the README has to be updated in the same commit.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIGANDS = REPO_ROOT / "step_04_hit_selection" / "results" / "panels" / "ligands_descriptors.parquet"

# (column, comparison, threshold, expected count) exactly as quoted in README.md
LIPINSKI = [
    ("MW", "le", 500, 789),
    ("LogP", "le", 5, 739),
    ("HBD", "le", 5, 814),
    ("HBA", "le", 10, 814),
    ("TPSA", "le", 140, 811),
]
CNS = [
    ("MW", "le", 450, 729),
    ("LogP", "le", 3, 387),
    ("TPSA", "le", 90, 580),
    ("RotB", "le", 3, 287),
    ("CNS_MPO", "ge", 4, 200),
]

TOTAL_MOLECULES = 814
LIPINSKI_ALL = 727
CNS_ALL = 63
DOCKED_OK = 809
DOCKING_FAILED = 5
DOCKING_MEAN = -6.63
DOCKING_BEST = -10.72
DOCKING_UNDER_MINUS_7 = 226


@pytest.fixture(scope="module")
def table():
    if not LIGANDS.exists():
        pytest.skip(f"{LIGANDS.relative_to(REPO_ROOT)} is not present")
    return pq.read_table(LIGANDS).to_pydict()


def _count(table, column, comparison, threshold):
    values = [v for v in table[column] if v is not None]
    if comparison == "le":
        return sum(1 for v in values if v <= threshold)
    return sum(1 for v in values if v >= threshold)


def test_total_molecules(table):
    assert len(table["smiles"]) == TOTAL_MOLECULES


@pytest.mark.parametrize(("column", "comparison", "threshold", "expected"), LIPINSKI)
def test_lipinski_panel(table, column, comparison, threshold, expected):
    assert _count(table, column, comparison, threshold) == expected


@pytest.mark.parametrize(("column", "comparison", "threshold", "expected"), CNS)
def test_cns_panel(table, column, comparison, threshold, expected):
    assert _count(table, column, comparison, threshold) == expected


def test_lipinski_all_rules(table):
    rows = zip(table["MW"], table["LogP"], table["HBD"], table["HBA"], table["TPSA"])
    passing = sum(1 for mw, logp, hbd, hba, tpsa in rows if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and tpsa <= 140)
    assert passing == LIPINSKI_ALL


def test_cns_all_filters(table):
    rows = zip(table["MW"], table["LogP"], table["TPSA"], table["RotB"], table["CNS_MPO"])
    passing = sum(1 for mw, logp, tpsa, rotb, mpo in rows if mw <= 450 and logp <= 3 and tpsa <= 90 and rotb <= 3 and mpo >= 4)
    assert passing == CNS_ALL


def test_docking_success_rate(table):
    scores = table["docking_score"]
    assert sum(1 for s in scores if s is None) == DOCKING_FAILED
    assert sum(1 for s in scores if s is not None) == DOCKED_OK


def test_docking_distribution(table):
    scores = [s for s in table["docking_score"] if s is not None]
    assert round(sum(scores) / len(scores), 2) == DOCKING_MEAN
    assert round(min(scores), 2) == DOCKING_BEST
    assert sum(1 for s in scores if s <= -7) == DOCKING_UNDER_MINUS_7


# "Строгие критерии для терапии болезни Альцгеймера" — the second results block.
STRICT = [
    ("MW", lambda v: v <= 450, 729),
    ("LogP", lambda v: 1 <= v <= 4, 538),
    ("TPSA", lambda v: v < 70, 295),
    ("HBD", lambda v: v <= 3, 795),
    ("HBA", lambda v: v <= 7, 786),
    ("RotB", lambda v: v <= 8, 809),
    ("QED", lambda v: 0.4 <= v <= 0.8, 478),
    ("CNS_MPO", lambda v: v >= 4, 200),
    ("logBB_est", lambda v: v > -0.3, 427),
    ("pgp_efflux", lambda v: v < 2, 46),
    ("docking_score", lambda v: v < -8, 38),
    ("herg_blocker", lambda v: v < 0.3, 32),
    ("SA_score", lambda v: v < 5, 814),
]
STRICT_ALL = 0


@pytest.mark.parametrize(("column", "predicate", "expected"), STRICT, ids=[s[0] for s in STRICT])
def test_strict_criteria_counts(table, column, predicate, expected):
    assert sum(1 for v in table[column] if v is not None and predicate(v)) == expected


def test_no_molecule_passes_every_strict_criterion(table):
    columns = [c for c, _, _ in STRICT]
    predicates = [p for _, p, _ in STRICT]
    passing = 0
    for row in zip(*(table[c] for c in columns)):
        if all(v is not None and pred(v) for v, pred in zip(row, predicates)):
            passing += 1
    assert passing == STRICT_ALL
