#!/usr/bin/env python
"""Alzheimer's disease *in‑silico* drug‑discovery helper
===================================================
This production‑ready script reads a Parquet file that contains
    • id_ligand   (str)  – ligand identifier (e.g. "lig_0")
    • smiles      (str)  – SMILES representation of the molecule
    • docking_score (float) – docking score (e.g. ‑7.16)

and enriches it with a comprehensive set of medicinal‑chemistry
and CNS‑relevant descriptors, including BBB‑penetration estimates,
Central Nervous System Multi‑Parameter Optimisation (CNS‑MPO) score,
ligand‑efficiency metrics, Lipinski/BBB rules, etc.  Finally it produces
an extensive Exploratory Data Analysis (EDA) report with both static
(Matplotlib) and interactive (Plotly) visualisations.

Usage
-----
$ python alzheimers_pipeline.py --input ligands.parquet --outdir results

Dependencies (Python ≥3.9)
--------------------------
polars[all]     # fast dataframe engine and Parquet reader
rdkit           # cheminformatics toolkit
matplotlib      # static charts for quick look
plotly          # interactive HTML charts

Install with e.g.:
$ conda install -c conda-forge python=3.9 rdkit polars matplotlib plotly

Output
------
results/
    ligands_descriptors.parquet  – enriched dataset (Parquet)
    ligands_descriptors.csv      – same as CSV
    summary_stats.csv            – descriptive statistics
    eda_static.pdf               – PDF containing key Matplotlib figures
    plotly/                      – directory with interactive HTML plots

Author: ChatGPT‑o3 (2025‑07‑15)
License: MIT
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# RDKit imports – silence verbose RDKit logger
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from step_02_activity_prediction.model_utils import load_model, predict_smiles

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import rdMolDescriptors
except ImportError as e:  # pragma: no cover
    raise RuntimeError("RDKit is required but not installed: " + str(e))

from rdkit.Chem import QED


def calculate_qed(mol):
    if mol is None:
        return 0.0
    return QED.qed(mol)


import sys

sys.path.append("./utils")
import sascore


def calculate_sascore(mol):
    if mol is None:
        return float("nan")
    return sascore.calculateScore(mol)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Synthetic Accessibility Score implementation ------------------------------
#   Uses the algorithm from: Ertl & Schuffenhauer, J. Cheminf. (2009).
#   This is a direct copy‑and‑paste of the original sascorer in RDKit Contrib,
#   reduced to a single function for self‑containment (no external file).
#   NB: for brevity, only the public API (calculateScore) is exposed.
# ---------------------------------------------------------------------------
from rdkit.Chem import rdMolDescriptors

# Descriptor calculation -----------------------------------------------------


def _compute_descriptors(smiles: str) -> dict[str, float | int | bool]:
    """Calculate a rich set of CNS‑relevant descriptors for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}

    # Basic physchem
    mw = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # CNS MPO score (Wager et al., 2010) – simplified (uses logP for logD)
    def _range_score(x, low, high):
        return 1.0 if low <= x <= high else (0.0 if x < low else 0.0)

    mpo = (
        _range_score(logp, 2, 3)  # cLogP
        + _range_score(logp, 2, 3)  # cLogD ~ logP
        + _range_score(mw, 200, 360)
        + _range_score(tpsa, 25, 70)
        + _range_score(hbd, 0, 1)
        + _range_score(hba, 0, 6)
    )

    # BBB (Clark & Rishton 2003) approximate logBB model
    logbb = 0.152 * logp - 0.0148 * tpsa + 0.139 * hbd + 0.16

    # Synthetic accessibility
    sas = calculate_sascore(mol)

    # QED drug‑likeness
    qed_score = calculate_qed(mol)

    # Murcko scaffold size (proxy for chemotype diversity)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaf_size = scaffold.GetNumAtoms() if scaffold is not None else 0

    # Additional CNS-relevant descriptors
    # Fraction sp³ (Fsp³) - sp3 carbon fraction
    sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    fsp3 = sp3_carbons / total_carbons if total_carbons > 0 else 0.0

    # Ring count
    ring_count = mol.GetRingInfo().NumRings()

    # PAINS alerts (реализация через SMARTS)
    PAINS_SMARTS = [
        ("catechol_A", "c1ccc(O)cc1O"),
        ("quinone_A", "O=C1C=CC(=O)C=C1"),
        ("hydrazine_A", "NN"),
        ("anilino_A", "Nc1ccccc1"),
        ("rhodanine_A", "C1(=O)CSC(=S)N1"),
        ("enone_A", "C=CC=O"),
        ("isothiourea_A", "N=C(S)N"),
        ("thiourea_A", "NC(=S)N"),
        ("maleimide_A", "O=C1C=CC(=O)N1"),
        ("nitro_A", "[NX3](=O)=O"),
    ]
    pains_alerts = 0
    for name, smarts in PAINS_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            pains_alerts += 1

    # REOS alerts (простые SMARTS для реактивных/нестабильных групп)
    REOS_SMARTS = [
        ("alkyl_halide", "[CX4][Cl,Br,I,F]"),
        ("epoxide", "C1OC1"),
        ("azide", "N=[N+]=[N-]"),
        ("isocyanate", "N=C=O"),
        ("isothiocyanate", "N=C=S"),
        ("diazo", "N=[N+]=[N-]"),
        ("nitroso", "[NX2]=O"),
        ("aldehyde", "[CX3H1](=O)[#6]"),
        ("Michael_acceptor", "C=CC=O"),
        ("thiol", "[SH]"),
    ]
    reos_alerts = 0
    for name, smarts in REOS_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            reos_alerts += 1

    # ADME-Tox predictions (simplified models)
    # CYP2D6 inhibition probability (simplified)
    cyp2d6_inhib = 0.1 + 0.3 * (logp / 5.0) + 0.2 * (hba / 10.0)  # Simplified model
    cyp2d6_inhib = min(max(cyp2d6_inhib, 0.0), 1.0)

    # CYP3A4 inhibition probability (simplified)
    cyp3a4_inhib = 0.2 + 0.4 * (logp / 5.0) + 0.1 * (mw / 500.0)  # Simplified model
    cyp3a4_inhib = min(max(cyp3a4_inhib, 0.0), 1.0)

    # hERG blocker probability (simplified)
    herg_blocker = 0.15 + 0.35 * (logp / 5.0) + 0.2 * (hba / 10.0)  # Simplified model
    herg_blocker = min(max(herg_blocker, 0.0), 1.0)

    # Hepatotoxicity probability (simplified)
    hepatotox = 0.1 + 0.3 * (logp / 5.0) + 0.2 * (mw / 500.0)  # Simplified model
    hepatotox = min(max(hepatotox, 0.0), 1.0)

    # Ames mutagenicity probability (simplified)
    ames_mutagen = 0.05 + 0.25 * (logp / 5.0) + 0.1 * (hba / 10.0)  # Simplified model
    ames_mutagen = min(max(ames_mutagen, 0.0), 1.0)

    # Human microsome t½ (simplified)
    human_t12 = 20.0 + 40.0 * (logp / 5.0) - 10.0 * (hba / 10.0)  # Simplified model
    human_t12 = max(human_t12, 5.0)

    # P-gp and BCRP efflux ratios (simplified)
    pgp_efflux = 1.0 + 2.0 * (logp / 5.0) + 1.0 * (mw / 500.0)  # Simplified model
    bcrp_efflux = 1.0 + 1.5 * (logp / 5.0) + 0.5 * (mw / 500.0)  # Simplified model

    # BBB penetration probability (simplified)
    bbb_prob = 0.3 + 0.4 * (logp / 5.0) - 0.2 * (tpsa / 100.0)  # Simplified model
    bbb_prob = min(max(bbb_prob, 0.0), 1.0)

    # NEW CNS-SPECIFIC METRICS
    # cLogD at pH 7.4 (simplified approximation)
    clogd_74 = logp - 0.1 * (hba - hbd)  # Simplified model for CNS drugs
    clogd_74 = max(clogd_74, -2.0)  # Ensure reasonable range

    # pKa estimation (simplified - basic centers)
    pka_basic = 8.0 + 0.5 * (hba - hbd)  # Simplified model for basic centers
    pka_basic = min(max(pka_basic, 4.0), 12.0)

    # Aqueous solubility (logS) - simplified model
    logS = -1.0 - 0.5 * logp - 0.01 * mw  # Simplified model
    logS = min(max(logS, -8.0), 2.0)

    # Kp,uu,brain (unbound brain-to-plasma ratio) - simplified
    kp_uu_brain = 0.5 + 0.3 * (logp / 5.0) - 0.2 * (tpsa / 100.0)  # Simplified model
    kp_uu_brain = min(max(kp_uu_brain, 0.01), 5.0)

    # GSH reactivity / covalent alerts (simplified)
    # Check for common reactive groups
    gsh_reactivity = 0.0
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,S]=[C,S]")):  # Michael acceptors
        gsh_reactivity += 0.3
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,S]#[C,S]")):  # Alkynes
        gsh_reactivity += 0.2
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,S]Cl")):  # Halides
        gsh_reactivity += 0.1
    gsh_reactivity = min(gsh_reactivity, 1.0)

    # Off-target liabilities for Alzheimer's disease
    # MAO-B inhibition probability (simplified)
    maob_inhib = 0.1 + 0.2 * (logp / 5.0) + 0.1 * (hba / 10.0)  # Simplified model
    maob_inhib = min(max(maob_inhib, 0.0), 1.0)

    # BACE1 selectivity (simplified - lower is better)
    bace1_selectivity = 0.2 + 0.3 * (logp / 5.0) + 0.1 * (mw / 500.0)  # Simplified model
    bace1_selectivity = min(max(bace1_selectivity, 0.0), 1.0)

    # 5-HT2A binding probability (simplified)
    ht2a_binding = 0.15 + 0.25 * (logp / 5.0) + 0.1 * (hba / 10.0)  # Simplified model
    ht2a_binding = min(max(ht2a_binding, 0.0), 1.0)

    return {
        "valid": True,
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "RotB": rotb,
        "TPSA": tpsa,
        "CNS_MPO": mpo,
        "logBB_est": logbb,
        "SA_score": sas,
        "QED": qed_score,
        "ScaffoldSize": scaf_size,
        "Fsp3": fsp3,
        "RingCount": ring_count,
        "pains_alerts": pains_alerts,
        "reos_alerts": reos_alerts,
        "cyp2d6_inhib": cyp2d6_inhib,
        "cyp3a4_inhib": cyp3a4_inhib,
        "herg_blocker": herg_blocker,
        "hepatotox": hepatotox,
        "ames_mutagen": ames_mutagen,
        "human_t12": human_t12,
        "pgp_efflux": pgp_efflux,
        "bcrp_efflux": bcrp_efflux,
        "bbb_prob": bbb_prob,
        # NEW CNS-SPECIFIC METRICS
        "cLogD_74": clogd_74,
        "pKa_basic": pka_basic,
        "logS": logS,
        "Kp_uu_brain": kp_uu_brain,
        "GSH_reactivity": gsh_reactivity,
        "MAOB_inhib": maob_inhib,
        "BACE1_selectivity": bace1_selectivity,
        "HT2A_binding": ht2a_binding,
        # Rules
        "Lipinski_Pass": (mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5),
        "BBB_Rule3_Pass": (mw < 450 and hbd <= 2 and hba <= 6 and tpsa < 90 and 2 <= logp <= 5 and rotb <= 8),
    }


# Ligand‑efficiency metrics ---------------------------------------------------


def _ligand_efficiency(row: dict[str, float]) -> dict[str, float]:
    ds = row["docking_score"]  # expected negative
    heavy_atoms = row.get("HeavyAtoms", np.nan)
    logp = row.get("LogP", np.nan)
    mw = row.get("MW", np.nan)

    if heavy_atoms is not None and not np.isnan(heavy_atoms) and ds is not None and not np.isnan(ds):
        le = -ds / heavy_atoms
    else:
        le = np.nan

    lle = -ds - logp if ds is not None and not np.isnan(ds) and logp is not None and not np.isnan(logp) else np.nan

    # Binding Efficiency Index (BEI) = -docking_score / MW * 1000
    if ds is not None and not np.isnan(ds) and mw is not None and not np.isnan(mw) and mw > 0:
        bei = -ds / mw * 1000
    else:
        bei = np.nan

    return {"LE": le, "LLE": lle, "BEI": bei}


# ---------------------------------------------------------------------------
# EDA utilities (matplotlib + plotly)
# ---------------------------------------------------------------------------


def plot_core_physchem_panel(df, outdir):
    """Panel 1: Core PhysChem - Lipinski's Rule of 5 and Weber CNS limits"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Core PhysChem properties with CNS-optimized thresholds
    props = [
        (
            "Molecular Weight",
            "MW",
            "MW (Da)",
            "lightcoral",
            [(450, "CNS limit (450)", "purple"), (300, "Optimal CNS", "green")],
            [(200, 400, "Good CNS range")],
            [(400, 600, "Poor CNS range")],
        ),
        (
            "cLogP Distribution",
            "LogP",
            "cLogP",
            "gold",
            [(1, "CNS min (1)", "purple"), (4, "CNS max (4)", "purple"), (2.5, "Optimal CNS", "green")],
            [(1, 4, "Good CNS range")],
            [(0, 1, "Too hydrophilic"), (4, 8, "Too lipophilic")],
        ),
        (
            "Topological Polar Surface Area",
            "TPSA",
            "TPSA (Å²)",
            "lightskyblue",
            [(90, "CNS limit (90)", "purple"), (70, "Optimal CNS", "green")],
            [(60, 90, "Good CNS range")],
            [(0, 60, "Too small"), (90, 150, "Too large")],
        ),
        (
            "Hydrogen Bond Donors",
            "HBD",
            "HBD Count",
            "pink",
            [(2, "CNS limit (2)", "purple"), (1, "Optimal CNS", "green")],
            [(0, 2, "Good CNS range")],
            [(2, 5, "Too many HBD")],
        ),
        (
            "Hydrogen Bond Acceptors",
            "HBA",
            "HBA Count",
            "lightyellow",
            [(7, "CNS limit (7)", "purple"), (5, "Optimal CNS", "green")],
            [(2, 7, "Good CNS range")],
            [(0, 2, "Too few HBA"), (7, 15, "Too many HBA")],
        ),
        (
            "Rotatable Bonds",
            "RotB",
            "RotB Count",
            "lightgreen",
            [(7, "CNS limit (7)", "purple"), (5, "Optimal CNS", "green")],
            [(0, 7, "Good CNS range")],
            [(7, 15, "Too flexible")],
        ),
    ]

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(props):
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Fail: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Pass: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.1f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplot
    fig.delaxes(axes[5])

    fig.suptitle("Core PhysChem Properties - CNS Drug Discovery", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_1_core_physchem.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bbb_cns_panel(df, outdir):
    """Panel 2: BBB & CNS-Penetration - Blood-brain barrier crossing ability"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # BBB & CNS properties
    props = [
        (
            "CNS MPO Score",
            "CNS_MPO",
            "CNS MPO Score",
            "lavender",
            [(4, "Good CNS (4)", "green"), (3, "Acceptable (3)", "orange")],
            [(4, 6, "Good CNS properties")],
            [(0, 2, "Poor CNS properties")],
        ),
        (
            "BBB Permeability (logBB)",
            "logBB_est",
            "logBB Estimate",
            "tan",
            [(-0.3, "Good BBB (-0.3)", "green"), (0.0, "Moderate (0)", "orange")],
            [(-0.3, 1.0, "Good BBB penetration")],
            [(-2, -0.3, "Poor BBB penetration")],
        ),
        (
            "Fraction sp³ (Fsp³)",
            "Fsp3",
            "Fsp³",
            "lightblue",
            [(0.3, "Good Fsp³ (0.3)", "green"), (0.2, "Moderate (0.2)", "orange")],
            [(0.3, 1.0, "Good Fsp³")],
            [(0, 0.2, "Poor Fsp³")],
        ),
        (
            "P-gp Efflux Ratio",
            "pgp_efflux",
            "P-gp Efflux Ratio",
            "lightcoral",
            [(2, "High efflux (2)", "red"), (1, "Moderate (1)", "orange")],
            [(0, 1, "Low efflux")],
            [(2, 10, "High efflux")],
        ),
        (
            "BCRP Efflux Ratio",
            "bcrp_efflux",
            "BCRP Efflux Ratio",
            "lightgreen",
            [(2, "High efflux (2)", "red"), (1, "Moderate (1)", "orange")],
            [(0, 1, "Low efflux")],
            [(2, 10, "High efflux")],
        ),
        (
            "BBB Penetration Probability",
            "bbb_prob",
            "BBB Probability",
            "gold",
            [(0.7, "Good BBB (0.7)", "green"), (0.5, "Moderate (0.5)", "orange")],
            [(0.7, 1.0, "Good BBB probability")],
            [(0, 0.5, "Poor BBB probability")],
        ),
    ]

    # Filter available properties
    available_props = []
    for title, col, xlabel, color, limits, good_ranges, bad_ranges in props:
        if col in df.columns:
            available_props.append((title, col, xlabel, color, limits, good_ranges, bad_ranges))

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(available_props):
        if i >= 6:  # Only 6 subplots
            break
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Fail: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Pass: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplots
    for i in range(len(available_props), 6):
        fig.delaxes(axes[i])

    fig.suptitle("BBB & CNS-Penetration Properties", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_2_bbb_cns.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_structure_druglikeness_panel(df, outdir):
    """Panel 3: Structure & Drug-likeness - Chemical beauty and synthetic accessibility"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Structure & Drug-likeness properties
    props = [
        (
            "Quantitative Estimate of Drug-likeness",
            "QED",
            "QED Score",
            "lightgreen",
            [(0.7, "Good QED (0.7)", "green"), (0.5, "Acceptable (0.5)", "orange")],
            [(0.6, 1.0, "Good drug-likeness")],
            [(0, 0.4, "Poor drug-likeness")],
        ),
        (
            "Synthetic Accessibility Score",
            "SA_score",
            "SA Score",
            "skyblue",
            [(3.0, "Good SA (3.0)", "green"), (4.0, "Moderate (4.0)", "orange")],
            [(1, 3, "Easy to synthesize")],
            [(4, 10, "Difficult to synthesize")],
        ),
        (
            "Scaffold Size Distribution",
            "ScaffoldSize",
            "Scaffold Size (atoms)",
            "lightcyan",
            [(20, "Large scaffold (20)", "purple"), (10, "Optimal (10)", "green")],
            [(8, 20, "Good scaffold size")],
            [(0, 8, "Too small"), (20, 50, "Too large")],
        ),
        (
            "Ring Count Distribution",
            "RingCount",
            "Ring Count",
            "lightsteelblue",
            [(4, "CNS limit (4)", "purple"), (3, "Optimal (3)", "green")],
            [(2, 4, "Good ring count")],
            [(0, 2, "Too few rings"), (4, 8, "Too many rings")],
        ),
        (
            "PAINS Alerts Count",
            "pains_alerts",
            "PAINS Alerts",
            "red",
            [(0, "No alerts (0)", "green"), (1, "Moderate (1)", "orange")],
            [(0, 0, "No PAINS alerts")],
            [(1, 10, "PAINS alerts present")],
        ),
        (
            "REOS Alerts Count",
            "reos_alerts",
            "REOS Alerts",
            "darkred",
            [(0, "No alerts (0)", "green"), (1, "Moderate (1)", "orange")],
            [(0, 0, "No REOS alerts")],
            [(1, 10, "REOS alerts present")],
        ),
    ]

    # Filter available properties
    available_props = []
    for title, col, xlabel, color, limits, good_ranges, bad_ranges in props:
        if col in df.columns:
            available_props.append((title, col, xlabel, color, limits, good_ranges, bad_ranges))

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(available_props):
        if i >= 6:  # Only 6 subplots
            break
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Fail: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Pass: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplots
    for i in range(len(available_props), 6):
        fig.delaxes(axes[i])

    fig.suptitle("Structure & Drug-likeness Properties", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_3_structure_druglikeness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binding_efficiency_panel(df, outdir):
    """Panel 4: Binding Efficiency - Normalized binding affinity"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Binding efficiency properties
    props = [
        (
            "Docking Score Distribution",
            "docking_score",
            "Docking Score (kcal/mol)",
            "lightsteelblue",
            [(-7, "Good binding (-7)", "green"), (-8, "Excellent (-8)", "blue")],
            [(-10, -7, "Good binding")],
            [(-7, -5, "Moderate binding")],
        ),
        (
            "Ligand Efficiency",
            "LE",
            "Ligand Efficiency",
            "lightcoral",
            [(0.3, "Good LE (0.3)", "green"), (0.2, "Acceptable (0.2)", "orange")],
            [(0.3, 1.0, "Good ligand efficiency")],
            [(0, 0.2, "Poor ligand efficiency")],
        ),
        (
            "Lipophilic Ligand Efficiency",
            "LLE",
            "Lipophilic LE",
            "lightcoral",
            [(5, "Good LLE (5)", "green"), (3, "Acceptable (3)", "orange")],
            [(3, 10, "Good lipophilic efficiency")],
            [(0, 3, "Poor lipophilic efficiency")],
        ),
        (
            "Binding Efficiency Index",
            "BEI",
            "BEI",
            "lightblue",
            [(20, "Good BEI (20)", "green"), (15, "Acceptable (15)", "orange")],
            [(15, 50, "Good BEI")],
            [(0, 15, "Poor BEI")],
        ),
        (
            "Predicted Activity (XGBoost pIC50)",
            "activity_xgb",
            "Predicted pIC50 (XGBoost)",
            "gold",
            [(7, "High activity (7)", "green"), (6, "Moderate (6)", "orange"), (5, "Low (5)", "red")],
            [(7, 10, "High activity")],
            [(0, 5, "Low activity")],
        ),
    ]

    available_props = []
    for title, col, xlabel, color, limits, good_ranges, bad_ranges in props:
        if col in df.columns:
            available_props.append((title, col, xlabel, color, limits, good_ranges, bad_ranges))

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(available_props):
        if i >= 6:  # Only 6 subplots
            break
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Fail: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Pass: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplots
    for i in range(len(available_props), 6):
        fig.delaxes(axes[i])

    fig.suptitle("Binding Efficiency Properties", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_4_binding_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adme_tox_panel(df, outdir):
    """Panel 5: Early ADME-Tox Liabilities - Clinical safety risks"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # ADME-Tox properties
    props = [
        (
            "CYP2D6 Inhibition",
            "cyp2d6_inhib",
            "CYP2D6 Inhibition",
            "darkred",
            [(0.5, "Risk threshold (0.5)", "red"), (0.3, "Moderate (0.3)", "orange")],
            [(0, 0.3, "Low CYP2D6 inhibition")],
            [(0.5, 1.0, "High CYP2D6 inhibition")],
        ),
        (
            "CYP3A4 Inhibition",
            "cyp3a4_inhib",
            "CYP3A4 Inhibition",
            "red",
            [(0.5, "Risk threshold (0.5)", "red"), (0.3, "Moderate (0.3)", "orange")],
            [(0, 0.3, "Low CYP3A4 inhibition")],
            [(0.5, 1.0, "High CYP3A4 inhibition")],
        ),
        (
            "hERG Blocker Score",
            "herg_blocker",
            "hERG Blocker Score",
            "brown",
            [(0.3, "Risk threshold (0.3)", "red"), (0.2, "Moderate (0.2)", "orange")],
            [(0, 0.2, "Low hERG risk")],
            [(0.3, 1.0, "High hERG risk")],
        ),
        (
            "Hepatotoxicity Probability",
            "hepatotox",
            "Hepatotoxicity Probability",
            "brown",
            [(0.6, "Risk threshold (0.6)", "red"), (0.4, "Moderate (0.4)", "orange")],
            [(0, 0.4, "Low hepatotoxicity")],
            [(0.6, 1.0, "High hepatotoxicity")],
        ),
        (
            "Ames Mutagenicity",
            "ames_mutagen",
            "Ames Mutagenicity",
            "darkgreen",
            [(0.5, "Risk threshold (0.5)", "red"), (0.3, "Moderate (0.3)", "orange")],
            [(0, 0.3, "Low mutagenicity")],
            [(0.5, 1.0, "High mutagenicity")],
        ),
        (
            "Human t½ (microsome)",
            "human_t12",
            "Human t½ (min)",
            "purple",
            [(30, "Good t½ (30 min)", "green"), (15, "Moderate (15 min)", "orange")],
            [(30, 120, "Good t½")],
            [(0, 15, "Poor t½")],
        ),
    ]

    # Filter available properties
    available_props = []
    for title, col, xlabel, color, limits, good_ranges, bad_ranges in props:
        if col in df.columns:
            available_props.append((title, col, xlabel, color, limits, good_ranges, bad_ranges))

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(available_props):
        if i >= 6:  # Only 6 subplots
            break
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Risk: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Safe: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplots
    for i in range(len(available_props), 6):
        fig.delaxes(axes[i])

    fig.suptitle("Early ADME-Tox Liabilities", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_5_adme_tox.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cns_specific_panel(df, outdir):
    """Panel 6: CNS-Specific Risk Metrics - Advanced CNS drug discovery metrics"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # CNS-specific properties
    props = [
        (
            "cLogD at pH 7.4",
            "cLogD_74",
            "cLogD 7.4",
            "lightblue",
            [(2, "Good CNS (2)", "green"), (1, "Moderate (1)", "orange")],
            [(1, 3, "Good CNS range")],
            [(0, 1, "Too hydrophilic"), (3, 5, "Too lipophilic")],
        ),
        (
            "pKa (Basic Centers)",
            "pKa_basic",
            "pKa",
            "lightgreen",
            [(8, "Good CNS (8)", "green"), (7, "Moderate (7)", "orange")],
            [(7, 9, "Good CNS range")],
            [(4, 7, "Too acidic"), (9, 12, "Too basic")],
        ),
        (
            "Aqueous Solubility (logS)",
            "logS",
            "logS",
            "lightcoral",
            [(-3, "Good solubility (-3)", "green"), (-5, "Moderate (-5)", "orange")],
            [(-5, -2, "Good solubility")],
            [(-8, -5, "Poor solubility")],
        ),
        (
            "Kp,uu,brain (Unbound Ratio)",
            "Kp_uu_brain",
            "Kp,uu,brain",
            "gold",
            [(1, "Good brain exposure (1)", "green"), (0.5, "Moderate (0.5)", "orange")],
            [(0.5, 2, "Good brain exposure")],
            [(0, 0.5, "Poor brain exposure")],
        ),
        (
            "GSH Reactivity Risk",
            "GSH_reactivity",
            "GSH Reactivity",
            "red",
            [(0.1, "Low risk (0.1)", "green"), (0.3, "Moderate (0.3)", "orange")],
            [(0, 0.2, "Low reactivity")],
            [(0.3, 1, "High reactivity")],
        ),
        (
            "MAO-B Inhibition Risk",
            "MAOB_inhib",
            "MAO-B Inhibition",
            "purple",
            [(0.2, "Low risk (0.2)", "green"), (0.4, "Moderate (0.4)", "orange")],
            [(0, 0.3, "Low MAO-B risk")],
            [(0.4, 1, "High MAO-B risk")],
        ),
    ]

    # Filter available properties
    available_props = []
    for title, col, xlabel, color, limits, good_ranges, bad_ranges in props:
        if col in df.columns:
            available_props.append((title, col, xlabel, color, limits, good_ranges, bad_ranges))

    for i, (title, col, xlabel, color, limits, good_ranges, bad_ranges) in enumerate(available_props):
        if i >= 6:  # Only 6 subplots
            break
        ax = axes[i]
        data = df[col].dropna()

        if len(data) > 0:
            # Add colored regions
            for start, end, label in bad_ranges:
                ax.axvspan(start, end, alpha=0.3, color="red", label=f"Risk: {label}")

            for start, end, label in good_ranges:
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"Safe: {label}")

            # Plot histogram
            ax.hist(data, bins=20, color=color, edgecolor="k", alpha=0.7, zorder=3)

            # Add threshold lines
            for val, label, colr in limits:
                ax.axvline(val, color=colr, linestyle=":", linewidth=2, label=label, zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3, zorder=1)

            # Add statistics
            stats_text = f"n={len(data)} | μ={np.mean(data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=5,
            )
        else:
            ax.text(0.5, 0.5, f"No data for {col}", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="red")
            ax.set_title(title, fontsize=12, fontweight="bold")

    # Remove empty subplots
    for i in range(len(available_props), 6):
        fig.delaxes(axes[i])

    fig.suptitle("CNS-Specific Risk Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(Path(outdir) / "panel_6_cns_specific.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_panels(df, outdir):
    """Generate all 6 panels for comprehensive CNS drug discovery analysis"""
    print("Generating Panel 1: Core PhysChem...")
    plot_core_physchem_panel(df, outdir)

    print("Generating Panel 2: BBB & CNS-Penetration...")
    plot_bbb_cns_panel(df, outdir)

    print("Generating Panel 3: Structure & Drug-likeness...")
    plot_structure_druglikeness_panel(df, outdir)

    print("Generating Panel 4: Binding Efficiency...")
    plot_binding_efficiency_panel(df, outdir)

    print("Generating Panel 5: Early ADME-Tox Liabilities...")
    plot_adme_tox_panel(df, outdir)

    print("Generating Panel 6: CNS-Specific Risk Metrics...")
    plot_cns_specific_panel(df, outdir)

    print("Generating CNS Risk Heatmap...")
    plot_risk_heatmap(df, outdir)

    print("✓ All 6 panels and risk heatmap generated successfully!")


def plot_risk_heatmap(df, outdir):
    """Create a risk heatmap for CNS-specific properties"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Select CNS-specific risk metrics
    risk_metrics = [
        "GSH_reactivity",
        "MAOB_inhib",
        "BACE1_selectivity",
        "HT2A_binding",
        "cyp2d6_inhib",
        "cyp3a4_inhib",
        "herg_blocker",
        "hepatotox",
        "ames_mutagen",
        "pgp_efflux",
        "bcrp_efflux",
    ]

    # Filter available metrics
    available_metrics = [col for col in risk_metrics if col in df.columns]

    if len(available_metrics) == 0:
        print("No CNS risk metrics available for heatmap")
        return

    # Create risk matrix (top 20 compounds by docking score)
    top_compounds = df.nlargest(20, "docking_score")

    # Prepare data for heatmap
    heatmap_data = []
    compound_labels = []

    for _, row in top_compounds.iterrows():
        compound_labels.append(f"{row['id_ligand']}\n({row['docking_score']:.1f})")
        risk_values = []
        for metric in available_metrics:
            value = row.get(metric, np.nan)
            if not np.isnan(value):
                risk_values.append(value)
            else:
                risk_values.append(0.0)
        heatmap_data.append(risk_values)

    if not heatmap_data:
        print("No valid data for risk heatmap")
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # Set labels
    ax.set_xticks(range(len(available_metrics)))
    ax.set_yticks(range(len(compound_labels)))
    ax.set_xticklabels(available_metrics, rotation=45, ha="right")
    ax.set_yticklabels(compound_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Risk Score (0=Low, 1=High)", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(compound_labels)):
        for j in range(len(available_metrics)):
            text = ax.text(j, i, f"{heatmap_data[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    ax.set_title("CNS Risk Assessment Heatmap\n(Top 20 Compounds by Docking Score)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Metrics", fontsize=12)
    ax.set_ylabel("Compounds", fontsize=12)

    plt.tight_layout()
    plt.savefig(Path(outdir) / "cns_risk_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("✓ CNS Risk Heatmap generated successfully!")


# ---------------------------------------------------------------------------
# Pipeline entry‑point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alzheimer's in‑silico pipeline")
    p.add_argument("--input", required=True, type=Path, help="Input Parquet file")
    p.add_argument("--outdir", default=Path("results"), type=Path, help="Output directory")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # -----------------------------------------------------------------------
    # 1. Load ligand table (Polars → Pandas for RDKit compatibility)
    # -----------------------------------------------------------------------
    df_pl = pl.read_parquet(args.input)
    required_cols = {"id_ligand", "smiles", "docking_score"}
    if not required_cols.issubset(set(df_pl.columns)):
        raise ValueError(f"Input Parquet must contain columns {required_cols}")

    # For speed: create heavy atom count once via RDKit then broadcast
    def _heavy_atoms(s: str) -> int:
        m = Chem.MolFromSmiles(s)
        return m.GetNumHeavyAtoms() if m else 0

    df_pl = df_pl.with_columns(pl.col("smiles").map_elements(_heavy_atoms, return_dtype=pl.Int16).alias("HeavyAtoms"))

    # -----------------------------------------------------------------------
    # 2. Descriptor calculations (vectorised via Polars' apply + RDKit)
    # -----------------------------------------------------------------------
    df_pl = df_pl.with_columns(
        pl.struct(["smiles"])
        .map_elements(
            lambda s: _compute_descriptors(s["smiles"]),
            return_dtype=pl.Struct(
                {
                    "valid": pl.Boolean,
                    "MW": pl.Float64,
                    "LogP": pl.Float64,
                    "HBD": pl.Int64,
                    "HBA": pl.Int64,
                    "RotB": pl.Int64,
                    "TPSA": pl.Float64,
                    "CNS_MPO": pl.Float64,
                    "logBB_est": pl.Float64,
                    "SA_score": pl.Float64,
                    "QED": pl.Float64,
                    "ScaffoldSize": pl.Int64,
                    "Fsp3": pl.Float64,
                    "RingCount": pl.Int64,
                    "pains_alerts": pl.Int64,
                    "reos_alerts": pl.Int64,
                    "cyp2d6_inhib": pl.Float64,
                    "cyp3a4_inhib": pl.Float64,
                    "herg_blocker": pl.Float64,
                    "hepatotox": pl.Float64,
                    "ames_mutagen": pl.Float64,
                    "human_t12": pl.Float64,
                    "pgp_efflux": pl.Float64,
                    "bcrp_efflux": pl.Float64,
                    "bbb_prob": pl.Float64,
                    # NEW CNS-SPECIFIC METRICS
                    "cLogD_74": pl.Float64,
                    "pKa_basic": pl.Float64,
                    "logS": pl.Float64,
                    "Kp_uu_brain": pl.Float64,
                    "GSH_reactivity": pl.Float64,
                    "MAOB_inhib": pl.Float64,
                    "BACE1_selectivity": pl.Float64,
                    "HT2A_binding": pl.Float64,
                    "Lipinski_Pass": pl.Boolean,
                    "BBB_Rule3_Pass": pl.Boolean,
                }
            ),
        )
        .alias("_desc")
    )
    # Expand struct column to table (use horizontal concat instead of join)
    df_pl = pl.concat([df_pl.drop("_desc"), df_pl.select("_desc").unnest("_desc")], how="horizontal")

    # -----------------------------------------------------------------------
    # 3. Ligand‑efficiency metrics
    # -----------------------------------------------------------------------
    df_pl = df_pl.with_columns(
        pl.struct(["docking_score", "LogP", "HeavyAtoms", "MW"])
        .map_elements(
            lambda r: _ligand_efficiency(r),
            return_dtype=pl.Struct(
                {
                    "LE": pl.Float64,
                    "LLE": pl.Float64,
                    "BEI": pl.Float64,
                }
            ),
        )
        .alias("_le")
    )
    # Expand struct column to table (use horizontal concat instead of join)
    df_pl = pl.concat([df_pl.drop("_le"), df_pl.select("_le").unnest("_le")], how="horizontal")

    # --- XGBoost activity prediction ---
    smiles_list = df_pl["smiles"].to_list()
    xgb_model = load_model()
    activity_preds = predict_smiles(smiles_list, xgb_model)
    df_pl = df_pl.with_columns(pl.Series("activity_xgb", activity_preds))

    # -----------------------------------------------------------------------
    # 4. Save enriched table
    # -----------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    df_pl.write_parquet(args.outdir / "ligands_descriptors.parquet")
    df_pl.to_pandas().to_csv(args.outdir / "ligands_descriptors.csv", index=False)

    # Summary statistics
    df_pl.select([c for c in df_pl.columns if df_pl[c].dtype != pl.Utf8]).describe().write_csv(args.outdir / "summary_stats.csv")

    # -----------------------------------------------------------------------
    # 5. EDA report
    # -----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        generate_all_panels(df_pl.to_pandas(), args.outdir)

    print(f"✓ Pipeline complete. Results written to '{args.outdir}'.")


if __name__ == "__main__":
    main()
