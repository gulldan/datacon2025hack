#!/usr/bin/env python
"""Draw CNS-oriented molecules with parameter table
================================================
Creates visualization with:
1. Horizontal row of molecules with scores
2. Parameter table with CNS-focused metrics

Usage:
    python draw_cns_molecules_with_histograms.py --input desirability_results/top5_ranked.parquet --n 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import Draw

# CNS-focused parameters with thresholds
CNS_PARAMS = {
    "MW": (200, 450, "Da"),
    "LogP": (2, 4, ""),
    "TPSA": (40, 90, "Å²"),
    "HBD": (0, 2, ""),
    "HBA": (2, 6, ""),
    "RotB": (0, 8, ""),
    "RingCount": (3, 5, ""),
    "CNS_MPO": (4, 6, ""),
    "QED": (0.35, 1.0, ""),
    "LE": (0.3, 1.0, ""),
    "LLE": (5, 10, ""),
    "docking_score": (-10, -7, "kcal/mol"),
    "logBB_est": (-1, 1, ""),
    "bbb_prob": (0.7, 1.0, ""),
}

# Parameters for table display
TABLE_PARAMS = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "RingCount",
    "CNS_MPO", "QED", "LE", "LLE", "docking_score",
    "logBB_est", "bbb_prob"
]

def get_color_for_cns_param(param, value):
    """Get color based on CNS drug discovery thresholds"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "gray"

    if param in CNS_PARAMS:
        min_val, max_val, unit = CNS_PARAMS[param]

        if param == "docking_score":
            # More negative is better for docking
            return "green" if value <= max_val else "red"
        if param in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "RingCount"]:
            # Range-based parameters
            return "green" if min_val <= value <= max_val else "red"
        if param in ["CNS_MPO", "QED", "LE", "LLE", "logBB_est", "bbb_prob"]:
            # Higher is better
            return "green" if value >= min_val else "red"

    return "black"

def draw_mols_horizontal(df, n, out_file):
    """Draw horizontal row of molecules"""
    # Sort by Composite_Score in descending order
    df_sorted = df.sort("Composite_Score", descending=True)

    mols = [Chem.MolFromSmiles(smi) for smi in df_sorted["smiles"][:n]]
    legends = [f"{row['id_ligand']}\nScore: {row['Composite_Score']:.3f}"
               for row in df_sorted.head(n).iter_rows(named=True)]

    img = Draw.MolsToGridImage(mols, molsPerRow=n, subImgSize=(300, 300),
                              legends=legends, useSVG=False)
    img.save(out_file)
    print(f"Saved: {out_file}")

def draw_params_table(df, n, out_file):
    """Draw parameter table with CNS-focused metrics"""
    # Sort by Composite_Score in descending order
    df_sorted = df.sort("Composite_Score", descending=True)
    top = df_sorted.head(n)
    data = []
    colors = []

    for row in top.iter_rows(named=True):
        row_data = []
        row_colors = []
        for param in TABLE_PARAMS:
            val = row.get(param, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                sval = "NA"
            elif param in ["MW", "TPSA"]:
                sval = f"{val:.1f}"
            elif param in ["LogP", "QED", "LE", "LLE", "logBB_est", "bbb_prob"] or param == "docking_score":
                sval = f"{val:.2f}"
            elif param in ["HBD", "HBA", "RotB", "RingCount", "CNS_MPO"]:
                sval = f"{int(val)}"
            else:
                sval = str(val)

            row_data.append(sval)
            row_colors.append(get_color_for_cns_param(param, val))
        data.append(row_data)
        colors.append(row_colors)

    # Create table
    col_labels = [f"{param}" for param in TABLE_PARAMS]
    row_labels = [row["id_ligand"] for row in top.iter_rows(named=True)]

    fig, ax = plt.subplots(figsize=(2+1.1*len(TABLE_PARAMS), 1+n*0.7))
    table = ax.table(cellText=data, rowLabels=row_labels, colLabels=col_labels,
                     cellColours=[[c for c in row] for row in colors],
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.9, 1.1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file}")



def main():
    parser = argparse.ArgumentParser(description="Draw CNS-oriented molecules with histograms")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--n", default=5, type=int)
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    print(f"Loaded {len(df)} molecules")

    # Create all visualizations
    draw_mols_horizontal(df, args.n, f"top{args.n}_cns_mols.png")
    draw_params_table(df, args.n, f"top{args.n}_cns_params.png")

    print("✓ All CNS-focused visualizations complete!")

if __name__ == "__main__":
    main()
