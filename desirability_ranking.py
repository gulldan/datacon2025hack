#!/usr/bin/env python
"""Desirability-based ranking for CNS drug discovery
==================================================
Implements geometric mean desirability scoring with weighted contributions
from CNS-penetration, safety, binding efficiency, and physicochemical properties.

Usage:
    python desirability_ranking.py --input results_new_metrics/ligands_descriptors.parquet --output top5_ranked.parquet
"""

import argparse
from pathlib import Path

import polars as pl


def desir_less(x: pl.Series, t: float, h: float) -> pl.Expr:
    """Desirability function for 'less is better' metrics"""
    return pl.when(x <= t).then(1.0).otherwise(
        pl.when(x >= h).then(0.0).otherwise((h - x) / (h - t))
    )


def desir_more(x: pl.Series, l: float, t: float) -> pl.Expr:
    """Desirability function for 'more is better' metrics"""
    return pl.when(x >= t).then(1.0).otherwise(
        pl.when(x <= l).then(0.0).otherwise((x - l) / (t - l))
    )


def desir_target(x: pl.Series, l: float, opt: float, h: float) -> pl.Expr:
    """Desirability function for 'target is best' metrics (bell curve)"""
    return pl.when(x <= opt).then(
        desir_more(x, l, opt)
    ).otherwise(
        desir_less(x, opt, h)
    )


def calculate_desirability_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate desirability scores for all metrics"""
    # Create desirability DataFrame
    d_scores = pl.DataFrame()

    # CNS-penetration block (weight = 3)
    d_scores = d_scores.with_columns([
        desir_more(df["CNS_MPO"], 3, 5).alias("d_mpo"),
        desir_more(df["logBB_est"], -1.0, -0.3).alias("d_logBB"),
        desir_less(df["pgp_efflux"], 1, 2).alias("d_pgp"),
        desir_less(df["bcrp_efflux"], 1, 2).alias("d_bcrp"),
        desir_more(df["Kp_uu_brain"], 0.3, 0.8).alias("d_kp_brain"),
    ])

    # PhysChem block (weight = 1)
    d_scores = d_scores.with_columns([
        desir_target(df["MW"], 250, 350, 400).alias("d_mw"),
        desir_target(df["LogP"], 1, 2.8, 4).alias("d_logp"),
        desir_target(df["cLogD_74"], 1, 2.5, 3.5).alias("d_clogd"),
        desir_target(df["TPSA"], 40, 60, 80).alias("d_tpsa"),
        desir_less(df["HBD"], 0, 2).alias("d_hbd"),
        desir_less(df["HBA"], 2, 7).alias("d_hba"),
        desir_less(df["RotB"], 0, 7).alias("d_rotb"),
    ])

    # Binding-efficiency block (weight = 2)
    d_scores = d_scores.with_columns([
        desir_more(-df["docking_score"], 6, 8).alias("d_dock"),  # Negative for "more is better"
        desir_more(df["LE"], 0.2, 0.35).alias("d_le"),
        desir_more(df["LLE"], 3, 5).alias("d_lle"),
        desir_more(df["BEI"], 15, 25).alias("d_bei"),
    ])

    # Safety block (weight = 2)
    d_scores = d_scores.with_columns([
        desir_less(df["herg_blocker"], 0.2, 0.3).alias("d_herg"),
        desir_less(df["cyp3a4_inhib"], 0.3, 0.5).alias("d_cyp3a4"),
        desir_less(df["cyp2d6_inhib"], 0.3, 0.5).alias("d_cyp2d6"),
        desir_less(df["hepatotox"], 0.2, 0.3).alias("d_hepatotox"),
        desir_less(df["ames_mutagen"], 0.2, 0.3).alias("d_ames"),
        desir_less(df["GSH_reactivity"], 0.1, 0.2).alias("d_gsh"),
    ])

    # Selectivity/PD block (weight = 1)
    d_scores = d_scores.with_columns([
        desir_less(df["MAOB_inhib"], 0.2, 0.3).alias("d_maob"),
        desir_less(df["HT2A_binding"], 0.2, 0.3).alias("d_ht2a"),
        desir_less(df["BACE1_selectivity"], 0.3, 0.5).alias("d_bace1"),
    ])

    return d_scores


def calculate_composite_score(df: pl.DataFrame, d_scores: pl.DataFrame) -> pl.DataFrame:
    """Calculate composite score using geometric mean with weights"""
    import numpy as np

    # Define weights for each block
    weights = {
        # CNS-penetration block (weight = 3)
        "d_mpo": 3, "d_logBB": 3, "d_pgp": 3, "d_bcrp": 3, "d_kp_brain": 3,
        # PhysChem block (weight = 1)
        "d_mw": 1, "d_logp": 1, "d_clogd": 1, "d_tpsa": 1, "d_hbd": 1, "d_hba": 1, "d_rotb": 1,
        # Binding-efficiency block (weight = 2)
        "d_dock": 2, "d_le": 2, "d_lle": 2, "d_bei": 2,
        # Safety block (weight = 2)
        "d_herg": 2, "d_cyp3a4": 2, "d_cyp2d6": 2, "d_hepatotox": 2, "d_ames": 2, "d_gsh": 2,
        # Selectivity/PD block (weight = 1)
        "d_maob": 1, "d_ht2a": 1, "d_bace1": 1,
    }

    # Calculate total weight
    total_weight = sum(weights.values())

    # Convert to numpy for easier calculation
    d_scores_np = d_scores.to_numpy()

    # Calculate log scores with small epsilon to avoid log(0)
    log_scores = np.log(d_scores_np + 1e-9)

    # Apply weights
    weighted_log_scores = np.zeros(len(df))
    for i, col in enumerate(d_scores.columns):
        if col in weights:
            weighted_log_scores += log_scores[:, i] * weights[col]

    # Calculate final score
    composite_scores = np.exp(weighted_log_scores / total_weight)

    # Add scores to original DataFrame
    result_df = df.with_columns([
        pl.Series("Composite_Score", composite_scores)
    ])

    # Add desirability scores
    result_df = result_df.with_columns(d_scores.select(pl.all()))

    return result_df


def generate_top5_report(df: pl.DataFrame, output_dir: Path):
    """Generate Top-5 report with structures and scores"""
    import numpy as np
    def fmt(val, fmtstr):
        if val is None:
            return "NA"
        try:
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return "NA"
            return fmtstr.format(val)
        except Exception:
            return str(val)
    # Get top 5 compounds
    top5 = df.sort("Composite_Score", descending=True).head(5)

    print("\n" + "="*60)
    print("TOP-5 COMPOUNDS BY DESIRABILITY SCORE")
    print("="*60)

    for i, row in enumerate(top5.iter_rows(named=True), 1):
        print(f"\n{i}. {row['id_ligand']}")
        print(f"   SMILES: {row['smiles']}")
        print(f"   Composite Score: {fmt(row['Composite_Score'], '{:.4f}')}" )
        print(f"   Docking Score: {fmt(row['docking_score'], '{:.2f}')}" )
        print(f"   CNS MPO: {fmt(row['CNS_MPO'], '{:.2f}')}" )
        print(f"   MW: {fmt(row['MW'], '{:.1f}')} Da")
        print(f"   LogP: {fmt(row['LogP'], '{:.2f}')}" )
        print(f"   LE: {fmt(row['LE'], '{:.3f}')}" )
        print(f"   LLE: {fmt(row['LLE'], '{:.2f}')}" )
        print(f"   hERG Risk: {fmt(row['herg_blocker'], '{:.3f}')}" )
        print(f"   CYP3A4 Risk: {fmt(row['cyp3a4_inhib'], '{:.3f}')}" )

    # Save top 5 to CSV
    top5_csv = output_dir / "top5_compounds.csv"
    top5.write_csv(top5_csv)
    print(f"\nTop-5 compounds saved to: {top5_csv}")

    # Generate desirability breakdown for top 5
    print("\n" + "="*60)
    print("DESIRABILITY BREAKDOWN FOR TOP-5")
    print("="*60)

    d_columns = [col for col in top5.columns if col.startswith("d_")]
    d_columns.sort()

    for i, row in enumerate(top5.iter_rows(named=True), 1):
        print(f"\nCompound {i} ({row['id_ligand']}):")
        for col in d_columns:
            metric_name = col[2:]  # Remove 'd_' prefix
            value = row[col]
            print(f"  {metric_name:12}: {fmt(value, '{:.3f}')}")

    return top5


def main():
    parser = argparse.ArgumentParser(description="Desirability-based ranking for CNS compounds")
    parser.add_argument("--input", required=True, type=Path, help="Input Parquet file")
    parser.add_argument("--output", default=Path("top5_ranked.parquet"), type=Path, help="Output file")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.input}")
    df = pl.read_parquet(args.input)
    print(f"Loaded {len(df)} compounds")

    # Filter out molecules with missing key metrics
    print("Filtering molecules with missing key metrics...")
    df_filtered = df.filter(
        (~pl.col("docking_score").is_null()) &
        (~pl.col("LE").is_null()) &
        (~pl.col("LLE").is_null()) &
        (~pl.col("BEI").is_null())
    )
    print(f"After filtering: {len(df_filtered)} compounds")

    # Calculate desirability scores
    print("Calculating desirability scores...")
    d_scores = calculate_desirability_scores(df_filtered)

    # Calculate composite score
    print("Calculating composite scores...")
    result_df = calculate_composite_score(df_filtered, d_scores)

    # Generate top 5 report
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    top5 = generate_top5_report(result_df, output_dir)

    # Save full ranked dataset
    result_df.write_parquet(args.output)
    print(f"\nFull ranked dataset saved to: {args.output}")

    # Save top 5 to separate file
    top5_parquet = output_dir / "top5_ranked.parquet"
    top5.write_parquet(top5_parquet)
    print(f"Top-5 compounds saved to: {top5_parquet}")

    print("\n" + "="*60)
    print("RANKING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
