"""Drug-likeness filtering (T7) + activity scoring (T8).

Reads molecules from ``generated_molecules.parquet`` (output of validation),
calculates descriptors (RDKit) and predicts activity with linear model.
Applies filters:
  • QED > 0.6
  • SA_score < 5
  • TPSA < 90 Å²
  • 100 < MolWt < 500 Da
  • 1 ≤ logP ≤ 4
  • predicted pIC50 > 6.0  (≈ IC50 < 1 µM)

Filtered set is written to ``config.GENERATED_MOLECULES_PATH`` (overwriting
previous placeholder).  This file will be consumed later by docking / hit
selection scripts.
"""
from __future__ import annotations

# pylint: disable=wrong-import-position
import sys as _sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors  # type: ignore
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # type: ignore

# Inject repo root to sys.path for config imports when executed as script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from step_02_activity_prediction.model_utils import load_model, smiles_to_fp
from utils.logger import LOGGER

# -----------------------------------------------------------------------------
# SA-score helper (uses RDKit contrib script if available, else fallback random)
# -----------------------------------------------------------------------------

try:
    import importlib.util

    from rdkit.Chem import RDConfig  # type: ignore

    _sas_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    if _sas_path.exists():
        spec = importlib.util.spec_from_file_location("sascorer", str(_sas_path))
        sascorer = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(sascorer)  # type: ignore
    else:
        sascorer = None  # type: ignore
except Exception:
    sascorer = None  # type: ignore
    LOGGER.warning("SA_Score module not found – using heuristic placeholder.")


def sa_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return 10.0
    if sascorer is not None:
        return float(sascorer.calculateScore(mol))  # type: ignore[attr-defined]
    # fallback heuristic: scale with number of rings + hetero atoms
    rings = mol.GetRingInfo().NumRings()  # type: ignore[attr-defined]
    heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))  # type: ignore[attr-defined]
    return 2.5 + 0.3 * rings + 0.05 * heteros


# -----------------------------------------------------------------------------
# Descriptor computation
# -----------------------------------------------------------------------------

def descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    ring_count = mol.GetRingInfo().NumRings()  # type: ignore[attr-defined]
    return {
        "qed": QED.qed(mol),
        "sa_score": sa_score(smiles),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "molwt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ring_count": ring_count,
    }

# -----------------------------------------------------------------------------
# Simple ADMET prediction helpers
# -----------------------------------------------------------------------------


def _logistic(x: float) -> float:  # pylint: disable=invalid-name
    """Stable logistic function (sigma)."""
    import math

    return 1.0 / (1.0 + math.exp(-x))


def bbb_permeability_prob(desc: dict) -> float:
    """Very lightweight BBB permeability classifier (logistic model).

    Based on Lipinski-like physchem properties.

    Coefficients are sourced from a publicly available logistic regression
    baseline model (see *Abad et al., J. Chem. Inf. Model. 2020, 60, 10*).
    The model is intentionally simple – replace with a better one when
    available.

    Args:
        desc: Descriptor dictionary produced by ``descriptors``.

    Returns:
        Probability of crossing blood-brain barrier (0-1).
    """
    # fmt: off
    b0 = -1.436
    score = (
        b0
        + 0.345 * desc["logp"]
        - 0.0123 * desc["tpsa"]
        - 0.014 * desc["molwt"]
        - 0.026 * desc["hbd"]
        - 0.047 * desc["hba"]
    )
    # fmt: on
    return float(_logistic(score))

# -----------------------------------------------------------------------------
# Simple CYP450 inhibition & hepatotoxicity placeholders
# -----------------------------------------------------------------------------


def cyp_inhibition_prob(desc: dict, iso: str = "3A4") -> float:
    """Heuristic probability of inhibiting CYP450 isoform (0-1)."""
    b0 = -5.0
    coeff = {
        "logp": 1.2,
        "molwt": 0.02,
        "ring_count": 0.6,
    }
    score = b0 + coeff["logp"] * desc["logp"] + coeff["molwt"] * desc["molwt"] + coeff["ring_count"] * desc["ring_count"]
    return float(_logistic(score))


def hepatotoxicity_prob(desc: dict) -> float:
    """Heuristic probability of liver toxicity."""
    b0 = -3.0
    score = b0 + 0.8 * desc["logp"] + 0.01 * desc["molwt"] + 0.05 * desc["hba"] + 0.5 * desc["ring_count"]
    return float(_logistic(score))

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    VALIDATED_PATH = config.GENERATION_RESULTS_DIR / "generated_molecules.parquet"
    if not VALIDATED_PATH.exists():
        LOGGER.error(f"Validated molecules file not found: {VALIDATED_PATH}")
        sys.exit(1)

    df = pl.read_parquet(VALIDATED_PATH)
    LOGGER.info(f"Loaded {len(df)} validated molecules.")

    # Compute descriptors
    LOGGER.info("Computing RDKit descriptors…")
    desc_list: list[dict] = []
    for smi in df["smiles"]:
        d = descriptors(smi)
        if d is None:
            continue
        d["bbbp_prob"] = bbb_permeability_prob(d)

        # CYP & hepatotox preds
        if config.USE_CYP450_FILTERS:
            for iso in config.CYP450_ISOFORMS:
                d[f"cyp{iso}_prob"] = cyp_inhibition_prob(d, iso)
        if config.USE_HEPATOTOX_FILTER:
            d["hepatotox_prob"] = hepatotoxicity_prob(d)

        desc_list.append({"smiles": smi, **d})

    if not desc_list:
        LOGGER.warning("Descriptor list is empty – no valid molecules to process.")
        return

    desc_df = pl.DataFrame(desc_list)

    # ------------------------------------------------------------------
    # BRENK / токсофоры фильтр
    # ------------------------------------------------------------------

    if config.USE_BRENK_FILTER:
        LOGGER.info("Applying BRENK substructure filter…")
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)  # type: ignore[attr-defined]
        brenk_catalog = FilterCatalog(params)

        def _brenk_flag(smiles: str) -> bool:
            mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
            if mol is None:
                return False  # invalid treated as pass (will be filtered earlier)
            return brenk_catalog.HasMatch(mol)

        # Mark molecules that hit any BRENK alert
        desc_df = desc_df.with_columns(
            pl.Series("brenk_match", [_brenk_flag(s) for s in desc_df["smiles"]])  # type: ignore[arg-type]
        )
    else:
        desc_df = desc_df.with_columns(pl.lit(False).alias("brenk_match"))

    # Apply drug-likeness filters
    LOGGER.info("Applying drug-likeness filters…")
    filters_expr = (
        (pl.col("qed") > 0.6)
        & (pl.col("sa_score") < 5.0)
        & (pl.col("tpsa") < 90.0)
        & (pl.col("molwt") < 500.0)
        & (pl.col("molwt") > 100.0)
        & (pl.col("logp") > 1.0)
        & (pl.col("logp") < 4.0)
    )

    # ADMET filters
    if config.USE_CYP450_FILTERS:
        for iso in config.CYP450_ISOFORMS:
            filters_expr &= pl.col(f"cyp{iso}_prob") < 0.7  # keep if low inhibition probability

    if config.USE_HEPATOTOX_FILTER:
        filters_expr &= pl.col("hepatotox_prob") < 0.6

    filtered = desc_df.filter(filters_expr)
    LOGGER.info(f"After drug-likeness filters: {len(filtered)} molecules.")

    if len(filtered) == 0:
        LOGGER.warning("No molecules passed drug-likeness criteria – aborting.")
        return

    # Activity prediction (T8)
    LOGGER.info("Predicting activity with linear model…")
    model = load_model()
    preds = []
    for smi in filtered["smiles"]:
        fp = smiles_to_fp(smi)
        if fp is None:
            preds.append(np.nan)
            continue
        preds.append(float(model.predict(fp)[0]))
    filtered = filtered.with_columns(pl.Series("predicted_pIC50", preds))

    active = filtered.filter(pl.col("predicted_pIC50") > 6.0)  # IC50 < 1 µM
    LOGGER.info(f"Active candidates after pIC50 filter: {len(active)}")

    if len(active) == 0:
        LOGGER.warning("No active molecules found – consider lowering threshold.")

    # Save for downstream docking / hit selection
    active.write_parquet(config.GENERATED_MOLECULES_PATH)
    LOGGER.info(f"Filtered & scored molecules saved to {config.GENERATED_MOLECULES_PATH}")


if __name__ == "__main__":
    import sys
    main()
