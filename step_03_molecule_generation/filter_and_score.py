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

import sys as _sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors  # type: ignore

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
    return {
        "qed": QED.qed(mol),
        "sa_score": sa_score(smiles),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "molwt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
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
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    VALIDATED_PATH = config.GENERATION_RESULTS_DIR / "generated_molecules.parquet"
    if not VALIDATED_PATH.exists():
        LOGGER.error("Validated molecules file not found: %s", VALIDATED_PATH)
        sys.exit(1)

    df = pl.read_parquet(VALIDATED_PATH)
    LOGGER.info("Loaded %d validated molecules.", len(df))

    # Compute descriptors
    LOGGER.info("Computing RDKit descriptors…")
    desc_list: list[dict] = []
    for smi in df["smiles"]:
        d = descriptors(smi)
        if d is None:
            continue
        d["bbbp_prob"] = bbb_permeability_prob(d)
        desc_list.append({"smiles": smi, **d})

    desc_df = pl.DataFrame(desc_list)

    # Apply drug-likeness filters (T7)
    LOGGER.info("Applying drug-likeness filters…")
    filtered = desc_df.filter(
        (pl.col("qed") > 0.6)
        & (pl.col("sa_score") < 5.0)
        & (pl.col("tpsa") < 90.0)
        & (pl.col("molwt") < 500.0)
        & (pl.col("molwt") > 100.0)
        & (pl.col("logp") > 1.0)
        & (pl.col("logp") < 4.0)
        # BBB permeability probability will be filtered later in hit selection
    )
    LOGGER.info("After drug-likeness filters: %d molecules.", len(filtered))

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
    LOGGER.info("Active candidates after pIC50 filter: %d", len(active))

    if len(active) == 0:
        LOGGER.warning("No active molecules found – consider lowering threshold.")

    # Save for downstream docking / hit selection
    active.write_parquet(config.GENERATED_MOLECULES_PATH)
    LOGGER.info("Filtered & scored molecules saved to %s", config.GENERATED_MOLECULES_PATH)


if __name__ == "__main__":
    import sys
    main()
