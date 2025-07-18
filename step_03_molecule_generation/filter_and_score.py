"""Drug-likeness annotation (T7) + activity scoring (T8).

Reads molecules from ``generated_molecules.parquet``, calculates all relevant
descriptors and predicts activity. This version ANNOTATES all molecules with
these properties but DOES NOT filter them. It only logs how many molecules
would have passed the filters.

The full, annotated set is written to ``config.GENERATED_MOLECULES_PATH``,
overwriting the previous file. This file will be consumed later by docking /
hit selection scripts.
"""

from __future__ import annotations

# pylint: disable=wrong-import-position
import sys
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

    # Enhanced heuristic for synthetic accessibility
    # Lower score = more synthetically accessible
    rings = mol.GetRingInfo().NumRings()  # type: ignore[attr-defined]
    heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))  # type: ignore[attr-defined]

    # Basic complexity factors
    num_atoms = mol.GetNumAtoms()  # type: ignore[attr-defined]
    num_bonds = mol.GetNumBonds()  # type: ignore[attr-defined]

    # Stereochemistry complexity
    stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))  # type: ignore[attr-defined]

    # Aromatic ring complexity
    aromatic_rings = sum(1 for ring in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))  # type: ignore[attr-defined]

    # Calculate base score (aim for realistic range 1-6)
    base_score = 1.5  # Start with low base for simple molecules

    # Add complexity penalties
    base_score += 0.4 * rings  # Ring complexity
    base_score += 0.3 * heteros / max(1, num_atoms)  # Heteroatom density
    base_score += 0.5 * stereo_centers  # Stereochemistry
    base_score += 0.2 * aromatic_rings  # Aromatic complexity

    # Size penalty for very large molecules
    if num_atoms > 30:
        base_score += 0.1 * (num_atoms - 30) / 10

    # Clamp to reasonable range
    return max(1.0, min(6.0, base_score))


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
    """Very lightweight BBB permeability classifier (logistic model)."""
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
# Main pipeline (REVISED LOGIC)
# -----------------------------------------------------------------------------


def main() -> None:
    VALIDATED_PATH = config.GENERATION_RESULTS_DIR / "generated_molecules.parquet"
    if not VALIDATED_PATH.exists():
        LOGGER.error(f"Validated molecules file not found: {VALIDATED_PATH}")
        sys.exit(1)

    df = pl.read_parquet(VALIDATED_PATH)
    LOGGER.info(f"Loaded {len(df)} validated molecules.")

    # --- Шаг 1: Вычисление всех дескрипторов и предсказаний для всех молекул ---
    LOGGER.info("Computing RDKit descriptors and ADMET properties…")
    desc_list: list[dict] = []
    for smi in df["smiles"]:
        d = descriptors(smi)
        if d is None:
            continue
        d["bbbp_prob"] = bbb_permeability_prob(d)

        if config.USE_CYP450_FILTERS:
            for iso in config.CYP450_ISOFORMS:
                d[f"cyp{iso}_prob"] = cyp_inhibition_prob(d, iso)
        if config.USE_HEPATOTOX_FILTER:
            d["hepatotox_prob"] = hepatotoxicity_prob(d)

        desc_list.append({"smiles": smi, **d})

    if not desc_list:
        LOGGER.warning("Descriptor list is empty – no valid molecules to process.")
        return

    # Создаем DataFrame со всеми вычисленными свойствами
    annotated_df = pl.DataFrame(desc_list)

    # --- Шаг 2: Добавление BRENK фильтра (как в оригинале) ---
    if config.USE_BRENK_FILTER:
        LOGGER.info("Applying BRENK substructure filter…")
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        brenk_catalog = FilterCatalog(params)

        def _brenk_flag(smiles: str) -> bool:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return brenk_catalog.HasMatch(mol)

        annotated_df = annotated_df.with_columns(pl.Series("brenk_match", [_brenk_flag(s) for s in annotated_df["smiles"]]))
    else:
        annotated_df = annotated_df.with_columns(pl.lit(False).alias("brenk_match"))

    # --- Шаг 3: Предсказание активности для ВСЕХ молекул ---
    LOGGER.info("Predicting activity with linear model…")
    model = load_model()
    preds = []
    for smi in annotated_df["smiles"]:
        # Use correct fingerprint size based on model type
        if hasattr(model, "coeffs") and len(model.coeffs()) > 0:
            # Linear model - use FP_BITS_LINEAR
            fp = smiles_to_fp(smi, n_bits=config.FP_BITS_LINEAR)
        else:
            # XGBoost model - use FP_BITS_XGB
            fp = smiles_to_fp(smi, n_bits=config.FP_BITS_XGB)
        if fp is None:
            preds.append(np.nan)
            continue
        preds.append(float(model.predict(fp)[0]))

    annotated_df = annotated_df.with_columns(pl.Series("predicted_pIC50", preds))

    LOGGER.info("--- End of Simulation ---")

    # --- Шаг 4: Сохранение полного, аннотированного файла ---
    # Сохраняем DataFrame со ВСЕМИ молекулами и ВСЕМИ вычисленными свойствами
    output_path = config.GENERATED_MOLECULES_PATH
    annotated_df.write_parquet(output_path)

    LOGGER.info(f"Process complete. All {len(annotated_df)} molecules were annotated with properties.")
    LOGGER.info(f"Full dataset saved to {output_path}")
    LOGGER.info("This file is now ready for the docking stage.")


if __name__ == "__main__":
    main()
