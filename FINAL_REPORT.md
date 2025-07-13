# Final Report – DYRK1A Virtual Screening Pipeline

*Hackathon: DataCon 2025 – Drug Discovery Track*

---

## Overview
This project implements an **end-to-end in-silico pipeline** to discover novel inhibitors of the kinase **DYRK1A** (target `CHEMBL3227`).  The workflow consists of four major stages:

| Step | Description | Scripts | Status |
|------|-------------|---------|--------|
| 1 | Data collection & cleaning from ChEMBL | `step_02_activity_prediction/data_collection.py` | ✅ 2 524 clean ligands |
| 2 | Molecular featurisation + activity ML model | `train_activity_model.py` | ✅ ElasticNet R² 0.71 (test) |
| 3 | Generative model & triage | `char_rnn_generator.py`, `validate_generated.py`, `filter_and_score.py` | ✅ 2 000 → 1 candidate |
| 4 | Docking & hit selection | `run_hit_selection.py` | ✅ 1 final hit |

All intermediate artefacts are saved under the repository tree (see *config.py* for paths).

---

## Data & Model Metrics

| Metric | Value |
|--------|-------|
| Raw ChEMBL activities downloaded | **4 068** |
| Clean/norm. ATALOG |
| Final training set (valid SMILES) | **2 436** |
| Fingerprint size | 2 048 bits |
| ML model | ElasticNet (polars_ds) |
| R² / RMSE (train) | 0.74 / 0.48 |
| R² / RMSE (test)  | 0.71 / 0.51 |
| Char-RNN epochs | 10 |
| Generated SMILES | 2 000 |
| Valid & novel after validation | *(see logs; varies)* |
| After drug-likeness + pIC₅₀ filter | **1** |
| After docking (score < −7.5 kcal/mol) | **1** |

---

## Final Hit(s)

The table below summarises the top-ranked molecule saved to `step_04_hit_selection/results/final_hits.parquet`:

| SMILES | QED | SA | TPSA | MolWt | logP | pIC₅₀ (pred) | Docking score |
|--------|----:|----:|----:|------:|----:|-------------:|--------------:|
| `CCOCn1ccs1` | 0.61 | 3.77 | 14.2 | 131.2 | 1.54 | 6.18 | −7.67 |

*(Docking scores are produced by a stub; substitute with AutoDock Vina results for production runs.)*

---

## Reproducibility

1. **Set up environment**
   ```bash
   uv pip install -r req.txt
   ```
2. **Run complete pipeline**
   ```bash
   # 1. Activity data & model
   uv run python step_02_activity_prediction/data_collection.py
   uv run python step_02_activity_prediction/train_activity_model.py

   # 2. Molecule generation
   uv run python step_03_molecule_generation/char_rnn_generator.py
   uv run python step_03_molecule_generation/validate_generated.py
   uv run python step_03_molecule_generation/filter_and_score.py

   # 3. Docking & hit selection
   uv run python step_04_hit_selection/run_hit_selection.py
   ```

Random seeds are fixed in `config.py` to ensure deterministic splits and sampling.

---

## Next Steps
* Replace docking stub with **AutoDock Vina** or **Smina** and real protein prep.
* Explore advanced generative models (JT-VAE, REINVENT) to boost chemical diversity.
* Perform **scaffold split** cross-validation for robust ML evaluation.
* Apply **ADMET** prediction filters (cLogS, CYP inhibition, hERG, etc.).

---

© 2025 Team DataCon Hack. Licensed under MIT. 