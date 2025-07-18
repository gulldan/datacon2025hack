#!/usr/bin/env bash
# run.sh – one-click execution of the full DYRK1A virtual-screening pipeline.
#
# Usage:
#   bash run.sh        # or ./run.sh (after chmod +x run.sh)
#
# Prerequisites:
#   • Homebrew/apt-get installed OpenBabel + AutoDock Vina binaries.
#   • "uv" tool installed (https://github.com/astral-sh/uv).
#     On macOS:  brew install astral-sh/astral/uv
#     On Linux :  curl -Ls https://astral.sh/uv/install.sh | bash
#   • On Arch   :  paru -S openbabel autodock-vina
#
# The script is idempotent: it creates a Python 3.11 virtualenv in .venv on the
# first run and reuses it next time. All project artefacts are written to the
# step_X/results/ and data/ folders as defined in config.py.
#
set -euo pipefail

CYAN="\033[1;36m"
GREEN="\033[1;32m"
RESET="\033[0m"

step() {
  echo -e "${CYAN}\n==== $* ====${RESET}"
}

# ---------------------------------------------------------------------------
# 0. Check that we are in project root (file config.py must exist here)
# ---------------------------------------------------------------------------
if [[ ! -f "config.py" ]]; then
  echo "ERROR: run.sh must be executed from the repository root directory." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Ensure uv is available
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it first (see header)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create & activate virtual environment (.venv)
# ---------------------------------------------------------------------------
step "Setting up Python virtual environment (.venv)"
if [[ ! -d ".venv" ]]; then
  uv sync -U
fi
source .venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# ---------------------------------------------------------------------------
# 4. Run pipeline steps sequentially
# ---------------------------------------------------------------------------
# step "Step 1 – Target selection report"
# uv run python step_01_target_selection/run_target_analysis.py

step "Step 2 – Activity data collection + QSAR model (scaffold split)"
uv run python step_02_activity_prediction/data_collection.py
uv run python step_02_activity_prediction/run_descriptor_calc.py
uv run python step_02_activity_prediction/feature_selection.py
uv run python step_02_activity_prediction/train_activity_model_scaffold.py
# дополнительная сложная модель XGBoost (GPU)
uv run python step_02_activity_prediction/train_activity_model_xgb.py

step "Step 3 – Molecule generation & filtering"
uv run python step_03_molecule_generation/run_generation.py
uv run python step_03_molecule_generation/validate_generated.py
uv run python step_03_molecule_generation/filter_and_score.py

step "Step 4 – Protein & ligand preparation for docking"
uv run python step_04_hit_selection/protein_prep.py
uv run python step_04_hit_selection/ligand_prep.py

step "Step 4 – GPU-accelerated docking and hit selection"
uv run python step_04_hit_selection/run_vina.py
uv run python analize_results.py --input step_04_hit_selection/docking/docking_results.parquet --outdir step_04_hit_selection/results/panels
uv run python desirability_ranking.py --input ./step_04_hit_selection/results/panels/ligands_descriptors.parquet --output step_04_hit_selection/results/top5_desirability_ranked.parquet
uv run python draw_molecules.py --input step_04_hit_selection/results/top5_desirability_ranked.parquet --n 5
echo -e "${GREEN}\nAll done!${RESET}" 