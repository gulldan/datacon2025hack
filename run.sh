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
uv run python step_02_activity_prediction/train_activity_model_scaffold.py

step "Step 3 – Molecule generation & filtering (SELFIES-VAE)"
uv run python step_03_molecule_generation/run_generation.py
uv run python step_03_molecule_generation/validate_generated.py
uv run python step_03_molecule_generation/filter_and_score.py

step "Step 4 – Protein & ligand preparation for docking"
uv run python step_04_hit_selection/protein_prep.py
uv run python step_04_hit_selection/ligand_prep.py

step "Step 4 – Docking with AutoDock Vina"
# Vina may be absent; fall back to stub scores if it fails
if uv run python step_04_hit_selection/run_vina.py; then
  echo -e "${GREEN}Vina docking completed successfully.${RESET}"
else
  echo "WARNING: AutoDock Vina failed or not installed – stub scores will be used."
fi

step "Step 4 – Hit selection"
uv run python step_04_hit_selection/run_hit_selection.py

step "Pipeline completed. Review the following artefacts:"
echo "  • step_02_activity_prediction/results/metrics_scaffold.json"
echo "  • step_03_molecule_generation/results/generated_molecules.parquet"
echo "  • step_04_hit_selection/results/final_hits.parquet"

echo -e "${GREEN}\nAll done!${RESET}" 