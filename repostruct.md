datacon2025hack/
│
├── step_01_target_selection/
│   ├── run_target_analysis.py
│   └── reports/
│       └── target_selection_report.md
│
├── step_02_activity_prediction/
│   ├── run_prediction_model.py
│   └── results/
│       ├── eda_plots.html
│       ├── model.joblib
│       └── feature_importance.html
│
├── step_03_molecule_generation/
│   ├── run_generation.py
│   └── results/
│       └── generated_molecules.parquet
│
├── step_04_hit_selection/
│   ├── run_hit_selection.py
│   └── results/
│       └── final_hits.parquet
│
├── data/
│   ├── raw/
│   └── processed/
│
├── utils/
│   ├── logger.py
│   └── docking.py
│
├── config.py
├── main.py
└── requirements.txt