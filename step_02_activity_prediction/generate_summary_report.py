"""Generate HTML summary report comparing activity models and descriptor pipeline (T31).

Outputs ``results/summary_report.html`` with:
    • Table of metrics for ElasticNet (scaffold split) and XGBoost variants.
    • Interactive bar chart of RMSE / R².
    • Feature importance plot for XGBoost (gain) and top-weighted features for ElasticNet.

Dependencies: plotly, pandas, polars, rdkit.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb  # type: ignore

import config

# Paths
RESULTS_DIR = config.PREDICTION_RESULTS_DIR
HTML_OUT = RESULTS_DIR / "summary_report.html"


def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []

    # ElasticNet (scaffold)
    path_sc = RESULTS_DIR / "metrics_scaffold.json"
    if path_sc.exists():
        with open(path_sc, encoding="utf-8") as f:
            data = json.load(f)
        rows.append({"model": "ElasticNet (scaffold)", **data})

    # XGBoost default
    path_xgb = RESULTS_DIR / "metrics_xgb.json"
    if path_xgb.exists():
        with open(path_xgb, encoding="utf-8") as f:
            data = json.load(f)
        rows.append({"model": "XGBoost (GPU)", **data})

    # Optuna best
    path_opt = RESULTS_DIR / "optuna_xgb_best.json"
    if path_opt.exists():
        with open(path_opt, encoding="utf-8") as f:
            best = json.load(f)
        if path_xgb.exists():
            # reuse test n_train/n_test from default metrics
            with open(path_xgb, encoding="utf-8") as f:
                base = json.load(f)
        else:
            base = {"n_train": 0, "n_test": 0}
        rows.append({
            "model": "XGBoost (Optuna)",
            "rmse_test": best.get("rmse"),
            "r2_test": None,
            **{k: base.get(k) for k in ("n_train", "n_test")}
        })

    if not rows:
        raise FileNotFoundError("No metrics JSON files found in results directory.")

    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["model"], y=df["rmse_test"], name="RMSE (test)", marker_color="indianred"))
    if "r2_test" in df.columns:
        fig.add_trace(go.Bar(x=df["model"], y=df["r2_test"], name="R² (test)", marker_color="steelblue"))

    fig.update_layout(title="Model performance on test set", barmode="group", yaxis_title="Metric value")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def xgb_importance_plot() -> str | None:
    if not config.XGB_MODEL_PATH.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(config.XGB_MODEL_PATH))
    imp = booster.get_score(importance_type="gain")  # dict feature->gain
    if not imp:
        return None
    # Take top 15 features
    sorted_items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
    labels, gains = zip(*sorted_items, strict=False)
    fig = go.Figure(go.Bar(x=gains, y=labels, orientation="h", marker_color="darkgreen"))
    fig.update_layout(title="XGBoost top 15 feature importances (gain)", yaxis_title="Feature", xaxis_title="Gain")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_html(df: pd.DataFrame, metric_html: str, xgb_imp_html: str | None):
    table_html = df.to_html(index=False, float_format="{:.3f}".format)

    parts = [
        "<html><head><title>Model Summary Report</title><meta charset='utf-8'></head><body>",
        "<h1>Model Summary Report</h1>",
        "<h2>Metrics</h2>",
        table_html,
        "<h2>Performance plots</h2>",
        metric_html,
    ]
    if xgb_imp_html:
        parts.extend(["<h2>XGBoost Feature Importance</h2>", xgb_imp_html])

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    df = load_metrics()
    metric_html = plot_metrics(df)
    xgb_imp_html = xgb_importance_plot()
    html = build_html(df, metric_html, xgb_imp_html)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Summary report written to {HTML_OUT}")


if __name__ == "__main__":
    main()
