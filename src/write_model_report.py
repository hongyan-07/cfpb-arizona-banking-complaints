"""Append model findings generated from the latest backtest and test outputs."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT = PROJECT_DIR / "report/executive_findings.md"
MODEL_DIR = PROJECT_DIR / "analysis/modeling"
HEADING = "## Forecasting and relief-likelihood modeling"


def main() -> None:
    forecast = json.loads((MODEL_DIR / "forecast_results.json").read_text())
    relief = json.loads((MODEL_DIR / "relief_model_results.json").read_text())
    methods = {row["method"]: row for row in forecast["total_series_metrics"]}
    best = methods[forecast["best_method_total"]]
    selected = relief["models"][relief["selected_model"]]
    baseline = relief["models"]["logistic_regression_baseline"]["test"]
    test = selected["test"]

    descriptive = REPORT.read_text(encoding="utf-8").split(HEADING)[0].rstrip()
    appendix = f"""

{HEADING}

- **Weekly workload forecast:** Across {forecast['backtest_weeks']} rolling weekly origins, the 8-week median achieved {best['mae']:.2f} complaints/week MAE, {best['mape_pct']:.1f}% MAPE, and {best['wape_pct']:.1f}% WAPE. This was {forecast['best_mae_reduction_vs_naive_pct']:.1f}% lower MAE than the last-week benchmark. The lagged Ridge model did not beat the median.
- **Relief-likelihood triage:** On the held-out Jul-Dec 2025 period, the selected XGBoost classifier achieved F1 {test['f1']:.2f}, recall {test['recall']:.2f}, and PR-AUC {test['pr_auc']:.2f}, versus F1 {baseline['f1']:.2f} for logistic regression. The top-scored 20% contained {test['relief_captured_in_top_20pct']:.0%} of complaints recorded as ending in relief.
- Model selection and threshold tuning used Apr-Jun 2025 validation data. Jul-Dec 2025 test data was held out until evaluation. Results are specific to this fixed CFPB snapshot and pinned software versions.
- These are research results. Company-reported relief is not an independent measure of consumer harm. ZIP3 and Older American / Servicemember tags may create fairness concerns; any operational design should assess them and remove or restrict them before deployment.
"""
    REPORT.write_text(descriptive + appendix, encoding="utf-8")


if __name__ == "__main__":
    main()
