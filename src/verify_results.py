"""Check the frozen input and the resume-facing claims after a full rebuild."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sqlite3
import sys
from importlib.metadata import version
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data/raw/cfpb_az_banking_complaints_2024_2025.csv"
RAW_SHA256 = "9525db0ace5e65557f8c4a3ddb237383c555fb8db87e62f5a45f85e6224045b1"
ANALYSIS = PROJECT_DIR / "analysis"
MODEL = ANALYSIS / "modeling"


def check(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)


def main() -> None:
    digest = hashlib.sha256(RAW_FILE.read_bytes()).hexdigest()
    check(digest == RAW_SHA256, "Raw CSV differs from the documented snapshot")

    findings = json.loads((ANALYSIS / "key_findings.json").read_text())
    quality = json.loads((ANALYSIS / "data_quality_report.json").read_text())
    forecast = json.loads((MODEL / "forecast_results.json").read_text())
    relief = json.loads((MODEL / "relief_model_results.json").read_text())

    check(findings["records_analyzed"] == 8279, "Expected 8,279 analyzed complaints")
    check(quality["duplicate_complaint_ids_after_deduplication"] == 0,
          "Complaint IDs must be unique")
    check(findings["complaints_2024"] == 3356 and findings["complaints_2025"] == 4923,
          "Annual complaint counts changed")
    check(findings["year_over_year_change_pct"] == 46.7, "YoY growth changed")
    check(findings["spike_share_of_total_yoy_increase_pct"] == 46.3,
          "Spike contribution changed")
    check(findings["top_two_company_share_of_spike_pct"] == 92.0,
          "Company concentration changed")
    check(findings["repeated_narratives_in_spike_with_unique_complaint_ids"] == 257,
          "Repeated narrative count changed")

    with sqlite3.connect(PROJECT_DIR / "data/processed/cfpb_az_banking_complaints.sqlite") as db:
        rows, distinct_ids = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT complaint_id) FROM complaints"
        ).fetchone()
        indexes = {row[1] for row in db.execute("PRAGMA index_list(complaints)")}
    check(rows == distinct_ids == 8279, "SQLite table differs from the clean data")
    check({"idx_complaint_id", "idx_product_year", "idx_issue"} <= indexes,
          "Expected SQLite indexes are missing")

    total = {row["method"]: row for row in forecast["total_series_metrics"]}
    check(forecast["backtest_weeks"] == 26 and forecast["best_method_total"] == "median_8w",
          "Forecast backtest setup or selected method changed")
    check(abs(total["median_8w"]["mae"] - 10.17) < 0.01,
          "Median forecast MAE changed")
    check(abs(total["median_8w"]["mape_pct"] - 14.2) < 0.1,
          "Median forecast MAPE changed")
    check(abs(total["median_8w"]["wape_pct"] - 12.7) < 0.1,
          "Median forecast WAPE changed")
    check(abs(forecast["best_mae_reduction_vs_naive_pct"] - 22.9) < 0.2,
          "Forecast improvement changed")

    selected = relief["selected_model"]
    check(selected == "xgboost_structured", "Selected classification model changed")
    check({name: relief["split"][name]["rows"] for name in ("train", "valid", "test")}
          == {"train": 5182, "valid": 974, "test": 2123},
          "Time-based train/validation/test sizes changed")
    test = relief["models"][selected]["test"]
    baseline = relief["models"]["logistic_regression_baseline"]["test"]
    # Fixed package versions do not guarantee bit-identical XGBoost predictions
    # across native builds. These bounds cover the observed macOS and reported
    # Linux runs while still detecting a material performance regression.
    check(0.455 <= test["f1"] <= 0.475 and 0.46 <= test["recall"] <= 0.53,
          "Selected model is outside the documented cross-platform performance range")
    check(0.43 <= baseline["f1"] <= 0.45,
          "Logistic regression baseline changed")
    check(test["f1"] >= baseline["f1"] + 0.01,
          "Selected model no longer improves on the logistic baseline")
    check(0.45 <= test["pr_auc"] <= 0.47,
          "Selected model PR-AUC is outside the documented range")
    check(0.41 <= test["relief_captured_in_top_20pct"] <= 0.43,
          "Top-20% capture rate changed")

    with (MODEL / "relief_test_scores.csv").open(newline="") as source:
        scored = list(csv.DictReader(source))
    check(len(scored) == relief["split"]["test"]["rows"] == 2123,
          "Test score row count changed")
    tp = sum(int(row["flagged"]) and int(row["target"]) for row in scored)
    fp = sum(int(row["flagged"]) and not int(row["target"]) for row in scored)
    fn = sum(not int(row["flagged"]) and int(row["target"]) for row in scored)
    f1 = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    check(abs(f1 - test["f1"]) < 0.0006,
          "Exported flags do not reproduce reported test F1")
    check(abs(recall - test["recall"]) < 0.0006,
          "Exported flags do not reproduce reported test recall")
    top_n = round(0.2 * len(scored))
    ranked = sorted(scored, key=lambda row: float(row["relief_score"]), reverse=True)
    capture = sum(int(row["target"]) for row in ranked[:top_n]) / sum(
        int(row["target"]) for row in scored
    )
    check(abs(capture - test["relief_captured_in_top_20pct"]) < 0.01,
          "Exported scores do not reproduce the reported top-20% capture rate")

    for filename in ["complaint_analytics_dashboard.png", "model_dashboard.png"]:
        check((PROJECT_DIR / "figures" / filename).stat().st_size > 100_000,
              f"Dashboard missing or empty: {filename}")

    summary = {
        "status": "passed",
        "raw_sha256": digest,
        "analyzed_complaints": rows,
        "selected_model": selected,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "xgboost_version": version("xgboost"),
        "scikit_learn_version": version("scikit-learn"),
        "test_f1_from_exported_flags": round(f1, 4),
        "test_recall_from_exported_flags": round(recall, 4),
        "top_20pct_capture_from_exported_scores": round(capture, 4),
        "forecast_mae_complaints_per_week": total["median_8w"]["mae"],
        "forecast_mape_pct": total["median_8w"]["mape_pct"],
    }
    (ANALYSIS / "reproducibility_check.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
