"""Build the modeling dashboard (forecast + relief-likelihood panels) as a PNG.

Run after src/model_relief.py and src/forecast_volume.py:
    python src/build_model_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "analysis" / "modeling"
OUT_FILE = PROJECT_DIR / "figures" / "model_dashboard.png"

# Validated categorical order (lightness band, chroma, CVD separation, contrast).
BLUE, ORANGE, TEAL, PURPLE = "#1D6FA5", "#D97706", "#16907A", "#7C5AA6"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)

plt.rcParams.update(
    {
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)

METHOD_LABELS = {
    "naive_last_week": "Naive (last week)",
    "moving_average_4w": "4-week moving average",
    "median_8w": "8-week median",
    "ridge_lags_spike_capped": "Ridge on lags (spike-capped)",
}
MODEL_LABELS = {
    "logistic_regression_baseline": "Logistic regression (baseline)",
    "xgboost_structured": "XGBoost (structured)",
    "xgboost_structured_plus_text": "XGBoost (structured + text)",
}
MODEL_COLORS = {
    "Logistic regression (baseline)": BLUE,
    "XGBoost (structured)": ORANGE,
    "XGBoost (structured + text)": TEAL,
}


def main() -> None:
    relief = json.loads((MODEL_DIR / "relief_model_results.json").read_text())
    fc = json.loads((MODEL_DIR / "forecast_results.json").read_text())
    weekly = pd.read_csv(MODEL_DIR / "weekly_volume.csv", parse_dates=["week"])
    backtest = pd.read_csv(MODEL_DIR / "forecast_backtest_predictions.csv", parse_dates=["week"])
    fmetrics = pd.read_csv(MODEL_DIR / "forecast_backtest_metrics.csv")
    curves = pd.read_csv(MODEL_DIR / "relief_pr_curves.csv")
    scores = pd.read_csv(MODEL_DIR / "relief_test_scores.csv")

    sel = relief["selected_model"]
    sel_test = relief["models"][sel]["test"]
    base_test = relief["models"]["logistic_regression_baseline"]["test"]
    best = fc["best_method_total"]
    best_row = next(r for r in fc["total_series_metrics"] if r["method"] == best)
    test_rate = relief["split"]["test"]["positive_rate"]

    fig = plt.figure(figsize=(16, 10.5), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.36, 1, 1], hspace=0.45, wspace=0.30,
                          left=0.05, right=0.97, top=0.90, bottom=0.10)
    fig.text(0.05, 0.955, "Arizona Banking Complaints — Forecasting & Relief-Likelihood Models",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.928,
             "CFPB public data, 2024–2025. Time-based evaluation: models are scored only on "
             "weeks/complaints they never saw during training.", fontsize=10.5, color=INK_2)

    # KPI tiles
    kpis = [
        (f"{best_row['mae']:.1f}", "complaints/week MAE",
         f"Next-week forecast, total volume ({best_row['wape_pct']:.1f}% WAPE)"),
        (f"−{fc['best_mae_reduction_vs_naive_pct']:.0f}%", "forecast error vs naive",
         f"{METHOD_LABELS[best]} vs last-week benchmark"),
        (f"{sel_test['f1']:.2f}", "test F1 (relief)",
         f"vs {base_test['f1']:.2f} logistic baseline · {test_rate:.0%} base rate"),
        (f"{sel_test['relief_captured_in_top_20pct']:.0%}", "of relief cases in top 20%",
         "Reviewing the highest-scored fifth of complaints"),
    ]
    kpi_gs = gs[0, :].subgridspec(1, 4, wspace=0.08)
    for i, (value, label, note) in enumerate(kpis):
        ax = fig.add_subplot(kpi_gs[0, i])
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white",
                                   edgecolor=GRID, linewidth=1))
        ax.text(0.06, 0.60, value, fontsize=24, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.06, 0.36, label, fontsize=10.5, color=INK_2, transform=ax.transAxes)
        ax.text(0.06, 0.12, note, fontsize=9, color=MUTED, transform=ax.transAxes)

    # Panel 1: weekly total, actual vs forecast
    ax1 = fig.add_subplot(gs[1, 0])
    bt = backtest[backtest["series"] == "Total"]
    hist = weekly[weekly["week"] < bt["week"].min()]
    ax1.plot(hist["week"], hist["Total"], color=AXIS, linewidth=1.5)
    ax1.plot(bt["week"], bt["actual"], color=BLUE, linewidth=2, marker="o", markersize=3.5)
    ax1.plot(bt["week"], bt[best], color=ORANGE, linewidth=2, linestyle="--")
    ax1.set_ylim(0, 160)
    spike = weekly.loc[weekly["Total"].idxmax()]
    ax1.annotate(f"Jan 2025 spike: {int(spike['Total'])}/wk (off scale)",
                 xy=(spike["week"], 158), xytext=(spike["week"] + pd.Timedelta(days=30), 148),
                 fontsize=8.5, color=INK_2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax1.axvline(bt["week"].min(), color=AXIS, linewidth=0.8, linestyle=":")
    ax1.text(bt["week"].min() + pd.Timedelta(days=4), 8, "Backtest window →", fontsize=8.5, color=MUTED)
    ax1.set_title("Weekly complaint volume: actual vs next-week forecast")
    ax1.set_ylabel("Complaints per week")
    ax1.legend(handles=[
        plt.Line2D([], [], color=AXIS, lw=1.5, label="Training history"),
        plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=3.5, label="Actual (backtest)"),
        plt.Line2D([], [], color=ORANGE, lw=2, ls="--", label=f"Forecast: {METHOD_LABELS[best]}"),
    ], loc="upper left", frameon=False, fontsize=8.5)

    # Panel 2: forecast MAE by method, by product
    ax2 = fig.add_subplot(gs[1, 1])
    order = ["naive_last_week", "moving_average_4w", "ridge_lags_spike_capped", "median_8w"]
    series = ["Credit card", "Checking / savings", "Money transfer / digital wallet", "Prepaid card"]
    piv = fmetrics.pivot(index="series", columns="method", values="mae").loc[series, order]
    y = np.arange(len(series))
    h = 0.19
    colors = [AXIS, BLUE, TEAL, ORANGE]
    for j, m in enumerate(order):
        ax2.barh(y + (j - 1.5) * (h + 0.02), piv[m], height=h, color=colors[j],
                 label=METHOD_LABELS[m], edgecolor=SURFACE, linewidth=1)
    for i, s in enumerate(series):
        ax2.text(piv.loc[s, "median_8w"] + 0.15, i + 1.5 * (h + 0.02), f"{piv.loc[s, 'median_8w']:.1f}",
                 va="center", fontsize=8.5, color=INK_2)
    ax2.set_yticks(y, [x.replace("Money transfer / ", "Money transfer /\n") for x in series])
    ax2.invert_yaxis()
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("MAE, complaints per week (lower is better)")
    ax2.set_title("Forecast error by product, 26-week rolling backtest")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")

    # Panel 3: precision-recall curves
    ax3 = fig.add_subplot(gs[2, 0])
    for key, label in MODEL_LABELS.items():
        c = curves[curves["model"] == label]
        pr_auc = relief["models"][key]["test"]["pr_auc"]
        ax3.plot(c["recall"], c["precision"], color=MODEL_COLORS[label], linewidth=2,
                 label=f"{label} — PR-AUC {pr_auc:.2f}")
    ax3.axhline(test_rate, color=MUTED, linestyle=":", linewidth=1)
    ax3.text(0.01, test_rate - 0.06, f"Random guess ({test_rate:.0%} base rate)", ha="left",
             fontsize=8.5, color=MUTED)
    ax3.plot(sel_test["recall"], sel_test["precision"], "o", color=INK, markersize=8,
             markeredgecolor=SURFACE, markeredgewidth=2)
    ax3.annotate(f"Chosen threshold: recall {sel_test['recall']:.2f}, precision {sel_test['precision']:.2f}",
                 xy=(sel_test["recall"], sel_test["precision"]),
                 xytext=(sel_test["recall"] + 0.08, sel_test["precision"] + 0.22), fontsize=8.5,
                 color=INK_2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Recall (share of relief cases caught)")
    ax3.set_ylabel("Precision")
    ax3.set_title("Relief prediction on held-out Jul–Dec 2025 complaints")
    ax3.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 4: cumulative gains (triage value of the relief-likelihood score)
    ax4 = fig.add_subplot(gs[2, 1])
    s = scores.sort_values("relief_score", ascending=False).reset_index(drop=True)
    reviewed = np.arange(1, len(s) + 1) / len(s)
    captured = s["target"].cumsum() / s["target"].sum()
    ax4.plot(reviewed * 100, captured * 100, color=ORANGE, linewidth=2, label="Ranked by model score")
    ax4.plot([0, 100], [0, 100], color=MUTED, linestyle=":", linewidth=1, label="Random order")
    k = int(round(0.2 * len(s))) - 1
    ax4.plot(reviewed[k] * 100, captured.iloc[k] * 100, "o", color=INK, markersize=8,
             markeredgecolor=SURFACE, markeredgewidth=2)
    ax4.annotate(f"Top 20% reviewed → {captured.iloc[k]:.0%} of relief cases",
                 xy=(reviewed[k] * 100, captured.iloc[k] * 100), xytext=(30, 22), fontsize=8.5,
                 color=INK_2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.set_xlabel("% of complaints reviewed (highest score first)")
    ax4.set_ylabel("% of relief cases captured")
    ax4.set_title("Triage value of the relief score — " + MODEL_LABELS[sel].replace(" (", ", ").rstrip(")"))
    ax4.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.text(0.05, 0.015,
             "Notes: CFPB complaints are not a statistical sample of all consumers. Relief = company closed the "
             "complaint with monetary or non-monetary relief. Only intake-time fields are used as features.",
             fontsize=8.5, color=MUTED)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, facecolor=SURFACE)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
