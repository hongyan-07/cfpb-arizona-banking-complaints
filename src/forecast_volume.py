"""Forecast weekly complaint volume and backtest it honestly.

Business use: a complaint-operations team staffs reviewers a week ahead. The
question is how many complaints to expect next week, overall and by product.

Method:
- Build weekly counts (Monday-start weeks) for each product group and in total.
  The final, partial week of 2025 is dropped so it does not look like a drop.
- Rolling-origin backtest over the last 26 full weeks of 2025: each week is
  forecast using only the weeks before it, then the origin moves forward.
- Compare simple benchmarks with a lag-feature regression. Spikes like the
  January 2025 money-transfer surge are capped in the TRAINING history only
  (median + 3 x MAD), so one event does not distort later forecasts; the
  actual values being scored are never altered.

Metrics: MAE (complaints per week), MAPE, and WAPE (sum |error| / sum actual),
which is more stable for low-volume series such as prepaid cards.

Run from the project folder after `python src/analyze.py`:
    python src/forecast_volume.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "processed" / "complaints_clean.csv"
MODEL_DIR = PROJECT_DIR / "analysis" / "modeling"

BACKTEST_WEEKS = 26
LAGS = 4
SERIES_ORDER = [
    "Total",
    "Credit card",
    "Checking / savings",
    "Money transfer / digital wallet",
    "Prepaid card",
]


def weekly_counts() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    received = pd.to_datetime(df["date_received"], utc=True).dt.tz_localize(None)
    df["week"] = received.dt.to_period("W-SUN").dt.start_time
    weekly = df.groupby(["week", "product_group"]).size().unstack(fill_value=0)
    weekly["Total"] = weekly.sum(axis=1)
    last_day = received.max().normalize()
    if weekly.index[-1] + pd.Timedelta(days=6) > last_day:
        weekly = weekly.iloc[:-1]  # drop partial final week
    return weekly[SERIES_ORDER]


def cap_spikes(history: np.ndarray) -> np.ndarray:
    median = np.median(history)
    mad = np.median(np.abs(history - median)) or 1.0
    return np.minimum(history, median + 3 * 1.4826 * mad)


def forecast_naive(history: np.ndarray) -> float:
    return float(history[-1])


def forecast_ma4(history: np.ndarray) -> float:
    return float(history[-4:].mean())


def forecast_median8(history: np.ndarray) -> float:
    return float(np.median(history[-8:]))


def forecast_ridge(history: np.ndarray) -> float:
    """Ridge regression on the last 4 weekly values + 8-week median, spike-capped."""
    h = cap_spikes(history)
    rows, target = [], []
    for t in range(8, len(h)):
        rows.append(np.r_[h[t - LAGS:t], np.median(h[t - 8:t])])
        target.append(h[t])
    model = Ridge(alpha=10.0).fit(np.array(rows), np.array(target))
    x_next = np.r_[h[-LAGS:], np.median(h[-8:])]
    return float(max(model.predict(x_next.reshape(1, -1))[0], 0.0))


METHODS = {
    "naive_last_week": forecast_naive,
    "moving_average_4w": forecast_ma4,
    "median_8w": forecast_median8,
    "ridge_lags_spike_capped": forecast_ridge,
}


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(actual - pred)
    nonzero = actual > 0
    return {
        "mae": round(float(err.mean()), 2),
        "mape_pct": round(float((err[nonzero] / actual[nonzero]).mean() * 100), 1),
        "wape_pct": round(float(err.sum() / actual.sum() * 100), 1),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weekly = weekly_counts()
    weekly.to_csv(MODEL_DIR / "weekly_volume.csv")
    n = len(weekly)
    test_idx = range(n - BACKTEST_WEEKS, n)

    rows = []
    for series in SERIES_ORDER:
        values = weekly[series].to_numpy(dtype=float)
        for i in test_idx:
            history = values[:i]
            rec = {"week": weekly.index[i].date().isoformat(), "series": series, "actual": values[i]}
            for name, fn in METHODS.items():
                rec[name] = round(fn(history), 2)
            rows.append(rec)
    backtest = pd.DataFrame(rows)
    backtest.to_csv(MODEL_DIR / "forecast_backtest_predictions.csv", index=False)

    summary = []
    for series in SERIES_ORDER:
        part = backtest[backtest["series"] == series]
        for name in METHODS:
            summary.append(
                {"series": series, "method": name, **metrics(part["actual"].to_numpy(), part[name].to_numpy())}
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(MODEL_DIR / "forecast_backtest_metrics.csv", index=False)

    total = summary[summary["series"] == "Total"].set_index("method")
    best = total["mae"].idxmin()
    naive_mae = total.loc["naive_last_week", "mae"]
    results = {
        "unit": "complaints per week",
        "weeks_available": int(n),
        "backtest_window": f"{weekly.index[n - BACKTEST_WEEKS].date()} to {weekly.index[-1].date()}",
        "backtest_weeks": BACKTEST_WEEKS,
        "total_series_metrics": total.reset_index().to_dict(orient="records"),
        "best_method_total": best,
        "best_mae_reduction_vs_naive_pct": round((1 - total.loc[best, "mae"] / naive_mae) * 100, 1),
        "mean_weekly_total_in_backtest": round(
            float(backtest.loc[backtest["series"] == "Total", "actual"].mean()), 1
        ),
    }
    (MODEL_DIR / "forecast_results.json").write_text(json.dumps(results, indent=2))
    print(summary.pivot(index="series", columns="method", values="mae").loc[SERIES_ORDER])
    print(summary.pivot(index="series", columns="method", values="wape_pct").loc[SERIES_ORDER])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
