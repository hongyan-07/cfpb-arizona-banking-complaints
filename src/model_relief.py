"""Predict which complaints are likely to end in monetary or non-monetary relief.

Business use: an operations team could use the score at intake to route
complaints that historically tend to end in relief (i.e., the company ended up
fixing something) to an experienced reviewer earlier.

Design choices that keep the evaluation honest:
- Only fields known when a complaint arrives are used as features. Anything
  written after the company responds (company response, public response,
  timely-response flag) is excluded to avoid target leakage.
- Train/validation/test splits are by time, not random: the model is trained
  on older complaints and scored on newer ones, the way it would be used.
- The decision threshold is chosen on the validation window and then frozen
  before the test window is scored.

Run from the project folder after `python src/analyze.py`:
    python src/model_relief.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "processed" / "complaints_clean.csv"
MODEL_DIR = PROJECT_DIR / "analysis" / "modeling"

RANDOM_STATE = 42
TRAIN_END = "2025-03"   # train: 2024-01 .. 2025-03
VALID_END = "2025-06"   # validation: 2025-04 .. 2025-06 (threshold tuning)
                        # test: 2025-07 .. 2025-12 (scored once, never tuned on)

CATEGORICAL = [
    "product_group",
    "sub_product",
    "issue",
    "sub_issue",
    "company_top",
    "submitted_via",
    "zip3",
]
NUMERIC = ["has_narrative", "older_american", "servicemember", "narrative_words"]
TEXT = "narrative_text"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df["target"] = df["relief_response"].astype(bool).astype(int)

    # Intake-time features only.
    train_mask = df["month"] <= TRAIN_END
    top_companies = df.loc[train_mask, "company"].value_counts().head(25).index
    df["company_top"] = np.where(df["company"].isin(top_companies), df["company"], "Other")
    df["zip3"] = df["zip_code"].astype(str).str[:3].where(
        df["zip_code"].astype(str).str.match(r"^\d{3}"), "unknown"
    )
    tags = df["tags"].fillna("")
    df["older_american"] = tags.str.contains("Older American").astype(int)
    df["servicemember"] = tags.str.contains("Servicemember").astype(int)
    df["has_narrative"] = df["has_narrative"].astype(bool).astype(int)
    df[TEXT] = df["consumer_complaint_narrative"].fillna("")
    df["narrative_words"] = np.log1p(df[TEXT].str.split().str.len().fillna(0))
    for col in CATEGORICAL:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["month"] <= TRAIN_END],
        "valid": df[(df["month"] > TRAIN_END) & (df["month"] <= VALID_END)],
        "test": df[df["month"] > VALID_END],
    }


def baseline_model() -> Pipeline:
    """Logistic regression on the structured intake fields only."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]
    )


class XGBIntakeModel:
    """XGBoost on one-hot structured fields, optionally + TF-IDF of the narrative."""

    def __init__(self, use_text: bool = True) -> None:
        self.use_text = use_text
        self.onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=5000, sublinear_tf=True,
            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        self.model: XGBClassifier | None = None

    def _matrix(self, df: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        a = self.onehot.fit_transform(df[CATEGORICAL]) if fit else self.onehot.transform(df[CATEGORICAL])
        blocks = [a, sparse.csr_matrix(df[NUMERIC].to_numpy(dtype=float))]
        if self.use_text:
            b = self.tfidf.fit_transform(df[TEXT]) if fit else self.tfidf.transform(df[TEXT])
            blocks.append(b)
        return sparse.hstack(blocks).tocsr()

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "XGBIntakeModel":
        x = self._matrix(df, fit=True)
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        self.model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=3, reg_lambda=1.0,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=4, tree_method="hist",
        )
        self.model.fit(x, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._matrix(df, fit=False))

    def feature_names(self) -> list[str]:
        names = list(self.onehot.get_feature_names_out(CATEGORICAL)) + NUMERIC
        if self.use_text:
            names += [f"text: {t}" for t in self.tfidf.get_feature_names_out()]
        return names


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.nanargmax(f1[:-1])])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    order = np.argsort(-p)
    top_n = int(round(0.2 * len(p)))
    return {
        # Keep the exact validation threshold so exported flags reproduce the
        # reported metrics. Round only when displaying it to a reader.
        "threshold": float(threshold),
        "f1": round(f1_score(y, pred), 3),
        "precision": round(precision_score(y, pred), 3),
        "recall": round(recall_score(y, pred), 3),
        "pr_auc": round(average_precision_score(y, p), 3),
        "roc_auc": round(roc_auc_score(y, p), 3),
        "flagged_share": round(pred.mean(), 3),
        "relief_captured_in_top_20pct": round(y[order[:top_n]].sum() / y.sum(), 3),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    parts = split(df)
    y = {k: v["target"].to_numpy() for k, v in parts.items()}

    results: dict[str, object] = {
        "target": "relief_response (closed with monetary or non-monetary relief)",
        "split": {
            k: {
                "months": f"{v['month'].min()} to {v['month'].max()}",
                "rows": int(len(v)),
                "positive_rate": round(float(v["target"].mean()), 3),
            }
            for k, v in parts.items()
        },
        "excluded_as_leakage": [
            "company_response_to_consumer", "company_public_response",
            "timely_response", "response_group", "date_sent_to_company",
        ],
    }

    # Candidate models. The final model is picked on VALIDATION PR-AUC only;
    # all test scores are reported for transparency.
    candidates = {
        "logistic_regression_baseline": ("structured intake fields", baseline_model()),
        "xgboost_structured": ("structured intake fields", XGBIntakeModel(use_text=False)),
        "xgboost_structured_plus_text": (
            "structured intake fields + narrative TF-IDF (1-2 grams)", XGBIntakeModel(use_text=True)
        ),
    }
    model_results: dict[str, dict] = {}
    decision_thresholds: dict[str, float] = {}
    test_scores: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    for name, (desc, model) in candidates.items():
        model.fit(parts["train"], y["train"])
        p_valid = model.predict_proba(parts["valid"])[:, 1]
        thr = best_f1_threshold(y["valid"], p_valid)   # frozen before test
        p_test = model.predict_proba(parts["test"])[:, 1]
        model_results[name] = {
            "features": desc,
            "valid": evaluate(y["valid"], p_valid, thr),
            "test": evaluate(y["test"], p_test, thr),
        }
        test_scores[name] = p_test
        decision_thresholds[name] = thr
        fitted[name] = model

    selected = max(model_results, key=lambda k: model_results[k]["valid"]["pr_auc"])
    results["models"] = model_results
    results["selected_model"] = selected
    results["selection_rule"] = "highest validation PR-AUC; threshold = best validation F1"

    # Feature importance (gain) for the selected XGBoost model.
    xgb = fitted[selected] if selected != "logistic_regression_baseline" else fitted["xgboost_structured"]
    xgb_test = test_scores[selected]
    xgb_thr = decision_thresholds[selected]
    booster = xgb.model.get_booster()
    gain = booster.get_score(importance_type="gain")
    names = xgb.feature_names()
    imp = pd.DataFrame(
        [(names[int(k[1:])], v) for k, v in gain.items()], columns=["feature", "gain"]
    ).sort_values("gain", ascending=False)
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(3)
    imp = imp.head(20)
    imp.to_csv(MODEL_DIR / "relief_feature_importance.csv", index=False)

    # Test-window risk scores for review / dashboard.
    scored = parts["test"][["complaint_id", "month", "product_group", "issue", "target"]].copy()
    scored["relief_score"] = xgb_test.round(4)
    scored["flagged"] = (xgb_test >= xgb_thr).astype(int)
    scored.to_csv(MODEL_DIR / "relief_test_scores.csv", index=False)

    by_product = (
        scored.groupby("product_group")
        .agg(
            complaints=("target", "size"),
            actual_relief_rate=("target", "mean"),
            mean_score=("relief_score", "mean"),
            flagged_share=("flagged", "mean"),
        )
        .round(3)
        .reset_index()
    )
    by_product.to_csv(MODEL_DIR / "relief_test_by_product.csv", index=False)

    # Precision-recall curves for the dashboard.
    curves = []
    labels = {
        "logistic_regression_baseline": "Logistic regression (baseline)",
        "xgboost_structured": "XGBoost (structured)",
        "xgboost_structured_plus_text": "XGBoost (structured + text)",
    }
    for key, p in test_scores.items():
        name = labels[key]
        pr, rc, _ = precision_recall_curve(y["test"], p)
        curves.append(pd.DataFrame({"model": name, "precision": pr, "recall": rc}))
    pd.concat(curves).to_csv(MODEL_DIR / "relief_pr_curves.csv", index=False)

    (MODEL_DIR / "relief_model_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
