"""
Feature selection comparison.

The base paper ranks features by Random Forest importance, keeps the top five, and
lists the absence of alternative selection methods as one of its own limitations. We
inherited that choice without testing it, so this module asks whether those five
features are the right five or merely the first five one method happened to pick.

Four methods are compared, each confined to the training split so the unseen subtype
holdout stays untouched:

  Random Forest importance   the base paper's method, our control
  Mutual information         non linear dependence, no model assumptions
  ANOVA F test               the method used by reference [12] on this dataset
  Recursive elimination      the method used by reference [10] on this dataset

Two things are measured: whether the methods agree on which features matter, and
whether the choice changes unseen subtype recall at all.

Usage:  python -m src.featsel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score

from src.adaptive import RANDOM_STATE, TOP_K, split_for_subtype
from src.preprocessing import Dataset, prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------- selectors ----
def select_rf_importance(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    sel = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    sel.fit(X, y)
    order = np.argsort(sel.feature_importances_)[::-1]
    return [X.columns[i] for i in order[:k]]


def select_mutual_info(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    order = np.argsort(scores)[::-1]
    return [X.columns[i] for i in order[:k]]


def select_anova_f(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(scores)[::-1]
    return [X.columns[i] for i in order[:k]]


def select_rfe(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    # A smaller forest and a step of five keep the repeated refits tractable
    # without changing which features survive in practice.
    est = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(estimator=est, n_features_to_select=k, step=5)
    rfe.fit(X, y)
    return [c for c, keep in zip(X.columns, rfe.support_) if keep]


SELECTORS = {
    "RF importance (base paper)": select_rf_importance,
    "Mutual information": select_mutual_info,
    "ANOVA F test": select_anova_f,
    "Recursive elimination": select_rfe,
}


# ----------------------------------------------------------------- driver -----
def run_all(ds: Dataset | None = None, k: int = TOP_K) -> pd.DataFrame:
    ds = ds or prepare()
    rows = []

    for i, subtype in enumerate(ds.subtypes, 1):
        X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, subtype)
        unseen = test_subtype != subtype

        for name, selector in SELECTORS.items():
            feats = selector(X_train, y_train, k)
            clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            clf.fit(X_train[feats], y_train)
            pred = clf.predict(X_test[feats])

            unseen_mal = unseen & (y_test == 1)
            rows.append({
                "Subtype": subtype,
                "Method": name,
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "Unseen recall": (
                    pred[unseen_mal.to_numpy()].mean()
                    if unseen_mal.any() else float("nan")
                ),
                "Features": ", ".join(feats),
            })
        print(f"  [{i:2d}/15] {subtype}")

    return pd.DataFrame(rows)


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    """How often each method picks the same features as the base paper's method."""
    base_name = "RF importance (base paper)"
    rows = []
    for method in SELECTORS:
        overlaps, jaccards = [], []
        for subtype, grp in df.groupby("Subtype"):
            a = set(grp.loc[grp["Method"] == base_name, "Features"].iloc[0].split(", "))
            b = set(grp.loc[grp["Method"] == method, "Features"].iloc[0].split(", "))
            overlaps.append(len(a & b))
            jaccards.append(len(a & b) / len(a | b))
        rows.append({
            "Method": method,
            "Mean shared features (of 5)": np.mean(overlaps),
            "Mean Jaccard vs base paper": np.mean(jaccards),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    base_name = "RF importance (base paper)"
    agg = df.groupby("Method").agg(**{
        "Mean accuracy": ("Accuracy", "mean"),
        "Mean unseen recall": ("Unseen recall", "mean"),
        "Min unseen recall": ("Unseen recall", "min"),
    }).reset_index()

    base = agg.loc[agg["Method"] == base_name, "Mean unseen recall"].iloc[0]
    agg["Δ vs base paper (pp)"] = ((agg["Mean unseen recall"] - base) * 100).round(3)

    wins = []
    for method in agg["Method"]:
        n = 0
        for subtype, grp in df.groupby("Subtype"):
            best = grp.loc[grp["Unseen recall"].idxmax(), "Method"]
            n += int(best == method)
        wins.append(n)
    agg["Subtypes won (of 15)"] = wins

    return agg.sort_values("Mean unseen recall", ascending=False).reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=== Feature selection comparison ===")
    print(f"  methods: {', '.join(SELECTORS)}")
    print(f"  selecting top {TOP_K} features inside the training split only\n")

    df = run_all()
    df.to_csv(RESULTS_DIR / "upgrade7_featsel_full.csv", index=False)

    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "upgrade7_featsel_summary.csv", index=False)

    agree = agreement(df)
    agree.to_csv(RESULTS_DIR / "upgrade7_featsel_agreement.csv", index=False)

    print("\n" + summary.round(6).to_string(index=False))
    print("\nAgreement with the base paper's selection:")
    print(agree.round(3).to_string(index=False))

    best = summary.iloc[0]
    base_row = summary[summary["Method"] == "RF importance (base paper)"].iloc[0]
    if best["Method"] == "RF importance (base paper)":
        print("\n  The base paper's choice is not beaten by any alternative here, "
              "which validates it rather than replacing it.")
    else:
        print(f"\n  {best['Method']} beats the base paper's method by "
              f"{best['Mean unseen recall'] - base_row['Mean unseen recall']:+.6f} "
              "mean unseen recall.")

    print("\n[featsel] wrote upgrade7_featsel_full.csv, summary and agreement")


if __name__ == "__main__":
    main()
