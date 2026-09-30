"""
MalXplain: Upgrade 2: multi-classifier comparison in the zero-shot setting.

Madamidola et al. (2024) evaluate seven classifiers in Stage 1 (known malware), but
Stage 2, the novel "train on one subtype, detect fourteen unseen ones" experiment , 
is run with Random Forest ONLY. So the paper cannot answer an obvious question:

    Is Random Forest's cross-subtype generalisation actually special,
    or would any reasonable classifier do just as well here?

This module answers it by running the identical Stage 2 protocol (same splits, same
top-5 RF-selected features, same holdout) with five classifier families, and reporting
accuracy and unseen-subtype recall for each.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .adaptive import RANDOM_STATE, TOP_K, select_top_features, split_for_subtype
from .preprocessing import Dataset, prepare

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_models() -> dict:
    """Classifier families to compare, all at library defaults for a fair comparison."""
    return {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0
        ),
        "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP": MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
    }


def compare_subtype(ds: Dataset, subtype: str) -> list[dict]:
    """Run every classifier through the zero-shot protocol for one subtype."""
    X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, subtype)

    # Feature selection is held constant (RF importance, as in the paper) so that the
    # comparison isolates the classifier, not the feature-selection method.
    top_features = select_top_features(X_train, y_train, TOP_K)
    Xtr, Xte = X_train[top_features], X_test[top_features]
    unseen_mask = ((y_test == 1) & (test_subtype != subtype)).to_numpy()

    rows = []
    for name, model in build_models().items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rows.append(
            {
                "Subtype": subtype,
                "Classifier": name,
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "Unseen recall": recall_score(y_test[unseen_mask], pred[unseen_mask]),
            }
        )
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare(verbose=False)

    print("=== Upgrade 2: classifier comparison under the zero-shot protocol ===")
    print("    (paper tested Random Forest only in this setting)\n")

    all_rows = []
    for subtype in ds.subtypes:
        rows = compare_subtype(ds, subtype)
        all_rows.extend(rows)
        best = max(rows, key=lambda r: r["Unseen recall"])
        summary = "  ".join(f"{r['Classifier'].split()[0]}={r['Unseen recall']:.4f}" for r in rows)
        print(f"  [cmp] {subtype:<13} {summary}   best={best['Classifier']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "upgrade2_comparison_full.csv", index=False)

    pivot = (
        df.groupby("Classifier")
        .agg(
            Mean_accuracy=("Accuracy", "mean"),
            Mean_unseen_recall=("Unseen recall", "mean"),
            Min_unseen_recall=("Unseen recall", "min"),
            Wins=("Classifier", "size"),
        )
        .drop(columns="Wins")
        .sort_values("Mean_unseen_recall", ascending=False)
    )

    wins = df.loc[df.groupby("Subtype")["Unseen recall"].idxmax(), "Classifier"].value_counts()
    pivot["Subtypes won (of 15)"] = wins.reindex(pivot.index).fillna(0).astype(int)

    pivot.to_csv(RESULTS_DIR / "upgrade2_comparison_summary.csv")
    print("\n" + pivot.round(4).to_string())

    rf_rank = list(pivot.index).index("Random Forest") + 1
    print(f"\n[cmp] Random Forest ranks {rf_rank}/5 on mean unseen-subtype recall.")


if __name__ == "__main__":
    main()
