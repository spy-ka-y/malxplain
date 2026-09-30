"""
MalXplain: Upgrade 1: hyperparameter tuning.

Madamidola et al. (2024) state their first limitation explicitly:

    "Firstly, we did not perform hyperparameter tuning, instead opting for
     default hyperparameter settings for our classifiers."

This module closes that gap. For each of the 15 subtype-specific detectors we run a
randomised search over the Random Forest hyperparameter space and report the accuracy
delta against the paper's default-parameter baseline.

Methodological note: the search is cross-validated *inside the training split only*.
The unseen-subtype holdout is never touched during tuning, so the reported improvement
is not an artefact of fitting to the test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .adaptive import RANDOM_STATE, TOP_K, select_top_features, split_for_subtype
from .preprocessing import Dataset, prepare

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

N_ITER = 30
CV_FOLDS = 3


def tune_subtype(ds: Dataset, subtype: str, n_iter: int = N_ITER) -> dict:
    """Tune one subtype detector and compare against the default-parameter baseline."""
    X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, subtype)
    top_features = select_top_features(X_train, y_train, TOP_K)
    Xtr, Xte = X_train[top_features], X_test[top_features]

    unseen_mask = ((y_test == 1) & (test_subtype != subtype)).to_numpy()

    # --- baseline: scikit-learn defaults, as in the paper ---
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    base.fit(Xtr, y_train)
    base_pred = base.predict(Xte)
    base_acc = accuracy_score(y_test, base_pred)
    base_unseen = recall_score(y_test[unseen_mask], base_pred[unseen_mask])

    # --- tuned: randomised search, cross-validated within the training split ---
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(Xtr, y_train)

    tuned = search.best_estimator_
    tuned_pred = tuned.predict(Xte)
    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_unseen = recall_score(y_test[unseen_mask], tuned_pred[unseen_mask])

    return {
        "Subtype": subtype,
        "Default accuracy": base_acc,
        "Tuned accuracy": tuned_acc,
        "Δ accuracy": tuned_acc - base_acc,
        "Default unseen recall": base_unseen,
        "Tuned unseen recall": tuned_unseen,
        "Δ unseen recall": tuned_unseen - base_unseen,
        "Default F1": f1_score(y_test, base_pred),
        "Tuned F1": f1_score(y_test, tuned_pred),
        "Best params": str(search.best_params_),
    }


def main(n_iter: int = N_ITER) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare(verbose=False)

    print(f"=== Upgrade 1: hyperparameter tuning ({n_iter} candidates x {CV_FOLDS}-fold CV per subtype) ===")
    print("    Baseline = scikit-learn defaults (what the paper used).\n")

    rows = []
    for subtype in ds.subtypes:
        rec = tune_subtype(ds, subtype, n_iter)
        rows.append(rec)
        arrow = "+" if rec["Δ accuracy"] >= 0 else ""
        print(
            f"  [tune] {subtype:<13} default={rec['Default accuracy']:.4f} -> "
            f"tuned={rec['Tuned accuracy']:.4f}  ({arrow}{rec['Δ accuracy']*100:.3f} pp)  "
            f"unseen-recall {arrow if rec['Δ unseen recall']>=0 else ''}{rec['Δ unseen recall']*100:.3f} pp",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("Δ accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "upgrade1_tuning.csv", index=False)

    show = ["Subtype", "Default accuracy", "Tuned accuracy", "Δ accuracy",
            "Default unseen recall", "Tuned unseen recall", "Δ unseen recall"]
    print("\n" + df[show].to_string(index=False))

    improved = (df["Δ accuracy"] > 0).sum()
    print(f"\n[tune] improved: {improved}/15 subtypes")
    print(f"[tune] mean Δ accuracy      : {df['Δ accuracy'].mean()*100:+.3f} pp")
    print(f"[tune] mean Δ unseen recall : {df['Δ unseen recall'].mean()*100:+.3f} pp")
    print(f"[tune] largest gain         : {df.iloc[0]['Subtype']} "
          f"({df.iloc[0]['Δ accuracy']*100:+.3f} pp)")


if __name__ == "__main__":
    main()
