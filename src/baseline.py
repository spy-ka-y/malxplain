"""
MalXplain: Stage 1: baseline classifier performance against known malware variants.

Replicates Section 3.3.1 / Table 3 of Madamidola et al. (2024): seven classifiers
trained and tested on the same distribution of malware subtypes, using a stratified
80/20 split, plus stratified 10-fold cross-validation for validation.

This stage answers "can we detect *known* malware at all?" and establishes that our
implementation matches the paper before we attempt the harder unseen-subtype task.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import prepare

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# The paper's reported Table 3 accuracies, for direct comparison.
PAPER_TABLE3 = {
    "Random Forest": 1.0000,
    "Gradient Boosting": 0.9996,
    "Decision Tree": 0.9997,
    "Support Vector": 0.9970,
    "Logistic Regression": 0.9958,
    "GaussianNB": 0.9922,
    "KNeighbors": 0.9998,
}


def build_classifiers(include_svm: bool = True) -> dict:
    """The seven classifiers used in the paper, with scikit-learn defaults."""
    clf = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
    }
    if include_svm:
        clf["Support Vector"] = SVC(random_state=RANDOM_STATE)
    return clf


def run_holdout(ds, test_size: float = 0.2, include_svm: bool = True) -> pd.DataFrame:
    """Train/test each classifier on a stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=test_size, random_state=RANDOM_STATE, stratify=ds.y
    )
    rows = []
    for name, clf in build_classifiers(include_svm).items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rows.append(
            {
                "Classifier": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1 Score": f1_score(y_test, pred),
                "Fit+Predict (s)": round(time.time() - t0, 1),
                "Paper Accuracy": PAPER_TABLE3.get(name),
            }
        )
        print(f"  [stage1] {name:<20} acc={rows[-1]['Accuracy']:.4f} "
              f"(paper {PAPER_TABLE3.get(name):.4f})  {rows[-1]['Fit+Predict (s)']}s", flush=True)

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df["Δ vs paper"] = (df["Accuracy"] - df["Paper Accuracy"]).round(4)
    return df


def run_cross_validation(ds, folds: int = 10, include_svm: bool = False) -> pd.DataFrame:
    """Stratified k-fold cross-validation on F1, as reported in the paper's Table A.2."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in build_classifiers(include_svm).items():
        t0 = time.time()
        scores = cross_val_score(clf, ds.X, ds.y, cv=cv, scoring="f1", n_jobs=-1)
        rows.append(
            {
                "Classifier": name,
                f"Mean F1 ({folds}-fold)": scores.mean(),
                "Std": scores.std(),
                "Time (s)": round(time.time() - t0, 1),
            }
        )
        print(f"  [cv] {name:<20} F1={scores.mean():.4f} ± {scores.std():.4f}", flush=True)
    return pd.DataFrame(rows).sort_values(f"Mean F1 ({folds}-fold)", ascending=False).reset_index(drop=True)


def main(include_svm: bool = True, do_cv: bool = True) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare()

    print("\n=== Stage 1: baseline classifiers (stratified 80/20 split) ===")
    holdout = run_holdout(ds, include_svm=include_svm)
    holdout.to_csv(RESULTS_DIR / "stage1_baseline_holdout.csv", index=False)
    print("\n" + holdout.to_string(index=False))

    if do_cv:
        print(f"\n=== Stage 1: {10}-fold cross-validation ===")
        cv = run_cross_validation(ds, include_svm=False)
        cv.to_csv(RESULTS_DIR / "stage1_baseline_cv.csv", index=False)
        print("\n" + cv.to_string(index=False))

    summary = {
        "best_classifier": holdout.iloc[0]["Classifier"],
        "best_accuracy": float(holdout.iloc[0]["Accuracy"]),
        "mean_accuracy": float(holdout["Accuracy"].mean()),
    }
    (RESULTS_DIR / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[stage1] best = {summary['best_classifier']} "
          f"({summary['best_accuracy']:.4f}); mean across 7 = {summary['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
