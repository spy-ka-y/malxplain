"""
Operating point analysis: what the detectors do at a realistic malware base rate.

Every figure in the base paper, and in our first pass, comes from a test set that is
half benign and half malware, and is reported as accuracy or recall. Malware is not
half of real traffic. At a low base rate a detector with excellent recall can still
produce far more false alarms than true detections, and recall alone cannot show that.
Reference [6] in our literature review flags this bias explicitly, and [18] notes that
recall is highly sensitive to imbalance.

Nothing is retrained here. The same fifteen detectors are re-scored on test sets
resampled to a chosen malware prevalence, and precision, false positive rate and
average precision are reported beside recall.

Two configurations are run so the comparison is against the base paper rather than
against nothing: the paper's own configuration (library defaults) and our tuned
configuration. The tuned column is the one that has to win.

Usage:  python -m src.operating_point
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix

from src.adaptive import RANDOM_STATE, select_top_features, split_for_subtype
from src.preprocessing import Dataset, prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# 0.50 is the base paper's own setting, kept so the comparison is visible.
# 0.10 is the realistic operating point requested for this analysis.
PREVALENCES = [0.50, 0.10]


def _resample_to_prevalence(
    y_true: np.ndarray, proba: np.ndarray, prevalence: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample the scored holdout so malware makes up `prevalence` of it.

    The benign pool is kept whole and malware is thinned to hit the target rate,
    which keeps the benign evidence as large as possible and so keeps the false
    positive rate estimate stable.
    """
    rng = np.random.RandomState(seed)
    mal_idx = np.flatnonzero(y_true == 1)
    ben_idx = np.flatnonzero(y_true == 0)

    n_mal_target = int(round(prevalence / (1 - prevalence) * len(ben_idx)))
    if n_mal_target <= len(mal_idx):
        keep_mal = rng.choice(mal_idx, size=n_mal_target, replace=False)
        keep_ben = ben_idx
    else:
        # Not enough malware to reach the target, so thin the benign side instead.
        n_ben_target = int(round((1 - prevalence) / prevalence * len(mal_idx)))
        keep_mal = mal_idx
        keep_ben = rng.choice(ben_idx, size=n_ben_target, replace=False)

    keep = np.concatenate([keep_mal, keep_ben])
    return y_true[keep], proba[keep]


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": (2 * precision * recall / (precision + recall)
               if precision and recall and not np.isnan(precision) else float("nan")),
        "FPR %": 100 * fp / (fp + tn) if (fp + tn) else float("nan"),
        "Average precision": average_precision_score(y_true, proba),
        "False alarms per true detection": fp / tp if tp else float("inf"),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def _tuned_params(subtype: str) -> dict:
    """Best parameters found by the tuning stage, if that stage has been run."""
    path = RESULTS_DIR / "upgrade1_tuning.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["Subtype"] == subtype]
    if row.empty or "Best params" not in df.columns:
        return {}
    raw = row["Best params"].iloc[0]
    try:
        # Stored as a Python repr, so literal_eval rather than a JSON parse.
        params = ast.literal_eval(raw)
        return params if isinstance(params, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def run_all(ds: Dataset | None = None) -> pd.DataFrame:
    ds = ds or prepare()
    rows = []

    for i, subtype in enumerate(ds.subtypes, 1):
        X_train, y_train, X_test, y_test, _ = split_for_subtype(ds, subtype)
        feats = select_top_features(X_train, y_train)
        Xtr, Xte = X_train[feats], X_test[feats]
        y_arr = y_test.to_numpy()

        configs = {
            "Base paper (defaults)": {},
            "MalXplain (tuned)": _tuned_params(subtype),
        }
        for config_name, params in configs.items():
            clf = RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, **params
            )
            clf.fit(Xtr, y_train)
            proba = clf.predict_proba(Xte)[:, 1]

            for prev in PREVALENCES:
                yt, pr = _resample_to_prevalence(y_arr, proba, prev, RANDOM_STATE)
                rec = {
                    "Subtype": subtype,
                    "Configuration": config_name,
                    "Prevalence": prev,
                }
                rec.update(_metrics(yt, pr))
                rows.append(rec)
        print(f"  [{i:2d}/15] {subtype}")

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Configuration", "Prevalence"])
        .agg(**{
            "Mean precision": ("Precision", "mean"),
            "Mean recall": ("Recall", "mean"),
            "Mean FPR %": ("FPR %", "mean"),
            "Mean average precision": ("Average precision", "mean"),
            "Mean false alarms per detection": ("False alarms per true detection", "mean"),
        })
        .reset_index()
        .sort_values(["Prevalence", "Configuration"], ascending=[False, True])
        .reset_index(drop=True)
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=== Operating point analysis ===")
    print(f"  prevalences: {', '.join(f'{p:.0%}' for p in PREVALENCES)}")
    print("  0.50 is the base paper's own balanced setting, kept for comparison\n")

    df = run_all()
    df.to_csv(RESULTS_DIR / "upgrade6_operating_point.csv", index=False)

    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "upgrade6_operating_point_summary.csv", index=False)

    print("\n" + summary.to_string(index=False))

    balanced = summary[summary["Prevalence"] == 0.50]
    realistic = summary[summary["Prevalence"] == 0.10]
    if not balanced.empty and not realistic.empty:
        drop = (balanced["Mean precision"].mean() - realistic["Mean precision"].mean())
        print(f"\n  mean precision falls {drop:.4f} going from 50% to 10% prevalence")

    tuned = summary[(summary["Configuration"] == "MalXplain (tuned)")
                    & (summary["Prevalence"] == 0.10)]
    base = summary[(summary["Configuration"] == "Base paper (defaults)")
                   & (summary["Prevalence"] == 0.10)]
    if not tuned.empty and not base.empty:
        gain = tuned["Mean precision"].iloc[0] - base["Mean precision"].iloc[0]
        print(f"  at 10% prevalence, tuned beats defaults on precision by {gain:+.4f}")

    print("\n[operating_point] wrote upgrade6_operating_point.csv and its summary")


if __name__ == "__main__":
    main()
