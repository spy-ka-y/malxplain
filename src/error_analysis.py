"""
MalXplain: Upgrade 3: error analysis.

Madamidola et al. (2024) report aggregate accuracy per subtype model but state as a
limitation that they did not "incorporate error analysis to evaluate failure cases".
Aggregate accuracy hides the operationally important question:

    WHICH unseen malware subtypes slip past a given detector, and why?

This module answers that by producing:
  1. A 15x15 cross-subtype detection matrix: for every trained detector (row), the
     recall achieved on every unseen subtype (column). This exposes blind spots that
     a single averaged accuracy figure conceals.
  2. A feature-level diagnosis of the worst blind spot, comparing the distribution of
     the model's top-5 features for missed instances vs correctly-detected instances.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .adaptive import RANDOM_STATE, TOP_K, select_top_features, split_for_subtype
from .preprocessing import Dataset, prepare

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def cross_subtype_matrix(ds: Dataset) -> tuple[pd.DataFrame, dict]:
    """
    Build the detector x target recall matrix.

    matrix.loc[A, B] = recall of the detector trained on subtype A when tested on
    malware instances of subtype B.
    """
    subtypes = ds.subtypes
    matrix = pd.DataFrame(index=subtypes, columns=subtypes, dtype=float)
    artefacts = {}

    for trained_on in subtypes:
        X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, trained_on)
        top_features = select_top_features(X_train, y_train, TOP_K)

        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train[top_features], y_train)
        pred = clf.predict(X_test[top_features])

        for target in subtypes:
            mask = ((y_test == 1) & (test_subtype == target)).to_numpy()
            if mask.sum():
                matrix.loc[trained_on, target] = pred[mask].mean()

        artefacts[trained_on] = {
            "features": top_features,
            "X_test": X_test,
            "y_test": y_test,
            "test_subtype": test_subtype,
            "pred": pred,
        }
        print(f"  [err] detector trained on {trained_on:<13} "
              f"mean recall on unseen = {matrix.loc[trained_on].drop(trained_on).mean():.4f}", flush=True)

    return matrix, artefacts


def diagnose_blind_spot(artefacts: dict, detector: str, target: str) -> pd.DataFrame:
    """
    Compare feature values for missed vs detected instances of `target`
    under the detector trained on `detector`.
    """
    a = artefacts[detector]
    features = a["features"]
    mask = ((a["y_test"] == 1) & (a["test_subtype"] == target)).to_numpy()

    X = a["X_test"].loc[mask, features]
    detected = a["pred"][mask] == 1

    rows = []
    for f in features:
        missed_mean = X.loc[~detected, f].mean() if (~detected).sum() else np.nan
        det_mean = X.loc[detected, f].mean() if detected.sum() else np.nan
        rows.append(
            {
                "Feature": f,
                "Missed (mean)": missed_mean,
                "Detected (mean)": det_mean,
                "Difference": (missed_mean - det_mean) if detected.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Difference", key=abs, ascending=False)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare(verbose=False)

    print("=== Upgrade 3: cross-subtype error analysis ===")
    print("    (the paper reports aggregate accuracy only; no failure-case analysis)\n")

    matrix, artefacts = cross_subtype_matrix(ds)
    matrix.to_csv(RESULTS_DIR / "upgrade3_cross_subtype_recall.csv")

    print("\nCross-subtype recall matrix (rows = detector trained on, cols = tested against):")
    print((matrix * 100).round(1).to_string())

    # Identify the worst blind spots, excluding the diagonal (the seen subtype).
    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)
    off_diag = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)

    stacked = off_diag.stack().sort_values()
    print("\nTen worst detector/target blind spots:")
    worst = stacked.head(10).rename("Recall").reset_index()
    worst.columns = ["Detector trained on", "Unseen subtype", "Recall"]
    print(worst.to_string(index=False))
    worst.to_csv(RESULTS_DIR / "upgrade3_worst_blindspots.csv", index=False)

    print("\nHardest subtypes to detect (mean recall across all 14 other detectors):")
    hardest = off_diag.mean(axis=0).sort_values()
    print(hardest.round(4).to_string())
    hardest.rename("Mean recall when unseen").to_csv(RESULTS_DIR / "upgrade3_hardest_subtypes.csv")

    print("\nMost transferable detectors (mean recall on the 14 unseen subtypes):")
    best_detectors = off_diag.mean(axis=1).sort_values(ascending=False)
    print(best_detectors.round(4).to_string())
    best_detectors.rename("Mean unseen recall").to_csv(RESULTS_DIR / "upgrade3_detector_ranking.csv")

    # Feature-level diagnosis of the single worst blind spot.
    d, t = worst.iloc[0]["Detector trained on"], worst.iloc[0]["Unseen subtype"]
    print(f"\nFeature-level diagnosis of the worst blind spot: {d} detector vs unseen {t}")
    diag = diagnose_blind_spot(artefacts, d, t)
    print(diag.round(4).to_string(index=False))
    diag.to_csv(RESULTS_DIR / "upgrade3_blindspot_diagnosis.csv", index=False)


if __name__ == "__main__":
    main()
