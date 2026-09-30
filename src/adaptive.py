"""
MalXplain: Stage 2: adaptive (zero-shot) malware detection.

Replicates Section 3.3.2 / Table 4 of Madamidola et al. (2024), the paper's core
contribution: 15 Random Forest models, each trained on a SINGLE malware subtype,
then evaluated against a holdout containing all 14 unseen subtypes.

Protocol, per the paper:
  * Training set = 80% of one subtype's malware instances + an equal number of
    randomly selected benign instances.
  * Holdout set  = the remaining 20% of that subtype, ALL instances of the other
    14 subtypes, and all remaining benign instances.
  * Only the top-5 features (by Random Forest importance, fitted on the training
    split) are retained, to keep the model lightweight and interpretable.

The paper also reports model size (~340 KB) and inference speed (~5.7 us/file);
both are measured here so the lightweight claim can be verified rather than assumed.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .preprocessing import BENIGN_LABEL, Dataset, prepare

RANDOM_STATE = 42
TOP_K = 5
TRAIN_FRACTION = 0.8

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Paper Table 4 accuracies, for direct side-by-side comparison.
PAPER_TABLE4 = {
    "Transponder": 0.9984,
    "Pysa": 0.9974,
    "Reconyc": 0.9967,
    "Conti": 0.9962,
    "Maze": 0.9953,
    "Shade": 0.9905,
    "Emotet": 0.9901,
    "Ako": 0.9844,
    "Refroso": 0.9810,
    "Scar": 0.9810,
    "Zeus": 0.9800,
}


def split_for_subtype(
    ds: Dataset, subtype: str, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build the one-subtype training set and the all-unseen-subtypes holdout.

    Returns (X_train, y_train, X_test, y_test, test_subtype).
    """
    rng = np.random.RandomState(seed)

    is_benign = ds.y == 0
    is_this_subtype = (ds.y == 1) & (ds.subtype == subtype)

    subtype_idx = ds.X.index[is_this_subtype].to_numpy()
    benign_idx = ds.X.index[is_benign].to_numpy()

    # 80% of this subtype for training
    n_train_mal = int(round(TRAIN_FRACTION * len(subtype_idx)))
    train_mal = rng.choice(subtype_idx, size=n_train_mal, replace=False)
    test_mal_same = np.setdiff1d(subtype_idx, train_mal)

    # An equal number of benign instances for training
    train_ben = rng.choice(benign_idx, size=n_train_mal, replace=False)
    test_ben = np.setdiff1d(benign_idx, train_ben)

    # All other subtypes are unseen at training time
    unseen_mal = ds.X.index[(ds.y == 1) & (ds.subtype != subtype)].to_numpy()

    train_idx = np.concatenate([train_mal, train_ben])
    test_idx = np.concatenate([test_mal_same, unseen_mal, test_ben])

    return (
        ds.X.loc[train_idx],
        ds.y.loc[train_idx],
        ds.X.loc[test_idx],
        ds.y.loc[test_idx],
        ds.subtype.loc[test_idx],
    )


def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list[str]:
    """Rank features by Random Forest importance on the training split, keep the top k."""
    selector = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    selector.fit(X, y)
    order = np.argsort(selector.feature_importances_)[::-1]
    return [X.columns[i] for i in order[:k]]


def train_subtype_model(
    ds: Dataset,
    subtype: str,
    k: int = TOP_K,
    seed: int = RANDOM_STATE,
    model_params: dict | None = None,
    save_model: bool = True,
) -> dict:
    """Train and evaluate one subtype-specific detector. Returns a result record."""
    X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, subtype, seed)

    top_features = select_top_features(X_train, y_train, k)
    Xtr, Xte = X_train[top_features], X_test[top_features]

    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **(model_params or {}))
    clf.fit(Xtr, y_train)

    # Inference speed: mean microseconds per file, measured over the full holdout.
    t0 = time.perf_counter()
    pred = clf.predict(Xte)
    micros_per_file = (time.perf_counter() - t0) * 1e6 / len(Xte)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    blob = pickle.dumps(clf)
    if save_model:
        MODELS_DIR.mkdir(exist_ok=True)
        (MODELS_DIR / f"rf_{subtype}.pkl").write_bytes(blob)

    # Per-subtype recall on the UNSEEN subtypes only - the real zero-shot signal.
    unseen_mask = (y_test == 1) & (test_subtype != subtype)
    unseen_recall = recall_score(y_test[unseen_mask], pred[unseen_mask.to_numpy()])

    return {
        "Subtype": subtype,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "Unseen-subtype recall": unseen_recall,
        "FN": int(fn),
        "FP": int(fp),
        "FNR %": round(100 * fn / (fn + tp), 4),
        "FPR %": round(100 * fp / (fp + tn), 4),
        "Model size (KB)": round(len(blob) / 1024, 2),
        "us/file": round(micros_per_file, 2),
        "Top-5 features": ", ".join(top_features),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "Paper Accuracy": PAPER_TABLE4.get(subtype),
    }


def run_all(ds: Dataset | None = None, k: int = TOP_K) -> pd.DataFrame:
    """Train all 15 subtype-specific models and collate results."""
    ds = ds or prepare(verbose=False)
    rows = []
    for subtype in ds.subtypes:
        rec = train_subtype_model(ds, subtype, k=k)
        paper = rec["Paper Accuracy"]
        paper_str = f" (paper {paper:.4f})" if paper else " (not in paper table)"
        print(
            f"  [stage2] {subtype:<13} acc={rec['Accuracy']:.4f}{paper_str}  "
            f"unseen-recall={rec['Unseen-subtype recall']:.4f}  "
            f"{rec['Model size (KB)']}KB  {rec['us/file']}us/file",
            flush=True,
        )
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df["Δ vs paper"] = (df["Accuracy"] - df["Paper Accuracy"]).round(4)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare()
    print("\n=== Stage 2: adaptive detection (train on ONE subtype, test on all 15) ===")
    df = run_all(ds)

    cols = [
        "Subtype", "Accuracy", "Paper Accuracy", "Δ vs paper", "Unseen-subtype recall",
        "F1 Score", "FN", "FP", "FNR %", "FPR %", "Model size (KB)", "us/file",
    ]
    print("\n" + df[cols].to_string(index=False))
    df.to_csv(RESULTS_DIR / "stage2_adaptive.csv", index=False)

    n_above_99 = (df["Accuracy"] > 0.99).sum()
    print(f"\n[stage2] models above 99% accuracy: {n_above_99}/15  (paper reports 11/15)")
    print(f"[stage2] best: {df.iloc[0]['Subtype']} at {df.iloc[0]['Accuracy']:.4f}")
    print(f"[stage2] mean model size: {df['Model size (KB)'].mean():.1f} KB")


if __name__ == "__main__":
    main()
