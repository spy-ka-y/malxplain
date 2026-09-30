"""
MalXplain: verification and reproduction report.

Builds the side-by-side comparison between the figures published in
Madamidola et al. (2024) and the figures this project actually measured, plus
leakage and sanity checks so that any claimed improvement can be defended.

Checks performed:
  1. Dataset integrity        - record counts and subtype distribution vs paper Table 2.
  2. Stage 1 reproduction     - our 7 classifiers vs paper Table 3.
  3. Stage 2 reproduction     - our 15 subtype models vs paper Table 4.
  4. Train/test disjointness  - no sample appears in both splits (leakage guard).
  5. Upgrade summary          - what each upgrade actually changed, honestly reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive import PAPER_TABLE4, split_for_subtype
from .preprocessing import prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Paper Table 2: subtype instance counts.
PAPER_TABLE2 = {
    "Zeus": 1950, "Emotet": 1967, "Refroso": 2000, "Scar": 2000, "Reconyc": 1570,
    "180solutions": 2000, "CWS": 2000, "Gator": 2200, "Transponder": 2410, "TIBS": 1410,
    "Conti": 1988, "Maze": 1958, "Pysa": 1717, "Ako": 2000, "Shade": 2128,
}


def check_dataset(ds) -> pd.DataFrame:
    counts = ds.subtype[ds.y == 1].value_counts()
    rows = [
        {
            "Subtype": s,
            "Paper count": PAPER_TABLE2[s],
            "Our count": int(counts.get(s, 0)),
            "Match": PAPER_TABLE2[s] == int(counts.get(s, 0)),
        }
        for s in sorted(PAPER_TABLE2)
    ]
    return pd.DataFrame(rows)


def check_leakage(ds, subtypes: list[str] | None = None) -> pd.DataFrame:
    """Confirm the training split and holdout share no samples, for every subtype."""
    rows = []
    for s in (subtypes or ds.subtypes):
        X_train, y_train, X_test, _, _ = split_for_subtype(ds, s)
        overlap = set(X_train.index) & set(X_test.index)
        rows.append(
            {
                "Subtype": s,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "Overlapping samples": len(overlap),
                "Train is balanced": int(y_train.sum()) == int((y_train == 0).sum()),
                "Clean": len(overlap) == 0,
            }
        )
    return pd.DataFrame(rows)


def build_report() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare(verbose=False)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 78)
    emit("MalXplain - Reproduction & Verification Report")
    emit("Base paper: Madamidola, Ngobigha & Ez-zizi (2024), Intell. Syst. Appl. 25, 200472")
    emit("=" * 78)

    # --- 1. dataset integrity ---
    emit("\n[1] DATASET INTEGRITY (vs paper Table 2)")
    dcheck = check_dataset(ds)
    emit(dcheck.to_string(index=False))
    emit(f"    -> all subtype counts match: {bool(dcheck['Match'].all())}")
    emit(f"    -> total records: {len(ds)} (paper: 58,596)")
    emit(f"    -> features after removing invariants: {len(ds.feature_names)} (paper: 52)")

    # --- 2. leakage guard ---
    emit("\n[2] TRAIN/TEST DISJOINTNESS (leakage guard)")
    leak = check_leakage(ds)
    emit(leak.to_string(index=False))
    emit(f"    -> all splits clean: {bool(leak['Clean'].all())}")
    emit(f"    -> all training sets balanced 50/50: {bool(leak['Train is balanced'].all())}")

    # --- 3. Stage 1 ---
    s1_path = RESULTS_DIR / "stage1_baseline_holdout.csv"
    if s1_path.exists():
        emit("\n[3] STAGE 1 REPRODUCTION (vs paper Table 3)")
        s1 = pd.read_csv(s1_path)
        emit(s1[["Classifier", "Accuracy", "Paper Accuracy", "Δ vs paper"]].to_string(index=False))
        emit(f"    -> max absolute deviation: {s1['Δ vs paper'].abs().max():.4f}")
        emit(f"    -> our best: {s1.iloc[0]['Classifier']} ({s1.iloc[0]['Accuracy']:.4f}); "
             f"paper's best: Random Forest (1.0000)")

    # --- 4. Stage 2 ---
    s2_path = RESULTS_DIR / "stage2_adaptive.csv"
    if s2_path.exists():
        emit("\n[4] STAGE 2 REPRODUCTION (vs paper Table 4)")
        s2 = pd.read_csv(s2_path)
        cols = ["Subtype", "Accuracy", "Paper Accuracy", "Δ vs paper", "Unseen-subtype recall"]
        emit(s2[cols].to_string(index=False))
        matched = s2.dropna(subset=["Paper Accuracy"])
        emit(f"    -> subtypes reported in the paper: {len(matched)}/15")
        emit(f"    -> mean absolute deviation on those: {matched['Δ vs paper'].abs().mean():.4f}")
        emit(f"    -> models above 99% accuracy: {(s2['Accuracy'] > 0.99).sum()}/15 "
             f"(paper reports 11/15)")
        tr = s2[s2['Subtype'] == 'Transponder'].iloc[0]
        emit(f"    -> Transponder model size: {tr['Model size (KB)']} KB (paper: ~340 KB)")
        emit(f"    -> Transponder inference: {tr['us/file']} us/file (paper: 5.7 us/file, "
             f"different hardware)")

    # --- 5. upgrades ---
    emit("\n[5] UPGRADE OUTCOMES (what we added beyond the paper)")

    t_path = RESULTS_DIR / "upgrade1_tuning.csv"
    if t_path.exists():
        t = pd.read_csv(t_path)
        emit(f"\n  Upgrade 1 - hyperparameter tuning (paper used defaults, stated as a limitation)")
        emit(f"    improved on accuracy      : {(t['Δ accuracy'] > 0).sum()}/15 subtypes")
        emit(f"    improved on unseen recall : {(t['Δ unseen recall'] > 0).sum()}/15 subtypes")
        emit(f"    mean Δ accuracy           : {t['Δ accuracy'].mean()*100:+.3f} pp")
        emit(f"    mean Δ unseen recall      : {t['Δ unseen recall'].mean()*100:+.3f} pp")
        top = t.nlargest(3, "Δ accuracy")
        for _, r in top.iterrows():
            emit(f"      {r['Subtype']:<13} {r['Δ accuracy']*100:+.3f} pp accuracy, "
                 f"{r['Δ unseen recall']*100:+.3f} pp unseen recall")

    c_path = RESULTS_DIR / "upgrade2_comparison_summary.csv"
    if c_path.exists():
        c = pd.read_csv(c_path, index_col=0)
        emit(f"\n  Upgrade 2 - classifier comparison (paper tested Random Forest only here)")
        emit("    " + c.round(4).to_string().replace("\n", "\n    "))
        best = c.index[0]
        emit(f"    -> best mean unseen recall: {best}")
        emit(f"    -> Random Forest rank: {list(c.index).index('Random Forest') + 1}/5")

    h_path = RESULTS_DIR / "upgrade3_hardest_subtypes.csv"
    d_path = RESULTS_DIR / "upgrade3_detector_ranking.csv"
    if h_path.exists() and d_path.exists():
        hardest = pd.read_csv(h_path, index_col=0).iloc[:, 0]
        detectors = pd.read_csv(d_path, index_col=0).iloc[:, 0]
        emit(f"\n  Upgrade 3 - error analysis (paper reports aggregate accuracy only)")
        emit(f"    hardest unseen subtype    : {hardest.index[0]} "
             f"({hardest.iloc[0]*100:.1f}% mean recall across 14 detectors)")
        emit(f"    next hardest              : {hardest.index[1]} ({hardest.iloc[1]*100:.1f}%)")
        emit(f"    most transferable detector: {detectors.index[0]} "
             f"({detectors.iloc[0]*100:.2f}% mean unseen recall)")
        emit(f"    least transferable        : {detectors.index[-1]} "
             f"({detectors.iloc[-1]*100:.2f}%)")

    emit("\n" + "=" * 78)
    (RESULTS_DIR / "verification_report.txt").write_text("\n".join(lines))
    print(f"\nreport saved -> {RESULTS_DIR / 'verification_report.txt'}")


if __name__ == "__main__":
    build_report()
