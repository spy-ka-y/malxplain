"""
Repeated runs and statistical significance.

Every figure in the base paper, and until now in this project, comes from a single
run at one fixed seed. Two of our headline claims rest on differences small enough
that a single run cannot support them: tuning improving unseen recall by a mean of
0.216 percentage points, and Extra Trees beating Random Forest by 0.002 mean unseen
recall. This module measures how much those quantities move when only the random
split changes, and reports whether the differences survive.

Scope is deliberately narrow. Re-running the full hyperparameter search ten times
would cost hours and would answer a different question. The claim under test is
whether the chosen parameters help, not whether the search finds them again, so the
best parameters found earlier are refitted across ten different splits. The classifier
claim is tested between Random Forest and Extra Trees only, since that is the specific
comparison stated as a finding.

Both a paired t test and a Wilcoxon signed rank test are reported. Recall is bounded
and its differences are not guaranteed to be normal, so the non parametric test is the
one to trust when the two disagree.

Usage:  python -m src.significance
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.adaptive import RANDOM_STATE, select_top_features, split_for_subtype
from src.preprocessing import Dataset, prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

N_SEEDS = 10
SEEDS = [RANDOM_STATE + i for i in range(N_SEEDS)]


def _tuned_params(subtype: str) -> dict:
    path = RESULTS_DIR / "upgrade1_tuning.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["Subtype"] == subtype]
    if row.empty:
        return {}
    try:
        params = ast.literal_eval(row["Best params"].iloc[0])
        return params if isinstance(params, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _unseen_recall(clf, Xte, y_test, test_subtype, subtype) -> float:
    pred = clf.predict(Xte)
    mask = ((test_subtype != subtype) & (y_test == 1)).to_numpy()
    return float(pred[mask].mean()) if mask.any() else float("nan")


def run_all(ds: Dataset | None = None) -> pd.DataFrame:
    ds = ds or prepare()
    rows = []

    for i, subtype in enumerate(ds.subtypes, 1):
        params = _tuned_params(subtype)
        for seed in SEEDS:
            X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(
                ds, subtype, seed
            )
            feats = select_top_features(X_train, y_train)
            Xtr, Xte = X_train[feats], X_test[feats]

            arms = {
                "RF default": RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1),
                "RF tuned": RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1, **params),
                "Extra Trees default": ExtraTreesClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1),
            }
            for arm, clf in arms.items():
                clf.fit(Xtr, y_train)
                rows.append({
                    "Subtype": subtype,
                    "Seed": seed,
                    "Arm": arm,
                    "Unseen recall": _unseen_recall(
                        clf, Xte, y_test, test_subtype, subtype),
                })
        print(f"  [{i:2d}/15] {subtype}")

    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame, arm_a: str, arm_b: str) -> dict:
    """Paired comparison of arm_a minus arm_b over every (subtype, seed) pair."""
    wide = df.pivot_table(index=["Subtype", "Seed"], columns="Arm",
                          values="Unseen recall")
    diff = (wide[arm_a] - wide[arm_b]).dropna()

    n = len(diff)
    mean = diff.mean()
    sem = stats.sem(diff) if n > 1 else 0.0
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 and sem > 0 \
        else (mean, mean)
    t_stat, t_p = stats.ttest_rel(wide[arm_a], wide[arm_b])
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:  # all differences identical
        w_stat, w_p = float("nan"), 1.0

    return {
        "Comparison": f"{arm_a} vs {arm_b}",
        "n pairs": n,
        "Mean difference (pp)": mean * 100,
        "95% CI low (pp)": ci[0] * 100,
        "95% CI high (pp)": ci[1] * 100,
        "t test p": float(t_p),
        "Wilcoxon p": float(w_p),
        "Significant at 0.05": bool(w_p < 0.05),
        "CI excludes zero": bool(ci[0] > 0 or ci[1] < 0),
    }


def per_subtype_tuning(df: pd.DataFrame) -> pd.DataFrame:
    """Which individual subtypes show a tuning gain that survives the noise."""
    wide = df.pivot_table(index=["Subtype", "Seed"], columns="Arm",
                          values="Unseen recall").reset_index()
    rows = []
    for subtype, grp in wide.groupby("Subtype"):
        diff = (grp["RF tuned"] - grp["RF default"]).dropna()
        n = len(diff)
        sem = stats.sem(diff) if n > 1 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=sem) \
            if n > 1 and sem > 0 else (diff.mean(), diff.mean())
        rows.append({
            "Subtype": subtype,
            "Mean gain (pp)": diff.mean() * 100,
            "95% CI low (pp)": ci[0] * 100,
            "95% CI high (pp)": ci[1] * 100,
            "Gain is significant": bool(ci[0] > 0),
        })
    return pd.DataFrame(rows).sort_values("Mean gain (pp)", ascending=False) \
        .reset_index(drop=True)


def arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, grp in df.groupby("Arm"):
        per_seed = grp.groupby("Seed")["Unseen recall"].mean()
        n = len(per_seed)
        sem = stats.sem(per_seed) if n > 1 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=per_seed.mean(), scale=sem) \
            if n > 1 and sem > 0 else (per_seed.mean(), per_seed.mean())
        rows.append({
            "Arm": arm,
            "Mean unseen recall": per_seed.mean(),
            "95% CI low": ci[0],
            "95% CI high": ci[1],
            "Std across seeds": per_seed.std(),
        })
    return pd.DataFrame(rows).sort_values("Mean unseen recall", ascending=False) \
        .reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=== Repeated runs and significance ===")
    print(f"  {N_SEEDS} seeds x 15 subtypes x 3 arms\n")

    df = run_all()
    df.to_csv(RESULTS_DIR / "upgrade8_seeds_raw.csv", index=False)

    arms = arm_summary(df)
    arms.to_csv(RESULTS_DIR / "upgrade8_arm_summary.csv", index=False)

    tests = pd.DataFrame([
        _paired(df, "RF tuned", "RF default"),
        _paired(df, "Extra Trees default", "RF default"),
    ])
    tests.to_csv(RESULTS_DIR / "upgrade8_significance.csv", index=False)

    per_sub = per_subtype_tuning(df)
    per_sub.to_csv(RESULTS_DIR / "upgrade8_per_subtype_tuning.csv", index=False)

    print("\n" + arms.round(6).to_string(index=False))
    print("\nPaired tests:")
    print(tests.round(5).to_string(index=False))
    print("\nTuning gain per subtype:")
    print(per_sub.round(4).to_string(index=False))

    n_sig = int(per_sub["Gain is significant"].sum())
    print(f"\n  subtypes with a tuning gain whose interval excludes zero: "
          f"{n_sig}/15")
    for _, row in tests.iterrows():
        verdict = "survives" if row["Significant at 0.05"] else "does NOT survive"
        print(f"  {row['Comparison']}: {verdict} "
              f"(mean {row['Mean difference (pp)']:+.3f} pp, "
              f"Wilcoxon p={row['Wilcoxon p']:.4f})")

    print("\n[significance] wrote upgrade8_seeds_raw.csv, arm summary, "
          "significance tests and per subtype gains")


if __name__ == "__main__":
    main()
