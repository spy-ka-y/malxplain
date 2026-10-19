"""
Head to head comparison against the published figures of the base paper.

Madamidola, Ngobigha & Ez-zizi (2024), Intelligent Systems with Applications 25, 200472.

Two comparisons are possible and they are not equally strong, so this module keeps
them apart rather than blending them into one chart.

Stage 1 is a true paired comparison. The paper's Table 3 reports all seven
classifiers on known malware and we reproduced every one of them, so each of our
numbers has a published counterpart.

The zero shot setting is only partly comparable. The paper's Table 4 reports
accuracy for Random Forest alone, and only for 11 of the 15 subtypes. It publishes
no unseen subtype recall at all, and no figure whatsoever for the other four
classifier families. So the strongest honest comparison is our Random Forest mean
accuracy against the paper's, computed over exactly the 11 subtypes they report.
Everything beyond that is our own extension, and this module labels it as such
rather than inventing a baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

PAPER_CITATION = (
    "Madamidola, O. A., Ngobigha, F., & Ez-zizi, A. (2024). Detecting new obfuscated "
    "malware variants: A lightweight and interpretable machine learning approach. "
    "Intelligent Systems with Applications, 25, 200472."
)


def stage1_comparison() -> pd.DataFrame:
    """Seven classifiers on known malware, ours beside the paper's Table 3."""
    df = pd.read_csv(RESULTS_DIR / "stage1_baseline_holdout.csv")
    out = df[["Classifier", "Accuracy", "Paper Accuracy"]].copy()
    out["Delta"] = out["Accuracy"] - out["Paper Accuracy"]
    out["Abs delta"] = out["Delta"].abs()
    out = out.sort_values("Paper Accuracy", ascending=False).reset_index(drop=True)
    out["Matches within 0.001"] = out["Abs delta"] < 0.001
    return out


def paper_reported_subtypes() -> list[str]:
    """The subtypes for which the paper publishes a Stage 2 accuracy."""
    df = pd.read_csv(RESULTS_DIR / "stage2_adaptive.csv")
    return df.loc[df["Paper Accuracy"].notna(), "Subtype"].tolist()


def stage2_comparison() -> pd.DataFrame:
    """Per subtype accuracy for Random Forest, ours beside the paper's Table 4."""
    df = pd.read_csv(RESULTS_DIR / "stage2_adaptive.csv")
    out = df[["Subtype", "Accuracy", "Paper Accuracy", "Unseen-subtype recall"]].copy()
    out["Delta"] = out["Accuracy"] - out["Paper Accuracy"]
    out["Reported by paper"] = out["Paper Accuracy"].notna()
    return out.sort_values("Accuracy", ascending=False).reset_index(drop=True)


def zeroshot_comparison() -> tuple[pd.DataFrame, dict]:
    """
    Classifier families under the zero shot protocol, restricted to the subtypes the
    paper reports so that the one available published number is a like for like
    reference rather than an average over a different set of subtypes.
    """
    reported = paper_reported_subtypes()
    full = pd.read_csv(RESULTS_DIR / "upgrade2_comparison_full.csv")
    subset = full[full["Subtype"].isin(reported)]

    rows = []
    for clf, grp in subset.groupby("Classifier"):
        rows.append({
            "Classifier": clf,
            "Mean accuracy (paper's 11 subtypes)": grp["Accuracy"].mean(),
            "Mean unseen recall (paper's 11 subtypes)": grp["Unseen recall"].mean(),
            "Published by paper": clf == "Random Forest",
        })
    out = pd.DataFrame(rows).sort_values(
        "Mean accuracy (paper's 11 subtypes)", ascending=False
    ).reset_index(drop=True)

    stage2 = pd.read_csv(RESULTS_DIR / "stage2_adaptive.csv")
    paper_mean = stage2.loc[stage2["Paper Accuracy"].notna(), "Paper Accuracy"].mean()
    ours_rf = out.loc[out["Classifier"] == "Random Forest",
                      "Mean accuracy (paper's 11 subtypes)"].iloc[0]

    context = {
        "n_subtypes_compared": len(reported),
        "subtypes_compared": reported,
        "paper_random_forest_mean_accuracy": float(paper_mean),
        "our_random_forest_mean_accuracy": float(ours_rf),
        "delta": float(ours_rf - paper_mean),
        "paper_publishes_unseen_recall": False,
        "paper_publishes_other_families": False,
        "note": (
            "The paper reports Stage 2 accuracy for Random Forest only, and only for "
            f"{len(reported)} of the 15 subtypes. It publishes no unseen subtype recall "
            "and no result for any other classifier family under this protocol, which is "
            "the gap this comparison fills."
        ),
    }
    return out, context


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    s1 = stage1_comparison()
    s1.to_csv(RESULTS_DIR / "papercmp_stage1.csv", index=False)

    s2 = stage2_comparison()
    s2.to_csv(RESULTS_DIR / "papercmp_stage2.csv", index=False)

    zs, ctx = zeroshot_comparison()
    zs.to_csv(RESULTS_DIR / "papercmp_zeroshot.csv", index=False)
    (RESULTS_DIR / "papercmp_context.json").write_text(json.dumps(ctx, indent=2))

    print("=== Stage 1, seven classifiers on known malware (paper Table 3) ===")
    print(s1.to_string(index=False))
    print(f"\n  max absolute deviation: {s1['Abs delta'].max():.4f}")
    print(f"  classifiers matching within 0.001: "
          f"{int(s1['Matches within 0.001'].sum())}/{len(s1)}")

    print("\n=== Stage 2, Random Forest per subtype (paper Table 4) ===")
    reported = s2[s2["Reported by paper"]]
    print(f"  subtypes the paper reports : {len(reported)}/15")
    print(f"  our mean accuracy          : {reported['Accuracy'].mean():.4f}")
    print(f"  paper mean accuracy        : {reported['Paper Accuracy'].mean():.4f}")
    print(f"  mean absolute deviation    : {reported['Delta'].abs().mean():.4f}")

    print("\n=== Zero shot classifier families, paper's 11 subtypes only ===")
    print(zs.to_string(index=False))
    print(f"\n  {ctx['note']}")
    print(f"  our Random Forest {ctx['our_random_forest_mean_accuracy']:.4f} vs "
          f"paper {ctx['paper_random_forest_mean_accuracy']:.4f} "
          f"(delta {ctx['delta']:+.4f})")

    print("\n[paper_compare] wrote papercmp_stage1.csv, papercmp_stage2.csv, "
          "papercmp_zeroshot.csv, papercmp_context.json")


if __name__ == "__main__":
    main()
