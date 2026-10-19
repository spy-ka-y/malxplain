"""
Results index: what every file in results/ is, and which claim it backs.

By this point the project has thirty odd result files across nine stages, and the
figures quoted in the report, the dashboard and the viva are scattered across them.
Without an index, checking a number means remembering which CSV it came from, which
is exactly the kind of thing nobody remembers a month later.

This module writes two files. RESULTS_INDEX.md is for a person: every result file,
what it holds, the module that produced it, and the limitation it addresses. The
JSON alongside is the same thing for code, so the dashboard and the verification
report can look up a figure by name rather than hard coding a path.

Usage:  python -m src.manifest
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

PAPER = ("Madamidola, Ngobigha & Ez-zizi (2024), Intelligent Systems with "
         "Applications 25, 200472")

# Each entry: the stage, what it answers, which limitation, and the module behind it.
STAGES = [
    {
        "stage": "Reproduction",
        "addresses": "Faithful reproduction of the base paper",
        "source": "src/baseline.py, src/adaptive.py",
        "files": {
            "stage1_baseline_holdout.csv": "Seven classifiers on known malware, beside the paper's Table 3",
            "stage1_baseline_cv.csv": "Ten fold cross validated F1 for the same classifiers",
            "stage2_adaptive.csv": "Fifteen withheld subtype detectors, beside the paper's Table 4",
            "verification_report.txt": "Dataset integrity, leakage guard and reproduction fidelity",
        },
    },
    {
        "stage": "Base paper comparison",
        "addresses": "Faculty request: show our numbers against the paper's on the classifier page",
        "source": "src/paper_compare.py",
        "files": {
            "papercmp_stage1.csv": "Paired Stage 1 comparison, ours against the published figures",
            "papercmp_stage2.csv": "Per subtype Stage 2 accuracy, ours against the published figures",
            "papercmp_zeroshot.csv": "Classifier families over the 11 subtypes the paper reports",
            "papercmp_context.json": "What the paper does and does not publish, so the gap is explicit",
        },
    },
    {
        "stage": "Upgrade 1, tuning",
        "addresses": "Paper limitation: no hyperparameter tuning",
        "source": "src/tuning.py",
        "files": {"upgrade1_tuning.csv": "Default against tuned detectors, with the best parameter set"},
    },
    {
        "stage": "Upgrade 2, classifier families",
        "addresses": "Paper limitation: only Random Forest tested zero shot",
        "source": "src/comparison.py",
        "files": {
            "upgrade2_comparison_full.csv": "Every classifier against every subtype",
            "upgrade2_comparison_summary.csv": "Mean and worst case unseen recall per family",
        },
    },
    {
        "stage": "Upgrade 3, blind spots",
        "addresses": "Paper limitation: no error analysis",
        "source": "src/error_analysis.py",
        "files": {
            "upgrade3_cross_subtype_recall.csv": "Fifteen by fifteen detector against target recall matrix",
            "upgrade3_hardest_subtypes.csv": "Subtypes ranked by how hard they are to catch unseen",
            "upgrade3_worst_blindspots.csv": "The worst individual detector and target pairings",
            "upgrade3_blindspot_diagnosis.csv": "Feature level diagnosis of the TIBS blind spot",
            "upgrade3_detector_ranking.csv": "Detectors ranked by mean unseen recall",
        },
    },
    {
        "stage": "Upgrade 5, inference cost",
        "addresses": "Paper limitation 2, completed: they speculated about speed and never measured it",
        "source": "src/benchmark.py",
        "files": {
            "upgrade5_benchmark_Yash_laptop.csv": "Per subtype timings and model sizes, Yash laptop",
            "upgrade5_benchmark_summary_Yash_laptop.csv": "Latency, size and accuracy per family, with ratios to Random Forest",
            "upgrade5_host_Yash_laptop.json": "Exact machine the timings were taken on",
        },
    },
    {
        "stage": "Upgrade 6, operating point",
        "addresses": "Literature limitation [6], [18]: balanced test sets and recall only",
        "source": "src/operating_point.py",
        "files": {
            "upgrade6_operating_point.csv": "Per subtype precision, FPR and average precision at each prevalence",
            "upgrade6_operating_point_summary.csv": "Base paper configuration against ours at 50 and 10 percent prevalence",
        },
    },
    {
        "stage": "Upgrade 7, feature selection",
        "addresses": "Paper limitation: only one feature selection method",
        "source": "src/featsel.py",
        "files": {
            "upgrade7_featsel_full.csv": "Every method against every subtype, with the features chosen",
            "upgrade7_featsel_summary.csv": "Mean unseen recall per method",
            "upgrade7_featsel_agreement.csv": "How far each method agrees with the base paper's choice",
        },
    },
    {
        "stage": "Upgrade 8, significance",
        "addresses": "Paper limitation: single run, no confidence intervals",
        "source": "src/significance.py",
        "files": {
            "upgrade8_seeds_raw.csv": "Every arm, subtype and seed",
            "upgrade8_arm_summary.csv": "Mean unseen recall per arm with 95 percent intervals",
            "upgrade8_significance.csv": "Paired t and Wilcoxon tests on the two contested claims",
            "upgrade8_per_subtype_tuning.csv": "Which subtypes show a tuning gain that survives the noise",
        },
    },
    {
        "stage": "Upgrade 9, explanation faithfulness",
        "addresses": "Literature limitation [9], [24]: explanation quality asserted, never measured",
        "source": "src/faithfulness.py",
        "files": {
            "upgrade9_faithfulness_ablation.csv": "Per sample ablation of the top SHAP feature against a random one",
            "upgrade9_faithfulness_per_detector.csv": "Ablation margin per detector",
            "upgrade9_faithfulness_grounding.csv": "Each generated finding and whether it is grounded, with the text",
            "upgrade9_faithfulness_summary.json": "Headline rates, instrument controls and known limitations",
        },
    },
]


def headline_figures() -> dict:
    """The numbers quoted most often, each read from its file rather than retyped."""
    out = {}

    def _read(name, fn):
        path = RESULTS_DIR / name
        if path.exists():
            try:
                out.update(fn(path))
            except Exception as exc:  # a malformed file should not kill the index
                out[f"{name} (unreadable)"] = str(exc)

    _read("papercmp_stage1.csv", lambda p: {
        "stage1_max_deviation_vs_paper": round(
            pd.read_csv(p)["Abs delta"].max(), 6)})
    _read("upgrade2_comparison_summary.csv", lambda p: {
        "best_family_mean_unseen_recall": pd.read_csv(p)
        .sort_values("Mean_unseen_recall", ascending=False).iloc[0]
        [["Classifier", "Mean_unseen_recall"]].to_dict()})
    _read("upgrade8_significance.csv", lambda p: {
        "significance_tests": pd.read_csv(p)[
            ["Comparison", "Mean difference (pp)", "Wilcoxon p",
             "CI excludes zero"]].to_dict("records")})
    _read("upgrade7_featsel_summary.csv", lambda p: {
        "feature_selection_winner": pd.read_csv(p).iloc[0]["Method"]})
    _read("upgrade6_operating_point_summary.csv", lambda p: {
        "operating_point": pd.read_csv(p)[
            ["Configuration", "Prevalence", "Mean precision",
             "Mean recall", "Mean FPR %"]].round(4).to_dict("records")})
    _read("upgrade5_benchmark_summary_Yash_laptop.csv", lambda p: {
        "inference_cost": pd.read_csv(p)[
            ["Classifier", "Single us/sample", "Model size (KB)",
             "Latency vs RF"]].to_dict("records")})
    _read("upgrade9_faithfulness_summary.json", lambda p: {
        "faithfulness": {k: v for k, v in json.loads(p.read_text()).items()
                         if k in ("faithfulness_margin",
                                  "top_moved_more_rate_excluding_ties",
                                  "feature_grounded_rate",
                                  "confidence_grounded_rate",
                                  "fully_grounded_rate",
                                  "mean_features_named_per_finding")}})
    return out


def build() -> tuple[str, dict]:
    present, missing = [], []
    for stage in STAGES:
        for name in stage["files"]:
            (present if (RESULTS_DIR / name).exists() else missing).append(name)

    figures = headline_figures()
    index = {
        "generated": date.today().isoformat(),
        "base_paper": PAPER,
        "files_present": len(present),
        "files_missing": missing,
        "stages": STAGES,
        "headline_figures": figures,
    }

    lines = [
        "# MalXplain results index",
        "",
        f"Generated {index['generated']}. Base paper: {PAPER}.",
        "",
        "Every figure quoted in the report, the dashboard or the viva traces to one "
        "of the files below. Nothing is retyped from memory.",
        "",
        f"{len(present)} of {len(present) + len(missing)} expected files are present.",
        "",
    ]
    if missing:
        lines += ["Missing: " + ", ".join(missing), ""]

    for stage in STAGES:
        lines += [f"## {stage['stage']}", "",
                  f"*{stage['addresses']}*", "",
                  f"Produced by `{stage['source']}`", "",
                  "| File | Contents | Present |", "| --- | --- | --- |"]
        for name, what in stage["files"].items():
            tick = "yes" if (RESULTS_DIR / name).exists() else "**no**"
            lines.append(f"| `{name}` | {what} | {tick} |")
        lines.append("")

    lines += ["## Headline figures", "",
              "Read straight from the files above at generation time.", "",
              "```json", json.dumps(figures, indent=2), "```", ""]

    return "\n".join(lines), index


def main() -> None:
    md, index = build()
    (RESULTS_DIR / "RESULTS_INDEX.md").write_text(md)
    (RESULTS_DIR / "results_index.json").write_text(json.dumps(index, indent=2))
    print(f"[manifest] {index['files_present']} files indexed")
    if index["files_missing"]:
        print(f"[manifest] missing: {', '.join(index['files_missing'])}")
    print("[manifest] wrote results/RESULTS_INDEX.md and results/results_index.json")


if __name__ == "__main__":
    main()
