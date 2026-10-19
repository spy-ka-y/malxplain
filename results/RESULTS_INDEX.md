# MalXplain results index

Generated 2026-10-19. Base paper: Madamidola, Ngobigha & Ez-zizi (2024), Intelligent Systems with Applications 25, 200472.

Every figure quoted in the report, the dashboard or the viva traces to one of the files below. Nothing is retyped from memory.

32 of 32 expected files are present.

## Reproduction

*Faithful reproduction of the base paper*

Produced by `src/baseline.py, src/adaptive.py`

| File | Contents | Present |
| --- | --- | --- |
| `stage1_baseline_holdout.csv` | Seven classifiers on known malware, beside the paper's Table 3 | yes |
| `stage1_baseline_cv.csv` | Ten fold cross validated F1 for the same classifiers | yes |
| `stage2_adaptive.csv` | Fifteen withheld subtype detectors, beside the paper's Table 4 | yes |
| `verification_report.txt` | Dataset integrity, leakage guard and reproduction fidelity | yes |

## Base paper comparison

*Faculty request: show our numbers against the paper's on the classifier page*

Produced by `src/paper_compare.py`

| File | Contents | Present |
| --- | --- | --- |
| `papercmp_stage1.csv` | Paired Stage 1 comparison, ours against the published figures | yes |
| `papercmp_stage2.csv` | Per subtype Stage 2 accuracy, ours against the published figures | yes |
| `papercmp_zeroshot.csv` | Classifier families over the 11 subtypes the paper reports | yes |
| `papercmp_context.json` | What the paper does and does not publish, so the gap is explicit | yes |

## Upgrade 1, tuning

*Paper limitation: no hyperparameter tuning*

Produced by `src/tuning.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade1_tuning.csv` | Default against tuned detectors, with the best parameter set | yes |

## Upgrade 2, classifier families

*Paper limitation: only Random Forest tested zero shot*

Produced by `src/comparison.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade2_comparison_full.csv` | Every classifier against every subtype | yes |
| `upgrade2_comparison_summary.csv` | Mean and worst case unseen recall per family | yes |

## Upgrade 3, blind spots

*Paper limitation: no error analysis*

Produced by `src/error_analysis.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade3_cross_subtype_recall.csv` | Fifteen by fifteen detector against target recall matrix | yes |
| `upgrade3_hardest_subtypes.csv` | Subtypes ranked by how hard they are to catch unseen | yes |
| `upgrade3_worst_blindspots.csv` | The worst individual detector and target pairings | yes |
| `upgrade3_blindspot_diagnosis.csv` | Feature level diagnosis of the TIBS blind spot | yes |
| `upgrade3_detector_ranking.csv` | Detectors ranked by mean unseen recall | yes |

## Upgrade 5, inference cost

*Paper limitation 2, completed: they speculated about speed and never measured it*

Produced by `src/benchmark.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade5_benchmark_Yash_laptop.csv` | Per subtype timings and model sizes, Yash laptop | yes |
| `upgrade5_benchmark_summary_Yash_laptop.csv` | Latency, size and accuracy per family, with ratios to Random Forest | yes |
| `upgrade5_host_Yash_laptop.json` | Exact machine the timings were taken on | yes |

## Upgrade 6, operating point

*Literature limitation [6], [18]: balanced test sets and recall only*

Produced by `src/operating_point.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade6_operating_point.csv` | Per subtype precision, FPR and average precision at each prevalence | yes |
| `upgrade6_operating_point_summary.csv` | Base paper configuration against ours at 50 and 10 percent prevalence | yes |

## Upgrade 7, feature selection

*Paper limitation: only one feature selection method*

Produced by `src/featsel.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade7_featsel_full.csv` | Every method against every subtype, with the features chosen | yes |
| `upgrade7_featsel_summary.csv` | Mean unseen recall per method | yes |
| `upgrade7_featsel_agreement.csv` | How far each method agrees with the base paper's choice | yes |

## Upgrade 8, significance

*Paper limitation: single run, no confidence intervals*

Produced by `src/significance.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade8_seeds_raw.csv` | Every arm, subtype and seed | yes |
| `upgrade8_arm_summary.csv` | Mean unseen recall per arm with 95 percent intervals | yes |
| `upgrade8_significance.csv` | Paired t and Wilcoxon tests on the two contested claims | yes |
| `upgrade8_per_subtype_tuning.csv` | Which subtypes show a tuning gain that survives the noise | yes |

## Upgrade 9, explanation faithfulness

*Literature limitation [9], [24]: explanation quality asserted, never measured*

Produced by `src/faithfulness.py`

| File | Contents | Present |
| --- | --- | --- |
| `upgrade9_faithfulness_ablation.csv` | Per sample ablation of the top SHAP feature against a random one | yes |
| `upgrade9_faithfulness_per_detector.csv` | Ablation margin per detector | yes |
| `upgrade9_faithfulness_grounding.csv` | Each generated finding and whether it is grounded, with the text | yes |
| `upgrade9_faithfulness_summary.json` | Headline rates, instrument controls and known limitations | yes |

## Headline figures

Read straight from the files above at generation time.

```json
{
  "stage1_max_deviation_vs_paper": 0.000616,
  "best_family_mean_unseen_recall": {
    "Classifier": "Extra Trees",
    "Mean_unseen_recall": 0.9888877884119016
  },
  "significance_tests": [
    {
      "Comparison": "RF tuned vs RF default",
      "Mean difference (pp)": 0.0301682674667037,
      "Wilcoxon p": 0.0252135807092021,
      "CI excludes zero": false
    },
    {
      "Comparison": "Extra Trees default vs RF default",
      "Mean difference (pp)": 0.1735650938701348,
      "Wilcoxon p": 5.01065184078262e-05,
      "CI excludes zero": true
    }
  ],
  "feature_selection_winner": "RF importance (base paper)",
  "operating_point": [
    {
      "Configuration": "Base paper (defaults)",
      "Prevalence": 0.5,
      "Mean precision": 0.9982,
      "Mean recall": 0.9877,
      "Mean FPR %": 0.1794
    },
    {
      "Configuration": "MalXplain (tuned)",
      "Prevalence": 0.5,
      "Mean precision": 0.9976,
      "Mean recall": 0.9892,
      "Mean FPR %": 0.2423
    },
    {
      "Configuration": "Base paper (defaults)",
      "Prevalence": 0.1,
      "Mean precision": 0.9841,
      "Mean recall": 0.987,
      "Mean FPR %": 0.1794
    },
    {
      "Configuration": "MalXplain (tuned)",
      "Prevalence": 0.1,
      "Mean precision": 0.9788,
      "Mean recall": 0.9889,
      "Mean FPR %": 0.2423
    }
  ],
  "inference_cost": [
    {
      "Classifier": "MLP",
      "Single us/sample": 101.65,
      "Model size (KB)": 33.804,
      "Latency vs RF": 0.02
    },
    {
      "Classifier": "Gradient Boosting",
      "Single us/sample": 175.44,
      "Model size (KB)": 102.124,
      "Latency vs RF": 0.04
    },
    {
      "Classifier": "XGBoost",
      "Single us/sample": 242.11,
      "Model size (KB)": 88.39266666666666,
      "Latency vs RF": 0.05
    },
    {
      "Classifier": "Random Forest",
      "Single us/sample": 4539.52,
      "Model size (KB)": 132.83933333333334,
      "Latency vs RF": 1.0
    },
    {
      "Classifier": "Extra Trees",
      "Single us/sample": 4561.21,
      "Model size (KB)": 365.0626666666667,
      "Latency vs RF": 1.0
    }
  ],
  "faithfulness": {
    "faithfulness_margin": 0.14739999999999998,
    "top_moved_more_rate_excluding_ties": 0.76,
    "mean_features_named_per_finding": 3.783333333333333,
    "feature_grounded_rate": 1.0,
    "confidence_grounded_rate": 1.0,
    "fully_grounded_rate": 1.0
  }
}
```
