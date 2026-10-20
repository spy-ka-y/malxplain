# MalXplain

**An Improved Lightweight and Interpretable Machine Learning Approach for Detecting New Obfuscated Malware Variants**

B.Tech Cybersecurity, Software Engineering Project
Pravara Jain (K020) · Yash Mehta (K035) · Somya Soni (K063)
NMIMS MPSTME, Mumbai

---

## What this is

MalXplain reproduces and then extends:

> Madamidola, O.A., Ngobigha, F., & Ez-zizi, A. (2024). *Detecting new obfuscated
> malware variants: A lightweight and interpretable machine learning approach.*
> Intelligent Systems with Applications, 25, 200472.
> https://doi.org/10.1016/j.iswa.2024.200472

The base paper trains 15 Random Forest detectors, each on a **single** malware subtype
(roughly 6.6 percent of the dataset), and shows they still detect the other 14
**unseen** subtypes, a proxy for zero-day detection. MalXplain reproduces that result,
closes six of the eight limitations the authors state themselves, adds three drawn from
the wider literature review, and produces a plain-language analyst finding instead of a
raw SHAP plot.

Every figure quoted here is read from a file in `results/`. See
[results/RESULTS_INDEX.md](results/RESULTS_INDEX.md) for the map from each claim to the
file and the module that produced it.

## Reproduction status

| Check | Paper | MalXplain | Status |
|---|---|---|---|
| Dataset records | 58,596 | 58,596 | exact match |
| Subtype distribution (Table 2) | 15 subtypes | 15 subtypes | all 15 counts match exactly |
| Features after removing invariants | 52 | 52 | invariant features detected independently |
| Stage 1 best classifier | Random Forest | Random Forest | max deviation 0.000616 across all 7 |
| Stage 2 models above 99% | 11 / 15 | 11 / 15 | exact match |
| Stage 2 RF accuracy, the 11 subtypes the paper reports | 0.9901 | 0.9914 | +0.0013 |
| Transponder model size | ~340 KB | 346.88 KB | consistent |
| Lowest-performing subtype | Zeus (98%) | Zeus (97.95%) | same subtype |

What the paper does **not** publish is as important as what it does. It gives Stage 2
accuracy for Random Forest only, for 11 of the 15 subtypes, and no unseen-subtype
recall figures at all. Our unseen-recall results therefore extend the paper rather than
contradict it, and `results/papercmp_context.json` records that gap explicitly so no
comparison is read as closer than the published data allows.

## What this project adds

**1. Hyperparameter tuning.** The paper used scikit-learn defaults and lists this as
its first limitation. Randomised search (30 candidates, 3-fold CV, inside the training
split only) is run across all 15 detectors. The honest result is in section 8 below:
over repeated runs the gain is +0.030 pp and not distinguishable from noise.

**2. Classifier comparison in the zero-shot setting.** The paper tests seven
classifiers on known malware but only Random Forest on unseen subtypes, so it cannot
say whether RF's generalisation is special. Running the identical protocol across five
families shows **Extra Trees** ahead of Random Forest on mean unseen recall
(0.9889 vs 0.9869), and **MLP** with the best worst case (0.9692 vs RF's 0.9591).
Random Forest is a good default, not an optimal one.

**3. Cross-subtype error analysis.** The paper reports aggregate accuracy and lists
"no error analysis" as a limitation. A full 15x15 detector-against-target recall matrix
exposes a blind spot invisible in aggregate figures: **TIBS** is detected at only 87.8
percent mean recall when unseen, falling to roughly 64 percent for the Zeus, Ako and
Refroso detectors. Feature-level diagnosis shows missed TIBS instances carry a markedly
higher `dlllist.avg_dlls_per_proc` (0.79 vs 0.68), so they load more DLLs per process
and look more like benign software.

**4. Explainability to plain-language findings.** The paper's SHAP output is a plot a
human must interpret. MalXplain serialises SHAP evidence into a structured payload and
has a **locally hosted LLM (Ollama)** turn it into a readable finding, with a
deterministic template fallback so the prototype never hard-fails during a demo.

**5. Inference cost, measured.** The authors speculated that other models "could offer
substantial improvements in speed while maintaining similar accuracy levels" and never
measured it. Single-sample latency, batch throughput and model size are now measured
with threads pinned to one core, a discarded warm-up pass and the median of 10 repeats.
On an i5-11400H laptop, Gradient Boosting runs at 175.44 us per sample against Random
Forest's 4539.52 us, about 26 times faster, for 0.23 pp less mean unseen recall. The
same measurement on a cloud Xeon container reproduces the family ratios to within 0.01,
which is the hardware-independent part of the claim.

**6. Operating point under realistic prevalence.** Limitation raised by references [6]
and [18]: results on a 50/50 test set and recall alone say nothing about how many false
alarms an analyst actually sees. Re-measured at 10 percent malware prevalence, the base
paper's default configuration gives 0.9841 mean precision at 0.179 percent FPR, and our
tuned configuration 0.9788 at 0.242 percent. Tuning buys a little recall and costs
precision.

**7. Feature selection, four methods instead of one.** The paper ranks by Random Forest
importance, keeps the top five, and lists the absence of alternatives as its own
limitation. Comparing RF importance against mutual information, the ANOVA F test and
recursive elimination, all confined to the training split, the paper's own choice wins
(0.98688 mean unseen recall against 0.98671 for recursive elimination, 0.98647 for
mutual information and 0.97846 for the ANOVA F test, which is 0.842 pp worse and shares
only 2.2 of 5 features with it). This validates the paper's choice rather than
replacing it, which is a result worth reporting either way.

**8. Repeated runs and significance.** Every figure in the paper, and until now in this
project, came from a single seed. Ten seeds across 15 subtypes and three arms, with
paired t tests and Wilcoxon signed-rank tests, give: tuning **+0.030 pp**, 95 percent CI
[-0.030, +0.090], which includes zero (t p=0.322, Wilcoxon p=0.025); Extra Trees over
Random Forest **+0.174 pp**, CI [0.088, 0.259], p=0.00005. Only 5 of 15 subtypes show a
tuning gain whose interval excludes zero, and Shade is significantly **worse** at
-0.39 pp.

**9. Explanation faithfulness, measured.** Limitation raised by references [9] and
[24]: explanation quality is asserted across this literature and never measured.
Ablating the top SHAP feature moves the model's output by 0.1902 against 0.0428 for a
random feature, a margin of +0.1474, and the top feature moves it more in 76.0 percent
of samples once 27.8 percent exact ties are excluded. Every one of 60 generated findings
names only features that are actually in its evidence payload (3.78 features per finding,
zero ungrounded) and states a confidence consistent with the payload. The measure carries
its own controls: deterministic template findings, which cannot invent a feature, score
zero false alarms, and three deliberately injected faults are all caught.

## Findings that correct our own earlier claims

Two results overturn what this project claimed before the repeated-run work, and both
are reported as such on the dashboard rather than quietly dropped.

- **Hyperparameter tuning does not meaningfully help.** The single-split run showed a
  mean +0.216 pp with CWS at +1.33 pp. Over ten seeds the mean falls to +0.030 pp with a
  confidence interval that includes zero, and CWS to +0.37 pp. The operating-point work
  reaches the same conclusion independently, by a different route.
- **The direction sub-check in the faithfulness measure was withdrawn.** It could only
  be made to pass by tightening it until it fired on roughly one finding in sixty, which
  measures the instrument and not the model. It is documented as a known limitation in
  `results/upgrade9_faithfulness_summary.json` instead of being reported as a pass.

## Project structure

```
malxplain/
├── data/               CIC-MalMem-2022 dataset
├── src/
│   ├── preprocessing.py     data preparation, paper Part I
│   ├── baseline.py          Stage 1, seven classifiers on known malware
│   ├── adaptive.py          Stage 2, 15 one-subtype detectors, zero-shot holdout
│   ├── paper_compare.py     our figures against the published ones
│   ├── tuning.py            hyperparameter search
│   ├── comparison.py        five classifier families, zero shot
│   ├── error_analysis.py    cross-subtype blind-spot analysis
│   ├── explainability.py    SHAP plus structured evidence payload
│   ├── reporter.py          LLM reporting layer (local Ollama plus template fallback)
│   ├── benchmark.py         inference cost, latency and model size
│   ├── operating_point.py   precision and FPR at realistic prevalence
│   ├── featsel.py           four feature selection methods compared
│   ├── significance.py      ten seeds, paired tests, confidence intervals
│   ├── faithfulness.py      explanation faithfulness, with instrument controls
│   ├── manifest.py          writes the results index
│   ├── charts.py            all dashboard charts
│   └── verify.py            reproduction report and leakage guards
├── scripts/setup.py    one-command bootstrap (downloads dataset, checks env)
├── app.py              Streamlit dashboard, ten sections
├── MalXplain_Implementation.ipynb   Phase 2 Colab implementation notebook
├── docs/               phase reports, literature review, block diagram
├── results/            all CSV and JSON outputs, plus RESULTS_INDEX.md
├── figures/            SHAP plots
└── models/             trained detectors and feature glossary
```

## Running it

```bash
pip install -r requirements.txt

python -m src.preprocessing      # verify dataset integrity
python -m src.baseline           # Stage 1          (~3 min)
python -m src.adaptive           # Stage 2          (~4 min)
python -m src.paper_compare      # paper comparison (~1 min)
python -m src.tuning             # tuning           (~8 min)
python -m src.comparison         # classifier families (~6 min)
python -m src.error_analysis     # blind spots      (~4 min)
python -m src.explainability     # SHAP             (~2 min)
python -m src.featsel            # feature selection (~12 min)
python -m src.operating_point    # operating point  (~6 min)
python -m src.significance       # ten seeds        (~35 min)
python -m src.benchmark          # inference cost   (~25 min)
python -m src.faithfulness       # faithfulness     (~15 min)
python -m src.verify             # full reproduction report
python -m src.manifest           # regenerate results/RESULTS_INDEX.md
```

Set `MALXPLAIN_MACHINE_LABEL` before `src.benchmark` so its output files are named for
the machine they were measured on. Timings are specific to that machine; the ratio
columns are the part that carries across hardware.

## Reproducibility notes

- All experiments use `random_state=42`, except `src.significance`, which varies the
  split seed on purpose across 10 values.
- Hyperparameter search and every feature selection method are cross-validated **within
  the training split only**; the unseen-subtype holdout is never used for model
  selection.
- `src/verify.py` asserts train/test disjointness for all 15 splits and confirms every
  training set is balanced 50/50, so reported gains are not leakage artefacts.
- `src.benchmark` pins BLAS and OpenMP threading to one core before importing
  scikit-learn, discards a warm-up pass and reports the median of 10 repeats, so the
  comparison reflects the work each model does rather than the parallelism each library
  happened to grab.
- Absolute inference timings are not comparable to the paper's hardware
  (Intel i5-1035G1). Ratios between families measured on one machine are, and model size
  in kilobytes is directly comparable.

## Hosted dashboard

The dashboard is deployed on Streamlit Community Cloud and redeploys on every push.
See DEPLOY.md for setup. Ten sections are available:

1. Overview, what the project does and what was added beyond the base paper
2. Reproduction results, our figures against the published ones
3. Hyperparameter tuning across all fifteen detectors, with the repeated-run correction
4. Classifier comparison, five families under the zero-shot protocol
5. Blind spot analysis, the fifteen by fifteen cross-subtype matrix
6. Explainability, SHAP evidence, generated findings and the faithfulness measure
7. Feature selection, four methods against the paper's one
8. Operating point, precision and false alarm rate at realistic prevalence
9. Inference cost, latency and model size on two machines
10. Live detection, interactive sample analysis

Every page except Live detection reads precomputed result files, so the hosted version
loads instantly without retraining anything.

The locally hosted language model requires Ollama on the machine running the app. A
hosted deployment has no Ollama server, so findings fall back to the deterministic
template and the interface states which path was used.

**Local LLM (recommended for the demo).** Install [Ollama](https://ollama.com) and pull
a small model:

```bash
ollama pull llama3.2:3b
python -m streamlit run app.py
```

Ollama binds `127.0.0.1:11434`. The app requests that address directly rather than
`localhost`, because on Windows `localhost` commonly resolves to `::1` first, where
nothing is listening, which silently sends every finding down the template path.

## Dataset

CIC-MalMem-2022, Canadian Institute for Cybersecurity, University of New Brunswick.
58,596 memory dump records, 29,298 benign and 29,298 malware across 15 subtypes.
Obtained via the base paper's own public repository:
https://github.com/Adnane017/Detecting_new_obfuscated_malware_variants

## Design

Navy and copper on a warm off-white surface. Navy carries interface structure, copper
marks emphasis and key figures.

Navy is deliberately never used as a chart series colour. It sits outside the validated
lightness band and below the chroma floor, so as a data mark it would read as grey.
Chart series use blue #2F6FAD and copper #C8622B, a pair validated for colour vision
deficiency separation (delta E 19.8 protan, 27.0 normal vision) against the page
surface.

The cross-subtype matrix encodes miss rate rather than recall. Recall sits between 95
and 100 percent for most cells, so shading recall leaves a uniform block with the
failures pale and easy to overlook, which defeats the purpose of the chart. Miss rate
keeps the convention that darker means a larger value and puts the ink on the blind
spots.

The paper-comparison chart plots signed deviation about a zero line rather than paired
bars, because paired bars in the 0.991 to 1.000 range are visually identical and hide
the very thing the chart exists to show.
