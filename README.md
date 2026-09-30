# MalXplain

**An Improved Lightweight and Interpretable Machine Learning Approach for Detecting New Obfuscated Malware Variants**

B.Tech Cybersecurity, Software Engineering Project
Pravara Jain (K020) · Yash Mehta (K035) · Somya Soni (K063)
NMIMS MPSTME, Mumbai

---

## What this is

MalXplain reproduces and then improves on:

> Madamidola, O.A., Ngobigha, F., & Ez-zizi, A. (2024). *Detecting new obfuscated
> malware variants: A lightweight and interpretable machine learning approach.*
> Intelligent Systems with Applications, 25, 200472.
> https://doi.org/10.1016/j.iswa.2024.200472

The base paper trains 15 Random Forest detectors, each on a **single** malware subtype,
and shows they still detect the other 14 **unseen** subtypes, a proxy for zero-day
detection. MalXplain reproduces that result, closes four limitations the authors
themselves list, and wraps the outcome in a runnable prototype that produces a
plain-language finding instead of a raw SHAP plot.

## Reproduction status

| Check | Paper | MalXplain | Status |
|---|---|---|---|
| Dataset records | 58,596 | 58,596 | exact match |
| Subtype distribution (Table 2) | 15 subtypes | 15 subtypes | all 15 counts match exactly |
| Features after removing invariants | 52 | 52 | invariant features detected independently |
| Stage 1 best classifier | Random Forest | Random Forest | max deviation 0.0006 |
| Stage 2 models above 99% | 11 / 15 | 11 / 15 | exact match |
| Stage 2 mean absolute deviation |, | 0.0023 | faithful |
| Transponder model size | ~340 KB | 346.88 KB | consistent |
| Lowest-performing subtype | Zeus (98%) | Zeus (97.95%) | same subtype |

## Improvements over the base paper

Each addresses a limitation the authors state explicitly in their own paper.

**1. Hyperparameter tuning**, the paper used scikit-learn defaults and lists this
as its first limitation. Randomised search (30 candidates, 3-fold CV, inside the
training split only) improves unseen-subtype recall on 11/15 detectors, mean
+0.216 pp. Gains concentrate where the baseline was weakest: CWS +1.33 pp,
Scar +1.27 pp unseen recall.

**2. Classifier comparison in the zero-shot setting**, the paper tests seven
classifiers on known malware but only Random Forest on unseen subtypes, so it cannot
say whether RF's generalisation is special. Running the identical protocol across five
families shows **Extra Trees** edges out Random Forest on mean unseen recall
(0.9889 vs 0.9869), and **MLP has the best worst case** (0.9692 vs RF's 0.9591).
Random Forest ranks 2nd of 5, a good default, not an optimal one.

**3. Cross-subtype error analysis**, the paper reports aggregate accuracy and lists
"no error analysis" as a limitation. A full 15×15 detector-vs-target recall matrix
exposes a systematic blind spot invisible in aggregate figures: **TIBS** is detected at
only 87.8% mean recall when unseen, dropping to ~64% for the Zeus, Ako and Refroso
detectors. Feature-level diagnosis shows missed TIBS instances carry a markedly higher
`dlllist.avg_dlls_per_proc` (0.79 vs 0.68), i.e. they load more DLLs per process and so
look more like benign software.

**4. Explainability → plain-language findings**, the paper's SHAP output is a plot a
human must interpret. MalXplain serialises SHAP evidence into a structured payload and
has a **locally-hosted LLM (Ollama)** turn it into a readable finding, so the analyst
gets a conclusion rather than a chart.

## Project structure

```
malxplain/
├── data/               CIC-MalMem-2022 dataset
├── src/
│   ├── preprocessing.py    data preparation, paper Part I
│   ├── baseline.py         Stage 1, seven classifiers on known malware
│   ├── adaptive.py         Stage 2, 15 one-subtype detectors, zero-shot holdout
│   ├── tuning.py           Upgrade 1, hyperparameter search
│   ├── comparison.py       Upgrade 2, five classifier families, zero-shot
│   ├── error_analysis.py   Upgrade 3, cross-subtype blind-spot analysis
│   ├── explainability.py   Upgrade 4, SHAP plus structured evidence payload
│   ├── reporter.py         LLM reporting layer (local Ollama + template fallback)
│   └── verify.py           reproduction report and leakage guards
├── scripts/setup.py    one-command bootstrap (downloads dataset, checks env)
├── app.py              Streamlit prototype
├── results/            CSV outputs + verification_report.txt
├── figures/            SHAP plots
└── models/             trained detectors + feature glossary
```

## Running it

```bash
pip install -r requirements.txt

python -m src.preprocessing      # verify dataset integrity
python -m src.baseline           # Stage 1  (~3 min)
python -m src.adaptive           # Stage 2  (~4 min)
python -m src.tuning             # Upgrade 1 (~8 min)
python -m src.comparison         # Upgrade 2 (~6 min)
python -m src.error_analysis     # Upgrade 3 (~4 min)
python -m src.explainability     # Upgrade 4 (~2 min)
python -m src.verify             # full reproduction report
```

## Reproducibility notes

- All experiments use `random_state=42`.
- Hyperparameter search is cross-validated **within the training split only**; the
  unseen-subtype holdout is never used for model selection.
- `src/verify.py` asserts train/test disjointness for all 15 splits and confirms every
  training set is balanced 50/50, so reported gains are not leakage artefacts.
- Inference timings were measured on a 2-core cloud container and are not comparable
  to the paper's hardware (Intel i5-1035G1).

## Dataset

CIC-MalMem-2022, Canadian Institute for Cybersecurity, University of New Brunswick.
Obtained via the base paper's own public repository:
https://github.com/Adnane017/Detecting_new_obfuscated_malware_variants

## Running the dashboard

```bash
python scripts/setup.py          # downloads the dataset, verifies the environment
python -m streamlit run app.py   # launches the dashboard at http://localhost:8501
```

The app implements the scenario in the project's UML sequence diagram: select a
memory-dump sample, the detector classifies it, SHAP explains the decision, and a
locally-hosted LLM turns that evidence into a readable analyst finding.

**Local LLM (optional but recommended for the demo).** Install
[Ollama](https://ollama.com) and pull a small model:

```bash
ollama pull llama3.2:3b
```

Ollama runs a background service on `http://localhost:11434`; the app talks to it
directly. If it is unreachable the app falls back to a deterministic template, so the
prototype never hard-fails during a demo, it just says which path it used.

## Hosted dashboard

The dashboard is deployed on Streamlit Community Cloud and redeploys on every push.
See DEPLOY.md for setup. Seven sections are available:

1. Overview, what the project does and what was added beyond the base paper
2. Reproduction results, our figures against the published ones
3. Upgrade 1, hyperparameter tuning across all fifteen detectors
4. Upgrade 2, five classifier families under the zero shot protocol
5. Upgrade 3, the fifteen by fifteen cross subtype blind spot matrix
6. Upgrade 4, SHAP explainability and the generated findings
7. Live detection, interactive sample analysis

Every page except Live detection reads precomputed result files, so the hosted
version loads instantly without retraining anything.

The locally hosted language model requires Ollama on the machine running the app.
On a hosted deployment there is no Ollama server, so findings fall back to a
deterministic template and the interface states which path was used.

## Design

Navy and copper on a warm off-white surface. Navy carries interface structure,
copper marks emphasis and key figures.

Navy is deliberately never used as a chart series colour. It sits outside the
validated lightness band and below the chroma floor, so as a data mark it would read
as grey. Chart series use blue #2F6FAD and copper #C8622B, a pair validated for
colour vision deficiency separation (delta E 19.8 protan, 27.0 normal vision) against
the page surface.

The cross subtype matrix encodes miss rate rather than recall. Recall sits between 95
and 100 percent for most cells, so shading it leaves a uniform block with the failures
pale and easy to overlook, which defeats the purpose of the chart. Miss rate keeps the
convention that darker means a larger value and puts the ink on the blind spots.
