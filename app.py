"""
MalXplain dashboard.

An Improved Lightweight and Interpretable Machine Learning Approach for
Detecting New Obfuscated Malware Variants.

Pravara Jain (K020), Yash Mehta (K035), Somya Soni (K063)
B.Tech Cybersecurity, NMIMS MPSTME, Mumbai

Run locally with:  python -m streamlit run app.py

Every page except "Live detection" reads precomputed result files, so the dashboard
loads instantly and can be hosted without retraining anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts
from src.theme import CSS, masthead, note, section, tiles

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

st.set_page_config(page_title="MalXplain", page_icon="🛡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

H = lambda s: st.markdown(s, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load(name: str, **kw):
    p = RESULTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def missing(what: str):
    H(note(f"<strong>{what}</strong> is not available. Run <code>python -m src.verify</code> "
           "to regenerate every result file this page needs.", "flag"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = [
    "Overview",
    "Reproduction results",
    "Hyperparameter tuning",
    "Classifier comparison",
    "Blind spot analysis",
    "Explainability",
    "Live detection",
]

with st.sidebar:
    H('<div class="sidebar-brand">MalXplain</div>'
      '<div class="sidebar-sub">Zero shot malware detection</div>')
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    H('<div class="sidebar-note">'
      '<strong>Base paper</strong><br>Madamidola, Ngobigha and Ez-zizi (2024)<br>'
      '<em>Intelligent Systems with Applications</em> 25, 200472'
      '<br><br><strong>Dataset</strong><br>CIC-MalMem-2022<br>58,596 memory dump records'
      '<br><br><strong>Team</strong><br>Pravara Jain · K020<br>Yash Mehta · K035<br>'
      'Somya Soni · K063</div>')


# =========================================================================== #
# Overview
# =========================================================================== #
if page == "Overview":
    H(masthead(
        "MalXplain",
        "A malware detector is trained on one single subtype, roughly 6.6 percent of the "
        "dataset, and then tested against all fourteen subtypes it has never seen. This "
        "project reproduces that result from the base paper, closes three of the four "
        "limitations the original authors state themselves, and produces a readable finding "
        "instead of a raw plot.",
        "Pravara Jain (K020) · Yash Mehta (K035) · Somya Soni (K063) &nbsp;|&nbsp; "
        "B.Tech Cybersecurity, NMIMS MPSTME, Mumbai",
        eyebrow="Software Engineering Project",
    ))

    s2 = load("stage2_adaptive.csv")
    if s2 is not None:
        matched = s2.dropna(subset=["Paper Accuracy"])
        H(tiles([
            ("Subtype detectors", "15", "each trained on one family only", False),
            ("Above 99 percent", f"{int((s2['Accuracy'] > 0.99).sum())} of 15",
             "the base paper reports 11 of 15", True),
            ("Deviation from paper", f"{matched['Δ vs paper'].abs().mean():.4f}",
             "mean absolute difference", False),
            ("Smallest detector", f"{s2['Model size (KB)'].min():.0f} KB",
             "lightweight enough to embed", False),
        ]))

    H(section(
        "The problem",
        "Signature based defences only catch what has already been catalogued. Published "
        "machine learning results on malware datasets routinely exceed 99 percent, but those "
        "figures come from a random split in which every malware family appears in both "
        "halves. That measures how well a model recognises families it has already seen, not "
        "how it copes with something new. The withheld subtype protocol used here removes an "
        "entire family from training so the detector genuinely meets it for the first time.",
        kicker="Context",
    ))

    H(section("What was added beyond the base paper", kicker="Contribution"))
    c1, c2 = st.columns(2)
    with c1:
        H('<div class="card"><strong>1. Hyperparameter tuning</strong><br>'
          '<span style="color:#6B7280;font-size:.88rem;line-height:1.6">The authors used '
          'library defaults and listed this as their first limitation. Tuning now runs across '
          'all fifteen detectors, cross validated inside the training split only.</span></div>')
        H('<div class="card"><strong>3. Cross subtype error analysis</strong><br>'
          '<span style="color:#6B7280;font-size:.88rem;line-height:1.6">The authors state '
          'they performed none. A full fifteen by fifteen matrix, 225 measurements, exposes a '
          'systematic blind spot that aggregate accuracy conceals entirely.</span></div>')
    with c2:
        H('<div class="card"><strong>2. Classifier comparison</strong><br>'
          '<span style="color:#6B7280;font-size:.88rem;line-height:1.6">The paper tests seven '
          'classifiers on known malware but only Random Forest on unseen subtypes, so it '
          'cannot say whether Random Forest is actually the right choice.</span></div>')
        H('<div class="card"><strong>4. A readable finding</strong><br>'
          '<span style="color:#6B7280;font-size:.88rem;line-height:1.6">SHAP output is a plot '
          'a human must interpret. A locally hosted language model turns the same evidence '
          'into prose, without ever touching the detection decision.</span></div>')

    H(note("<strong>Not closed:</strong> cross dataset validation. It is the base paper's own "
           "first future work item, it needs a second memory forensics dataset with compatible "
           "features, and it remains open. Stating that plainly is more useful than a weak "
           "attempt at it.", "flag"))

    H(section(
        "Verification",
        "Every figure on these pages is reproducible from the code in this repository. The "
        "verification script confirms the dataset matches the paper's record counts exactly, "
        "that no sample appears in both the training split and the holdout for any of the "
        "fifteen detectors, and that every training set is balanced fifty fifty.",
        kicker="Integrity",
    ))
    rep = RESULTS / "verification_report.txt"
    if rep.exists():
        with st.expander("Read the full verification report"):
            st.code(rep.read_text(), language="text")


# =========================================================================== #
# Reproduction
# =========================================================================== #
elif page == "Reproduction results":
    H(masthead(
        "Reproduction results",
        "Before improving on a paper you have to show you can reproduce it. Stage one "
        "establishes that the implementation is correct on known malware. Stage two runs the "
        "paper's actual contribution, the zero shot experiment.",
        "Against Madamidola, Ngobigha and Ez-zizi (2024), Tables 3 and 4",
        eyebrow="Fidelity check",
    ))

    H(section("Stage one: seven classifiers on known malware",
              "An eighty twenty stratified split with every subtype present in both halves, "
              "plus ten fold cross validation.", kicker="Baseline"))
    s1 = load("stage1_baseline_holdout.csv")
    if s1 is None:
        missing("The Stage 1 result file")
    else:
        st.dataframe(
            s1[["Classifier", "Accuracy", "Precision", "Recall", "F1 Score",
                "Paper Accuracy", "Δ vs paper"]],
            hide_index=True, width="stretch")
        H(note(f"Largest deviation from the published figures: "
               f"<strong>{s1['Δ vs paper'].abs().max():.4f}</strong>. Random Forest is the "
               "strongest classifier in both our run and theirs.", "key"))

    H(section("Stage two: trained on one subtype, tested on fourteen unseen ones",
              "Each detector sees eighty percent of a single subtype plus matched benign "
              "samples. Its holdout contains every other subtype and all unused benign data.",
              kicker="Zero shot"))
    s2 = load("stage2_adaptive.csv")
    if s2 is None:
        missing("The Stage 2 result file")
    else:
        st.pyplot(charts.reproduction_chart(s2))
        st.dataframe(
            s2[["Subtype", "Accuracy", "Paper Accuracy", "Δ vs paper",
                "Unseen-subtype recall", "F1 Score", "FNR %", "FPR %",
                "Model size (KB)", "us/file"]],
            hide_index=True, width="stretch")
        H(tiles([
            ("Above 99 percent", f"{int((s2['Accuracy'] > 0.99).sum())} of 15",
             "the paper reports the same", True),
            ("Weakest subtype", s2.iloc[-1]["Subtype"],
             f"{s2.iloc[-1]['Accuracy']*100:.2f} percent, weakest in the paper too", False),
            ("Transponder model", f"{s2[s2['Subtype']=='Transponder'].iloc[0]['Model size (KB)']:.0f} KB",
             "the paper reports about 340 KB", False),
        ]))


# =========================================================================== #
# Tuning
# =========================================================================== #
elif page == "Hyperparameter tuning":
    H(masthead(
        "Hyperparameter tuning",
        "The base paper states: \"we did not perform hyperparameter tuning, instead opting "
        "for default hyperparameter settings for our classifiers\". This closes that gap. The "
        "search is cross validated inside the training split only, so the unseen subtype "
        "holdout is never used for model selection.",
        "Upgrade one, closing the authors' first stated limitation",
        eyebrow="Upgrade 01",
    ))

    t = load("upgrade1_tuning.csv")
    if t is None:
        missing("The tuning result file")
    else:
        H(tiles([
            ("Improved on unseen recall", f"{int((t['Δ unseen recall'] > 0).sum())} of 15",
             "the metric that matters for zero shot", True),
            ("Mean change", f"{t['Δ unseen recall'].mean()*100:+.3f} pp", "unseen subtype recall", False),
            ("Largest single gain", f"{t['Δ unseen recall'].max()*100:+.2f} pp",
             "concentrated in the weakest detectors", False),
            ("Improved on accuracy", f"{int((t['Δ accuracy'] > 0).sum())} of 15",
             "a more modest picture", False),
        ]))
        st.pyplot(charts.tuning_chart(t))
        st.dataframe(
            t[["Subtype", "Default accuracy", "Tuned accuracy", "Δ accuracy",
               "Default unseen recall", "Tuned unseen recall", "Δ unseen recall"]],
            hide_index=True, width="stretch")
        H(note(
            "<strong>Reported honestly.</strong> Accuracy improved on only seven of fifteen "
            "subtypes and the mean gain is small. The gains concentrate exactly where the "
            "baseline was weakest, CWS and Scar, while already strong detectors shift slightly "
            "the other way. Because results sit near the ceiling, differences of a few "
            "hundredths of a percentage point fall inside normal seed variance and are not "
            "claimed as improvements.", "flag"))


# =========================================================================== #
# Classifier comparison
# =========================================================================== #
elif page == "Classifier comparison":
    H(masthead(
        "Is Random Forest actually the right choice?",
        "The base paper evaluates seven classifiers on known malware but runs the zero shot "
        "experiment with Random Forest alone. It therefore cannot answer whether Random Forest "
        "generalises unusually well, or whether any reasonable classifier would do the same. "
        "Five families run here through the identical protocol, same splits, same features.",
        "Upgrade two, a question the base paper structurally could not answer",
        eyebrow="Upgrade 02",
    ))

    summary = load("upgrade2_comparison_summary.csv", index_col=0)
    full = load("upgrade2_comparison_full.csv")
    if summary is None:
        missing("The classifier comparison file")
    else:
        ranked = summary.sort_values("Mean_unseen_recall", ascending=False)
        rank = list(ranked.index).index("Random Forest") + 1
        H(tiles([
            ("Best mean recall", ranked.index[0], f"{ranked.iloc[0]['Mean_unseen_recall']*100:.2f} percent", True),
            ("Random Forest rank", f"{rank} of 5", "the base paper's only choice here", False),
            ("Best worst case", summary.sort_values("Min_unseen_recall", ascending=False).index[0],
             "fails least badly on the hardest subtype", False),
        ]))
        st.pyplot(charts.classifier_chart(summary))
        st.dataframe(summary.round(4), width="stretch")
        H(note(
            "Extra Trees edges out Random Forest on mean unseen subtype recall, and the MLP "
            "has the best worst case, meaning it fails least badly on the hardest subtype. "
            "<strong>Random Forest is a sound default, not an optimal one.</strong> Reaching "
            "that conclusion requires running the comparison in the zero shot setting, which "
            "the base paper did not do.", "key"))
        if full is not None:
            with st.expander("Per subtype results across all five classifiers"):
                st.dataframe(full, hide_index=True, width="stretch")


# =========================================================================== #
# Blind spot analysis
# =========================================================================== #
elif page == "Blind spot analysis":
    H(masthead(
        "Where does zero shot detection actually fail?",
        "The base paper reports one aggregate accuracy per detector and states that no error "
        "analysis was performed. Aggregate accuracy hides the operational question: which "
        "unseen subtypes slip past which detector? Every one of the fifteen detectors was "
        "tested against every one of the fifteen subtypes, giving 225 measurements.",
        "Upgrade three, the project's principal finding",
        eyebrow="Upgrade 03",
    ))

    matrix = load("upgrade3_cross_subtype_recall.csv", index_col=0)
    hardest = load("upgrade3_hardest_subtypes.csv", index_col=0)
    ranking = load("upgrade3_detector_ranking.csv", index_col=0)
    worst = load("upgrade3_worst_blindspots.csv")
    diag = load("upgrade3_blindspot_diagnosis.csv")

    if matrix is None:
        missing("The cross subtype matrix")
    else:
        H(tiles([
            ("Hardest unseen subtype", "TIBS", "87.8 percent mean recall across 14 detectors", True),
            ("Worst single pairing", "Zeus to TIBS", "36 percent of samples missed", True),
            ("Most transferable", "Transponder", "99.93 percent mean recall on unseen data", False),
            ("Least transferable", "Zeus", "95.27 percent, despite being easy to detect", False),
        ]))
        st.pyplot(charts.blindspot_heatmap(matrix))
        H('<div style="font-size:.78rem;color:#6B7280;margin-top:-.4rem">'
          'Cells where more than five percent was missed carry a label. The diagonal is blank '
          'because there the subtype was seen during training.</div>')

        H(section("The finding", kicker="Result"))
        H(note(
            "<strong>TIBS is a systematic blind spot.</strong> Averaged across the fourteen "
            "detectors that never saw it, TIBS is caught at only 87.8 percent, against 96.7 "
            "percent or better for every other subtype. For the Zeus, Ako and Refroso detectors "
            "it collapses to roughly 64 percent, meaning more than a third of TIBS samples pass "
            "as benign. None of this is visible in the aggregate accuracy figures the base "
            "paper reports.", "key"))
        H(note(
            "<strong>There is also an asymmetry worth noting.</strong> Zeus is among the "
            "easiest subtypes to detect when unseen, at 99.94 percent, yet it is the worst "
            "subtype to train a detector on, at 95.27 percent mean recall. Being easy to "
            "recognise and being a good teacher are different properties.", "flag"))

        c1, c2 = st.columns([1.15, 1])
        with c1:
            H('<div class="sec"><h2 style="font-size:1.05rem">Hardest subtypes when unseen</h2></div>')
            if hardest is not None:
                st.pyplot(charts.hardest_chart(hardest.iloc[:, 0]))
        with c2:
            H('<div class="sec"><h2 style="font-size:1.05rem">Ten worst detector and target pairs</h2></div>')
            if worst is not None:
                st.dataframe(worst, hide_index=True, width="stretch", height=250)
            H('<div class="sec"><h2 style="font-size:1.05rem">Most transferable detectors</h2></div>')
            if ranking is not None:
                st.dataframe(ranking.round(4), width="stretch", height=220)

        if diag is not None:
            H(section("Feature level cause",
                      "Comparing the feature values of missed TIBS samples against detected ones, "
                      "under the Zeus trained detector.", kicker="Diagnosis"))
            st.dataframe(diag, hide_index=True, width="stretch")
            H(note(
                "Missed TIBS samples carry a noticeably higher average DLL count per process "
                "than detected ones, 0.79 against 0.68 on the scaled range. They load more "
                "libraries per process, which is what ordinary software does, so they read as "
                "benign to a detector trained on Zeus."))


# =========================================================================== #
# Explainability
# =========================================================================== #
elif page == "Explainability":
    H(masthead(
        "From evidence to a readable finding",
        "The base paper produces a SHAP plot and leaves an analyst to interpret it. MalXplain "
        "serialises the same evidence into a structured payload and hands it to a locally "
        "hosted language model, which writes a short finding in plain language.",
        "Upgrade four, the prototype's distinguishing feature",
        eyebrow="Upgrade 04",
    ))

    H(section("Global explanation",
              "SHAP values over the top five features of the Transponder detector, computed on "
              "a sample of the holdout. This reproduces the base paper's global interpretation.",
              kicker="SHAP"))
    bees = FIGURES / "shap_global_beeswarm.png"
    if bees.exists():
        st.image(str(bees), width="stretch")
        H('<div style="font-size:.78rem;color:#6B7280">Total Windows services running and '
          'services hosted inside shared processes dominate, matching the finding reported in '
          'the base paper.</div>')
    else:
        missing("The SHAP beeswarm figure")

    H(section("Three deliberate design decisions", kicker="Architecture"))
    d1, d2, d3 = st.columns(3)
    with d1:
        H('<div class="card"><strong>The model never decides</strong><br>'
          '<span style="color:#6B7280;font-size:.87rem;line-height:1.6">It receives a '
          'prediction that has already been computed and only narrates the evidence behind it. '
          'The detection path stays statistical and auditable.</span></div>')
    with d2:
        H('<div class="card"><strong>It never sees the answer</strong><br>'
          '<span style="color:#6B7280;font-size:.87rem;line-height:1.6">The ground truth label '
          'and true subtype are withheld from the prompt, so the finding explains the decision '
          'rather than restating the answer.</span></div>')
    with d3:
        H('<div class="card"><strong>Inference stays local</strong><br>'
          '<span style="color:#6B7280;font-size:.87rem;line-height:1.6">No sample, feature '
          'vector or finding leaves the machine, which is what makes the approach usable in a '
          'regulated or air gapped environment.</span></div>')

    ex = load_json("explanation_examples.json")
    if ex:
        H(section("Worked examples", kicker="Evidence"))
        labels = {
            "unseen_malware_detected": "Unseen subtype, correctly detected",
            "benign_correct": "Benign, correctly classified",
            "benign_false_positive": "A genuine false positive",
        }
        for tab, (k, v) in zip(st.tabs([labels.get(k, k) for k in ex]), ex.items()):
            with tab:
                a, b, c = st.columns(3)
                a.metric("Verdict", v["prediction"])
                b.metric("Confidence", f"{v['confidence']:.1%}")
                c.metric("Ground truth", f"{v['ground_truth']} · {v['actual_subtype']}")
                st.dataframe(pd.DataFrame(v["evidence"]), hide_index=True, width="stretch")
        H(note("The false positive is included on purpose. A demonstration that shows only "
               "successes tells an analyst nothing about how the system fails.", "flag"))


# =========================================================================== #
# Live detection
# =========================================================================== #
elif page == "Live detection":
    from src.explainability import build_explainer, explain_instance
    from src.preprocessing import prepare
    from src.reporter import (
        DEFAULT_MODEL, build_prompt, generate_finding, list_models, ollama_available,
    )

    @st.cache_resource(show_spinner="Loading the dataset and training the reference detector...")
    def load_context(subtype: str):
        ds = prepare(verbose=False)
        return ds, build_explainer(ds, subtype)

    @st.cache_data(show_spinner=False)
    def sample_pool(subtype: str):
        _, ctx = load_context(subtype)
        s = ctx["test_subtype"]
        pool = []
        for sub in sorted(s.unique()):
            pool.extend((i, sub) for i in s[s == sub].index[:40])
        return pool

    H(masthead(
        "Live detection",
        "Select a memory dump sample from the holdout. The detector classifies it, SHAP "
        "explains the decision, and the local language model writes the finding. Samples whose "
        "subtype differs from the detector's training subtype are the zero day analogues.",
        "Interactive prototype",
        eyebrow="Demonstration",
    ))

    c1, c2 = st.columns(2)
    with c1:
        detector = st.selectbox(
            "Detector trained on",
            ["Transponder", "Reconyc", "Gator", "TIBS", "Emotet", "Zeus"],
            help="Each detector is trained on this subtype only. Every other subtype is unseen.")
    with c2:
        online = ollama_available()
        if online:
            models = list_models()
            model_name = st.selectbox(
                "Local model", models or [DEFAULT_MODEL],
                index=(models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0))
        else:
            model_name = DEFAULT_MODEL
            st.selectbox("Local model", ["not reachable"], disabled=True)

    if online:
        H(note("Ollama is reachable. Findings are generated locally on this machine.", "key"))
    else:
        H(note("No local model server is reachable, which is expected on a hosted deployment. "
               "Findings fall back to a deterministic template and the source is stated below "
               "each finding.", "flag"))

    ds, ctx = load_context(detector)
    pool = sample_pool(detector)
    labels = {i: f"sample {i} · {sub}" + ("   (seen in training)" if sub == detector else "")
              for i, sub in pool}

    a, b = st.columns([3, 1])
    with a:
        choice = st.selectbox("Holdout sample", [i for i, _ in pool],
                              format_func=lambda i: labels[i], label_visibility="collapsed")
    with b:
        run = st.button("Analyse sample", type="primary", width="stretch")

    if not run:
        H(note("Choose a sample whose subtype differs from the detector's training subtype, "
               "then press Analyse. Those are the cases where the model has never seen that "
               "malware family before."))
        st.stop()

    t0 = time.perf_counter()
    explanation = explain_instance(ctx, choice)
    elapsed = (time.perf_counter() - t0) * 1000

    H(section("Detection result", kicker="Step 1"))
    verdict, truth = explanation["prediction"], explanation["ground_truth"]
    correct = verdict == truth
    H(tiles([
        ("Verdict", verdict, "", verdict == "Malware"),
        ("Confidence", f"{explanation['confidence']:.1%}", "", False),
        ("Ground truth", f"{truth}", explanation["actual_subtype"], False),
        ("Seen in training", "Yes" if explanation["seen_in_training"] else "No",
         f"analysed in {elapsed:.0f} ms", False),
    ]))

    if correct and not explanation["seen_in_training"] and truth == "Malware":
        st.success(f"Correctly detected an unseen subtype, {explanation['actual_subtype']}, "
                   f"using a detector trained only on {detector}.")
    elif correct:
        st.success("Correct classification.")
    elif truth == "Benign":
        st.error("False positive. A benign sample was flagged as malware.")
    else:
        st.error("False negative. Malware was missed.")

    H(section("Evidence", "Each feature's contribution to this specific decision.",
              kicker="Step 2"))
    ev = pd.DataFrame(explanation["evidence"])
    e1, e2 = st.columns([1.05, 1])
    with e1:
        st.dataframe(
            ev[["description", "scaled_value", "shap_value", "pushes_towards"]].rename(
                columns={"description": "Feature", "scaled_value": "Value (scaled)",
                         "shap_value": "SHAP contribution", "pushes_towards": "Pushes towards"}),
            hide_index=True, width="stretch")
    with e2:
        st.pyplot(charts.shap_bar(ev))

    H(section("Analyst finding", kicker="Step 3"))
    with st.spinner("Generating the finding..."):
        result = generate_finding(explanation, model=model_name)
    H(f'<div class="finding">{result["finding"]}</div>')
    src = (f'Generated locally by <code>{result["model"]}</code> through Ollama.'
           if result["source"] == "llm" else
           "Deterministic template used, because no local model server was reachable.")
    H(f'<div style="font-size:.75rem;color:#6B7280;margin-top:.5rem">{src}</div>')

    with st.expander("Exactly what the model received (audit trail)"):
        st.caption("The model is never given the ground truth label or the true subtype. "
                   "It sees only the detector's own output and the SHAP evidence.")
        st.code(build_prompt(explanation), language="text")


H('<div class="foot">MalXplain &nbsp;·&nbsp; An Improved Lightweight and Interpretable Machine '
  'Learning Approach for Detecting New Obfuscated Malware Variants &nbsp;·&nbsp; '
  'B.Tech Cybersecurity, NMIMS MPSTME &nbsp;·&nbsp; All figures reproducible from the '
  'accompanying repository.</div>')
