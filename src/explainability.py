"""
MalXplain: Upgrade 4: SHAP explainability layer.

Replicates Part III of Madamidola et al. (2024) (global beeswarm + local force plots)
and then goes beyond it: the SHAP output is serialised into a structured JSON payload
that the LLM reporting layer consumes to write a human-readable finding.

That serialisation is the bridge between the paper's contribution (a SHAP plot an
analyst must interpret themselves) and MalXplain's contribution (a plain-language
finding generated from those same SHAP values).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from .adaptive import RANDOM_STATE, TOP_K, select_top_features, split_for_subtype
from .preprocessing import Dataset, prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

# Human-readable descriptions of the memory-forensic features, used to make the
# LLM-generated findings meaningful rather than a list of raw column names.
FEATURE_GLOSSARY = {
    "svcscan.nservices": "total number of Windows services running",
    "svcscan.shared_process_services": "services hosted inside shared processes",
    "svcscan.kernel_drivers": "number of kernel drivers loaded",
    "svcscan.process_services": "services running in their own process",
    "svcscan.fs_drivers": "file-system drivers loaded",
    "svcscan.nactive": "number of active services",
    "handles.avg_handles_per_proc": "average OS handles held per process",
    "handles.nhandles": "total OS handles open",
    "handles.nevent": "event handles open",
    "handles.nmutant": "mutex handles open (often used for infection markers)",
    "handles.nsemaphore": "semaphore handles open",
    "handles.nkey": "registry key handles open",
    "handles.nthread": "thread handles open",
    "handles.nfile": "file handles open",
    "handles.ndirectory": "directory handles open",
    "handles.nsection": "memory-section handles open",
    "handles.ntimer": "timer handles open",
    "handles.ndesktop": "desktop object handles open",
    "pslist.nproc": "number of running processes",
    "pslist.nppid": "number of distinct parent process IDs",
    "pslist.avg_threads": "average threads per process",
    "pslist.avg_handlers": "average handlers per process",
    "dlllist.ndlls": "total DLLs loaded",
    "dlllist.avg_dlls_per_proc": "average DLLs loaded per process",
    "ldrmodules.not_in_load": "modules missing from the load list (hidden-module indicator)",
    "ldrmodules.not_in_init": "modules missing from the init list (hidden-module indicator)",
    "ldrmodules.not_in_mem": "modules missing from memory list (hidden-module indicator)",
    "malfind.ninjections": "suspected code-injection sites found",
    "malfind.commitCharge": "committed memory in injected regions",
    "malfind.protection": "memory protection flags on injected regions",
    "malfind.uniqueInjections": "unique code injections detected",
    "callbacks.ncallbacks": "kernel callbacks registered",
    "callbacks.nanonymous": "anonymous kernel callbacks",
    "modules.nmodules": "kernel modules loaded",
}


def describe(feature: str) -> str:
    return FEATURE_GLOSSARY.get(feature, feature)


def build_explainer(ds: Dataset, subtype: str = "Transponder") -> dict:
    """Train the reference detector and attach a SHAP TreeExplainer to it."""
    X_train, y_train, X_test, y_test, test_subtype = split_for_subtype(ds, subtype)
    features = select_top_features(X_train, y_train, TOP_K)

    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train[features], y_train)

    explainer = shap.TreeExplainer(clf)
    return {
        "subtype": subtype,
        "model": clf,
        "explainer": explainer,
        "features": features,
        "X_test": X_test[features],
        "y_test": y_test,
        "test_subtype": test_subtype,
    }


def _malware_shap(values: np.ndarray) -> np.ndarray:
    """Normalise SHAP output to the malware-class contribution matrix."""
    arr = np.asarray(values)
    if arr.ndim == 3:            # (n, features, classes)
        return arr[:, :, 1]
    return arr


def global_plot(ctx: dict, sample: int = 2000) -> Path:
    """SHAP beeswarm over the top-5 features (paper Figure 4 equivalent)."""
    FIGURES_DIR.mkdir(exist_ok=True)
    X = ctx["X_test"].sample(n=min(sample, len(ctx["X_test"])), random_state=RANDOM_STATE)
    sv = _malware_shap(ctx["explainer"].shap_values(X))

    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(9, 4.5))
    plt.title(f"MalXplain: global SHAP ({ctx['subtype']} detector, top-{len(ctx['features'])} features)",
              fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / "shap_global_beeswarm.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def explain_instance(ctx: dict, index) -> dict:
    """
    Produce the structured explanation payload for a single sample.

    This dict is exactly what the LLM reporting layer receives - no raw model
    internals, no free text, just the prediction and the ranked SHAP evidence.
    """
    row = ctx["X_test"].loc[[index]]
    proba = ctx["model"].predict_proba(row)[0]
    pred = int(ctx["model"].predict(row)[0])

    sv = _malware_shap(ctx["explainer"].shap_values(row))[0]

    contributions = sorted(
        (
            {
                "feature": f,
                "description": describe(f),
                "scaled_value": round(float(row.iloc[0][f]), 4),
                "shap_value": round(float(s), 4),
                "pushes_towards": "malware" if s > 0 else "benign",
            }
            for f, s in zip(ctx["features"], sv)
        ),
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return {
        "sample_id": str(index),
        "detector_trained_on": ctx["subtype"],
        "prediction": "Malware" if pred else "Benign",
        "confidence": round(float(proba[pred]), 4),
        "malware_probability": round(float(proba[1]), 4),
        "ground_truth": "Malware" if ctx["y_test"].loc[index] == 1 else "Benign",
        "actual_subtype": str(ctx["test_subtype"].loc[index]),
        "seen_in_training": bool(ctx["test_subtype"].loc[index] == ctx["subtype"]),
        "evidence": contributions,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    ds = prepare(verbose=False)

    print("=== Upgrade 4: SHAP explainability layer ===\n")
    ctx = build_explainer(ds, "Transponder")
    print(f"  reference detector : {ctx['subtype']}")
    print(f"  top-{TOP_K} features   : {ctx['features']}")

    path = global_plot(ctx)
    print(f"  global beeswarm    -> {path.name}")

    # Deliberately chosen examples: a correct detection of an unseen subtype, a correct
    # benign classification, and a genuine false positive. The false positive is kept on
    # purpose - a demo that only shows successes tells the analyst nothing about failure.
    y, st = ctx["y_test"], ctx["test_subtype"]
    pred_all = ctx["model"].predict(ctx["X_test"])

    unseen_mal = y[(y == 1) & (st != ctx["subtype"])].index
    benign = y[y == 0].index
    correct = pd.Series(pred_all, index=ctx["X_test"].index)

    hit = [i for i in unseen_mal if correct.loc[i] == 1][0]
    true_negative = [i for i in benign if correct.loc[i] == 0][0]
    false_positive = [i for i in benign if correct.loc[i] == 1][0]

    examples = {
        "unseen_malware_detected": explain_instance(ctx, hit),
        "benign_correct": explain_instance(ctx, true_negative),
        "benign_false_positive": explain_instance(ctx, false_positive),
    }
    (RESULTS_DIR / "explanation_examples.json").write_text(json.dumps(examples, indent=2))

    for name, ex in examples.items():
        print(f"\n  --- {name} ---")
        print(f"  predicted {ex['prediction']} ({ex['confidence']:.1%} confidence); "
              f"truth = {ex['ground_truth']} / {ex['actual_subtype']} "
              f"(seen in training: {ex['seen_in_training']})")
        for e in ex["evidence"][:3]:
            print(f"    {e['feature']:<38} shap={e['shap_value']:+.4f} -> {e['pushes_towards']}")

    # Persist the artefacts the Streamlit app and LLM layer will load.
    with open(MODELS_DIR / "reference_detector.pkl", "wb") as fh:
        pickle.dump(
            {"model": ctx["model"], "features": ctx["features"], "subtype": ctx["subtype"]}, fh
        )
    (MODELS_DIR / "feature_glossary.json").write_text(json.dumps(FEATURE_GLOSSARY, indent=2))
    print(f"\n  artefacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
