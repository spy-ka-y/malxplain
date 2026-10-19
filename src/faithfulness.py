"""
Explanation faithfulness: does the explanation describe what the model actually did?

This is the measurement our own contribution most needed. The project produces SHAP
attributions and then a written finding, and until now it asserted that both were
useful without testing either. Reference [9] in our literature review reports only
moderate explanation faithfulness for its own method, and [24] states plainly that its
summarisation scores are proxies for analyst usefulness rather than the thing itself.
No paper in the review measures faithfulness for a language model reporting layer.

Two independent checks are run.

A. Attribution faithfulness, by ablation.
   For each explained sample the feature SHAP ranked first is replaced by its training
   median, which is the neutral value carrying no evidence, and the prediction is taken
   again. The same is done for a randomly chosen other feature. If SHAP is faithful the
   top feature must move the prediction further than a random one, and it must move it
   in the direction its SHAP sign predicted. Both are reported, against the random
   baseline rather than in isolation, because a large movement means nothing without
   something to compare it to. Ties are counted separately: with five features and many
   saturated predictions, a sizeable fraction of samples do not move at all whichever
   feature is removed, and scoring those as failures would understate the result.

B. Report grounding, by parsing the generated finding.
   Three failure modes are counted: naming a memory feature that is not in the evidence
   payload at all, stating a confidence that does not match the payload, and asserting a
   direction of effect that contradicts the SHAP sign. A finding passing all three is
   grounded. This catches a language model that writes a plausible paragraph that the
   evidence does not support, which is the obvious objection to the whole reporting idea.

Note on which path is measured: with no local model service reachable, the reporting
layer falls back to its deterministic template and these figures describe that path.
Run this on a machine with Ollama up to measure the language model path, and the
results file records which one was measured.

Usage:  python -m src.faithfulness
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.adaptive import RANDOM_STATE, select_top_features, split_for_subtype
from src.explainability import (FEATURE_GLOSSARY, build_explainer, describe,
                                explain_instance)
from src.preprocessing import Dataset, prepare
from src.reporter import _template_finding, generate_finding, ollama_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

SAMPLES_PER_DETECTOR = 30
GROUNDING_SAMPLES = 60

# Only words describing the feature's VALUE. "more", "less", "greater" and "fewer"
# are left out on purpose: in "played a role, although to a lesser extent" they
# describe how big the contribution was, not whether the measurement was high or
# low, and treating them as direction claims produced false contradictions.
INCREASE_WORDS = ("higher", "elevated", "increased", "above",
                  "pushes towards malware", "unusually high")
DECREASE_WORDS = ("lower", "reduced", "decreased", "below",
                  "pushes towards benign", "unusually low")


# ------------------------------------------------- A. attribution ablation ----
def ablation_for_detector(ds: Dataset, subtype: str, rng: np.random.RandomState,
                          n: int = SAMPLES_PER_DETECTOR) -> list[dict]:
    ctx = build_explainer(ds, subtype)
    feats = ctx["features"]

    X_train, y_train, _, _, _ = split_for_subtype(ds, subtype)
    medians = X_train[feats].median()

    pool = ctx["X_test"].index.to_numpy()
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)

    rows = []
    for idx in chosen:
        exp = explain_instance(ctx, idx)
        top = exp["evidence"][0]
        others = [e["feature"] for e in exp["evidence"][1:]]
        if not others:
            continue
        rand_feature = str(rng.choice(others))

        row = ctx["X_test"].loc[[idx]]
        p_base = float(ctx["model"].predict_proba(row)[0][1])

        def _ablate(feature: str) -> float:
            mod = row.copy()
            mod[feature] = medians[feature]
            return float(ctx["model"].predict_proba(mod)[0][1])

        p_top = _ablate(top["feature"])
        p_rand = _ablate(rand_feature)

        shap_sign = np.sign(top["shap_value"])
        observed_sign = np.sign(p_base - p_top)

        rows.append({
            "Detector": subtype,
            "Sample": str(idx),
            "Top feature": top["feature"],
            "Top SHAP": top["shap_value"],
            "Base probability": p_base,
            "After removing top": p_top,
            "After removing random": p_rand,
            "Top |delta|": abs(p_base - p_top),
            "Random |delta|": abs(p_base - p_rand),
            "Top moved more": abs(p_base - p_top) > abs(p_base - p_rand),
            "Direction as SHAP predicted": bool(
                shap_sign == 0 or observed_sign == 0 or shap_sign == observed_sign),
        })
    return rows


# --------------------------------------------------- B. report grounding ------
# Words carrying no identifying power, dropped before token matching.
_STOP = {"of", "per", "the", "a", "an", "in", "inside", "their", "own", "often",
         "used", "for", "and", "or", "number", "total", "held", "open", "running",
         "loaded", "object", "distinct"}


def _feature_aliases(feature: str) -> tuple[str, str, list[str]]:
    """
    The strings that count as naming this feature in prose.

    The system prompt tells the model to use the plain-English description and never
    the raw column name, so matching on the column name alone finds nothing. Returns
    the raw name, the description with any parenthetical stripped, and its content
    tokens for an ordered window match that tolerates paraphrase.
    """
    desc = re.sub(r"\(.*?\)", "", describe(feature)).strip().lower()
    tokens = [t for t in re.findall(r"[a-z]+", desc)
              if t not in _STOP and len(t) > 2]
    return feature.lower(), desc, tokens


def _stem_pattern(token: str) -> str:
    """Prefix pattern so 'hosted' also matches 'hosting', 'host', 'hosts'."""
    keep = max(4, len(token) - 3)
    return r"\b" + re.escape(token[:keep]) + r"\w*"


def _best_span(tokens: list[str], text: str) -> tuple[int, int, int] | None:
    """
    Tightest span containing every token stem, order independent.

    Order independence matters: the glossary says "services hosted inside shared
    processes" while the model writes "hosting services inside shared processes".
    An ordered match misses that and then credits a different, more generic feature
    instead. Returns (start, end, matched token count), or None.
    """
    positions = []
    for tok in tokens:
        hits = [m.start() for m in re.finditer(_stem_pattern(tok), text)]
        if not hits:
            return None
        positions.append(hits)

    best = None
    for anchor in positions[0]:
        lo = hi = anchor
        for hits in positions[1:]:
            nearest = min(hits, key=lambda h: abs(h - anchor))
            lo, hi = min(lo, nearest), max(hi, nearest)
        if best is None or (hi - lo) < (best[1] - best[0]):
            best = (lo, hi, len(tokens))
    return best


def _mentioned_features(text: str, all_features: list[str]) -> list[str]:
    """
    Features this text names, resolving overlaps in favour of the specific.

    Two features can match the same words: "services hosted inside shared processes"
    contains the generic pair (services, process) belonging to a different feature.
    Candidates are therefore scored by how many tokens they matched and accepted
    greedily, so a four token match wins the span and the two token match that
    overlaps it is discarded rather than counted as a second, phantom mention.
    """
    lowered = text.lower()
    candidates = []
    for f in all_features:
        raw, desc, tokens = _feature_aliases(f)
        if raw in lowered:
            pos = lowered.find(raw)
            candidates.append((99, pos, pos + len(raw), f))
            continue
        if desc and desc in lowered:
            pos = lowered.find(desc)
            candidates.append((len(tokens) + 1, pos, pos + len(desc), f))
            continue
        if len(tokens) < 2:
            continue
        span = _best_span(tokens, lowered)
        if span is None:
            continue
        lo, hi, n = span
        if (hi - lo) <= max(40, len(desc) + 25):
            candidates.append((n, lo, hi, f))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    accepted, taken = [], []
    for n, lo, hi, f in candidates:
        if any(lo <= b and a <= hi for a, b in taken):
            continue
        taken.append((lo, hi))
        accepted.append(f)
    return accepted


# Phrases where a direction word describes the rank or size of a contribution rather
# than the measurement itself, e.g. "the lower-ranked features" or "to a lesser extent".
_NOT_A_VALUE_CLAIM = ("rank", "impact", "extent", "priority", "importance",
                      "contribution", "degree", "order", "shap")


def _direction_consistent(text: str, feature: str, scaled_value: float) -> bool | None:
    """
    Whether a stated direction for this feature matches its recorded value.

    An earlier version of this check compared the stated direction against the SHAP
    sign, which was wrong: a low measurement can push a prediction towards malware
    just as easily as a high one, so the two are independent and disagreement proved
    nothing. The claim that can actually be checked is whether the text describes the
    measurement the way the payload records it, against the midpoint of the scaled
    range. Returns None when no directional claim is made, which is not an error.
    """
    lowered = text.lower()
    raw, desc, tokens = _feature_aliases(feature)

    spots = [m.start() for m in re.finditer(re.escape(raw), lowered)]
    if desc:
        spots += [m.start() for m in re.finditer(re.escape(desc), lowered)]
    if not spots and len(tokens) >= 2:
        spots = [m.start() for m in re.finditer(_stem_pattern(tokens[0]), lowered)]
    if not spots:
        return None

    verdicts = []
    for pos in spots:
        # Confine the window to the sentence holding the mention. Reaching into the
        # next sentence credited a claim about different features entirely.
        left = max((lowered.rfind(c, 0, pos) for c in ".;"), default=-1) + 1
        right = min((r for r in (lowered.find(c, pos) for c in ".;") if r != -1),
                    default=len(lowered))
        window = lowered[max(left, pos - 90): min(right + 1, pos + 150)]
        says_up = any(w in window for w in INCREASE_WORDS)
        says_down = any(w in window for w in DECREASE_WORDS)
        if says_up == says_down:
            continue
        word = next((w for w in (INCREASE_WORDS + DECREASE_WORDS) if w in window), "")
        after = window[window.find(word) + len(word): window.find(word) + len(word) + 22]
        if any(k in after for k in _NOT_A_VALUE_CLAIM):
            continue  # describes the contribution, not the measurement
        verdicts.append(says_up == (scaled_value >= 0.5))

    if not verdicts:
        return None
    return all(verdicts)


def grounding_for_detector(ds: Dataset, subtype: str, all_features: list[str],
                           rng: np.random.RandomState, n: int) -> list[dict]:
    ctx = build_explainer(ds, subtype)
    pool = ctx["X_test"].index.to_numpy()
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)

    rows = []
    for idx in chosen:
        exp = explain_instance(ctx, idx)
        result = generate_finding(exp)
        text = result["finding"]

        payload_feats = {e["feature"] for e in exp["evidence"]}
        named = _mentioned_features(text, all_features)
        ungrounded = [f for f in named if f not in payload_feats]

        stated = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        expected = exp["confidence"] * 100
        confidence_ok = (not stated) or any(
            abs(float(s) - expected) < 0.5 for s in stated)

        checks = []
        for e in exp["evidence"]:
            verdict = _direction_consistent(text, e["feature"], e["scaled_value"])
            if verdict is not None:
                checks.append(verdict)
        direction_ok = all(checks) if checks else True

        rows.append({
            "Detector": subtype,
            "Sample": str(idx),
            "Source": result["source"],
            "Features named": len(named),
            "Ungrounded mentions": len(ungrounded),
            "Feature grounded": len(ungrounded) == 0,
            "Confidence grounded": confidence_ok,
            "Direction claims checked": len(checks),
            "Direction grounded": direction_ok,
            # Headline grounding rests on the two checks that survived calibration.
            # The value-direction check is recorded but excluded: see WITHDRAWN below.
            "Fully grounded": (len(ungrounded) == 0 and confidence_ok),
            "Value claim check (experimental)": direction_ok,
            "Finding": text.replace("\n", " ").strip(),
        })
    return rows


# ------------------------------------------------- calibrating the measure ----
def calibrate_matcher(ds: Dataset, all_features: list[str],
                      subtypes: list[str], per_subtype: int = 12) -> dict:
    """
    Check the grounding measure against text that cannot be ungrounded.

    The deterministic template is assembled from the evidence payload, so it is
    incapable of naming a feature outside it. Any ungrounded mention the matcher
    reports here is a false positive in the instrument, not a fault in the text.
    Running this before the real measurement is what makes the reported grounding
    rate worth quoting: an earlier, looser matcher scored the template at 0 percent
    grounded, which was entirely its own error.
    """
    named = ungrounded = n = 0
    for subtype in subtypes:
        ctx = build_explainer(ds, subtype)
        for idx in ctx["X_test"].index[:per_subtype]:
            exp = explain_instance(ctx, idx)
            text = _template_finding(exp)
            payload = {e["feature"] for e in exp["evidence"]}
            hits = _mentioned_features(text, all_features)
            named += len(hits)
            ungrounded += sum(1 for f in hits if f not in payload)
            n += 1
    # Negative control. A measure that reports nothing wrong is worthless unless it
    # can be shown to notice something wrong, so three faults are injected into a real
    # finding and each must be detected. Without this, a grounding rate of 100 percent
    # cannot be told apart from a measure that never fires.
    ctx = build_explainer(ds, subtypes[0])
    idx = ctx["X_test"].index[0]
    exp = explain_instance(ctx, idx)
    clean = _template_finding(exp)
    payload = {e["feature"] for e in exp["evidence"]}
    top = exp["evidence"][0]

    fabricated = clean + " The analysis also found an unusual number of mutex handles open."
    caught_feature = any(
        f not in payload for f in _mentioned_features(fabricated, all_features))

    miscited = re.sub(r"\d{1,3}(?:\.\d+)?\s*%", "73.4%", clean, count=1)
    stated = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", miscited)
    caught_confidence = bool(stated) and not any(
        abs(float(x) - exp["confidence"] * 100) < 0.5 for x in stated)

    wrong_way = "lower" if top["scaled_value"] >= 0.5 else "higher"
    contradicted = clean + (f" The {describe(top['feature'])} was unusually "
                            f"{wrong_way} in this sample.")
    caught_direction = _direction_consistent(
        contradicted, top["feature"], top["scaled_value"]) is False

    return {
        "control_samples": n,
        "control_mean_features_named": named / n if n else 0.0,
        "control_false_positives": ungrounded,
        "injected_feature_fault_caught": caught_feature,
        "injected_confidence_fault_caught": caught_confidence,
        "injected_direction_fault_caught": caught_direction,
        "matcher_trustworthy": (ungrounded == 0 and caught_feature
                                and caught_confidence and caught_direction),
    }


# ------------------------------------------------------------------ driver ----
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = prepare()
    rng = np.random.RandomState(RANDOM_STATE)
    all_features = list(ds.X.columns)

    online = ollama_available()
    print("=== Explanation faithfulness ===")
    print(f"  reporting path measured: "
          f"{'local language model' if online else 'deterministic template'}")
    print(f"  ablation samples per detector: {SAMPLES_PER_DETECTOR}\n")

    abl_path = RESULTS_DIR / "upgrade9_faithfulness_ablation.csv"
    grounding_only = "grounding" in sys.argv[1:]

    if grounding_only and abl_path.exists():
        abl = pd.read_csv(abl_path)
        print(f"  reusing {len(abl)} saved ablation results, "
              "re-running the grounding check only")
    else:
        ablation_rows = []
        for i, subtype in enumerate(ds.subtypes, 1):
            ablation_rows += ablation_for_detector(ds, subtype, rng)
            print(f"  [ablation {i:2d}/15] {subtype}")
        abl = pd.DataFrame(ablation_rows)
        abl.to_csv(abl_path, index=False)

    cal = calibrate_matcher(ds, all_features, ds.subtypes[:3])
    print(f"\n  matcher calibration on {cal['control_samples']} template findings: "
          f"{cal['control_false_positives']} false positives, "
          f"{cal['control_mean_features_named']:.2f} features named each")
    print(f"  injected faults detected: feature={cal['injected_feature_fault_caught']}, "
          f"confidence={cal['injected_confidence_fault_caught']}, "
          f"direction={cal['injected_direction_fault_caught']}")
    if not cal["matcher_trustworthy"]:
        print("  WARNING: the matcher reports ungrounded mentions on text that "
              "cannot contain any, so the grounding rates below are unreliable.")

    grounding_rows = []
    for subtype in ds.subtypes[:3]:
        grounding_rows += grounding_for_detector(
            ds, subtype, all_features, rng, GROUNDING_SAMPLES // 3)
        print(f"  [grounding] {subtype}")
    gnd = pd.DataFrame(grounding_rows)
    gnd.to_csv(RESULTS_DIR / "upgrade9_faithfulness_grounding.csv", index=False)

    summary = {
        "reporting_path": "local language model" if online else "deterministic template",
        "ablation_samples": len(abl),
        "mean_delta_top_feature": float(abl["Top |delta|"].mean()),
        "mean_delta_random_feature": float(abl["Random |delta|"].mean()),
        "faithfulness_margin": float(
            abl["Top |delta|"].mean() - abl["Random |delta|"].mean()),
        "top_moved_more_rate": float(abl["Top moved more"].mean()),
        "tie_rate": float((abl["Top |delta|"] == abl["Random |delta|"]).mean()),
        "top_moved_more_rate_excluding_ties": float(
            (abl.loc[abl["Top |delta|"] != abl["Random |delta|"], "Top |delta|"]
             > abl.loc[abl["Top |delta|"] != abl["Random |delta|"],
                       "Random |delta|"]).mean()),
        "direction_as_shap_predicted_rate": float(
            abl["Direction as SHAP predicted"].mean()),
        "grounding_samples": len(gnd),
        "feature_grounded_rate": float(gnd["Feature grounded"].mean()),
        "confidence_grounded_rate": float(gnd["Confidence grounded"].mean()),
        "value_claim_check_withdrawn": (
            "A third check compared stated value directions against the recorded "
            "scaled value. It is computed but NOT included in the grounding rate. "
            "Attributing a phrase such as 'the lower values of' to the right feature "
            "needs real parsing: successive versions either credited claims about "
            "SHAP magnitude and rank as claims about the measurement, or, once "
            "tightened, fired only once in 60 findings. One genuine error was found "
            "by reading the texts, recorded in the limitations section, but no rate "
            "from this check is trustworthy enough to report."),
        "direction_grounded_rate_not_reported": float(gnd["Direction grounded"].mean()),
        "fully_grounded_rate": float(gnd["Fully grounded"].mean()),
        "ungrounded_mentions_total": int(gnd["Ungrounded mentions"].sum()),
        "mean_features_named_per_finding": float(gnd["Features named"].mean()),
        "findings_naming_no_feature": int((gnd["Features named"] == 0).sum()),
        "direction_claims_tested": int(gnd["Direction claims checked"].sum()),
        **cal,
    }
    (RESULTS_DIR / "upgrade9_faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2))

    per_det = abl.groupby("Detector").agg(**{
        "Mean top delta": ("Top |delta|", "mean"),
        "Mean random delta": ("Random |delta|", "mean"),
        "Top moved more": ("Top moved more", "mean"),
        "Direction correct": ("Direction as SHAP predicted", "mean"),
    }).reset_index()
    per_det["Margin"] = per_det["Mean top delta"] - per_det["Mean random delta"]
    per_det.sort_values("Margin", ascending=False).to_csv(
        RESULTS_DIR / "upgrade9_faithfulness_per_detector.csv", index=False)

    print("\n--- A. Attribution faithfulness (ablation vs random baseline) ---")
    print(f"  mean probability shift, top SHAP feature removed : "
          f"{summary['mean_delta_top_feature']:.4f}")
    print(f"  mean probability shift, random feature removed   : "
          f"{summary['mean_delta_random_feature']:.4f}")
    print(f"  faithfulness margin                              : "
          f"{summary['faithfulness_margin']:+.4f}")
    print(f"  top feature moved prediction more often          : "
          f"{summary['top_moved_more_rate']:.1%}")
    print(f"  ties (prediction saturated, neither moved it)    : "
          f"{summary['tie_rate']:.1%}")
    print(f"  top feature won, ties excluded                   : "
          f"{summary['top_moved_more_rate_excluding_ties']:.1%}")
    print(f"  shift direction matched the SHAP sign            : "
          f"{summary['direction_as_shap_predicted_rate']:.1%}")

    print("\n--- B. Report grounding ---")
    print(f"  findings naming only payload features            : "
          f"{summary['feature_grounded_rate']:.1%}")
    print(f"  findings whose stated confidence matches         : "
          f"{summary['confidence_grounded_rate']:.1%}")
    print(f"  value-direction check (WITHDRAWN, not reported)  : "
          f"{summary['direction_grounded_rate_not_reported']:.1%} "
          f"over {summary['direction_claims_tested']} claims")
    print(f"  fully grounded                                   : "
          f"{summary['fully_grounded_rate']:.1%}")
    print(f"  ungrounded feature mentions in total             : "
          f"{summary['ungrounded_mentions_total']}")
    print(f"  mean features named per finding                  : "
          f"{summary['mean_features_named_per_finding']:.2f}")
    print(f"  findings naming no feature at all                : "
          f"{summary['findings_naming_no_feature']}/{len(gnd)}")
    print(f"  direction claims actually tested                 : "
          f"{summary['direction_claims_tested']}")
    if summary["mean_features_named_per_finding"] == 0:
        print("\n  WARNING: no finding named any feature, so the grounding rates "
              "above are vacuous and must not be reported as a result.")

    print("\n[faithfulness] wrote ablation, grounding, per detector and summary files")


if __name__ == "__main__":
    main()
