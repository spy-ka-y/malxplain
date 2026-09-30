"""
MalXplain: LLM reporting layer.

This is the component that distinguishes MalXplain from its base paper. Madamidola
et al. (2024) stop at a SHAP plot, which an analyst must interpret themselves. Here
the SHAP evidence is handed to a locally-hosted LLM (via Ollama) that writes a short,
readable finding.

Design constraints, deliberately chosen:

  * The LLM NEVER makes the detection decision. It receives an already-computed
    prediction, confidence and ranked SHAP evidence, and only narrates them. This
    keeps the detection path statistical and auditable.
  * The model runs locally. No sample, feature vector or finding leaves the machine,
    which is what makes the approach usable in an air-gapped or regulated environment.
  * If Ollama is unreachable, a deterministic template fallback produces the finding
    instead, so the prototype degrades gracefully rather than failing in a live demo.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
TIMEOUT_S = 120

SYSTEM_PROMPT = """You are a malware triage assistant writing a short finding for a \
security analyst.

You are given a classification that has ALREADY been made by a Random Forest model, \
together with SHAP values showing which memory-forensic features drove that decision.

Rules you must follow, without exception:
- State ONLY what the evidence below contains. Every claim in your finding must be \
traceable to a number or label you were given.
- NEVER say a feature is "associated with", "characteristic of", "typical of" or \
"linked to" any malware subtype or family. You have no evidence for such claims. The \
detector's training subtype tells you what the model learned from, NOT what any \
feature indicates.
- Discuss the top three contributors in the order given. Do not skip a stronger \
contributor to mention a weaker one.
- Do not invent feature values, malware names, file names, IP addresses, attack \
techniques, or behaviour the evidence does not describe.
- Do not claim certainty beyond the confidence score, and do not speculate about what \
the sample "is" or "does".
- Refer to features by their plain-English description, not the raw column name.
- Write 3 to 4 sentences of flowing prose. No bullet points, no headings, no preamble.
- Begin directly with the verdict."""


def _evidence_block(explanation: dict, top_n: int = 5) -> str:
    lines = []
    for e in explanation["evidence"][:top_n]:
        direction = "towards MALWARE" if e["shap_value"] > 0 else "towards BENIGN"
        lines.append(
            f"- {e['description']} (scaled value {e['scaled_value']}): "
            f"SHAP {e['shap_value']:+.4f}, pushing the decision {direction}"
        )
    return "\n".join(lines)


def build_prompt(explanation: dict) -> str:
    """Assemble the user-side prompt from the structured SHAP payload."""
    return f"""Classification result for sample {explanation['sample_id']}:

Verdict: {explanation['prediction']}
Confidence: {explanation['confidence']:.1%}
Probability the sample is malware: {explanation['malware_probability']:.1%}
Detector: a Random Forest trained ONLY on the {explanation['detector_trained_on']} \
malware subtype, using five memory-forensic features.

Evidence ranked by contribution:
{_evidence_block(explanation)}

Write the analyst finding."""


def _template_finding(explanation: dict) -> str:
    """Deterministic fallback used when the local LLM is unavailable."""
    top = explanation["evidence"][0]
    second = explanation["evidence"][1]
    verdict = explanation["prediction"].lower()
    conf = explanation["confidence"]

    hedge = "strongly indicates" if conf >= 0.9 else "suggests" if conf >= 0.7 else "weakly suggests"
    detector = explanation["detector_trained_on"]

    if explanation["prediction"] == "Malware":
        context = (
            f"The decision came from a detector trained only on the {detector} subtype, "
            f"so this verdict reflects generalisation to behaviour the model was never "
            f"explicitly trained on. Analyst review is recommended before any containment "
            f"action is taken."
        )
    else:
        context = (
            f"The decision came from a detector trained only on the {detector} subtype. "
            f"A benign verdict from a narrowly-trained model is weaker evidence than a "
            f"malicious one, so corroboration from a second detector is advisable before "
            f"the sample is cleared."
        )

    return (
        f"The sample was classified as {verdict} with {conf:.1%} confidence. "
        f"The evidence {hedge} this verdict, driven primarily by the {top['description']} "
        f"(contribution {top['shap_value']:+.4f}), followed by the {second['description']} "
        f"(contribution {second['shap_value']:+.4f}). "
        f"{context}"
    )


def ollama_available(host: str = DEFAULT_HOST, timeout: int = 5) -> bool:
    """Check whether a local Ollama server is reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def list_models(host: str = DEFAULT_HOST) -> list[str]:
    """Return the model names available on the local Ollama server."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def generate_finding(
    explanation: dict,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.2,
) -> dict:
    """
    Turn a structured SHAP explanation into a human-readable finding.

    Returns {"finding": str, "source": "llm"|"template", "model": str, "error": str|None}
    """
    payload = {
        "model": model,
        "prompt": build_prompt(explanation),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 300},
    }

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            body = json.loads(r.read())
        text = body.get("response", "").strip()
        if not text:
            raise ValueError("empty response from model")
        return {"finding": text, "source": "llm", "model": model, "error": None}

    except Exception as exc:  # unreachable server, timeout, bad model name, etc.
        return {
            "finding": _template_finding(explanation),
            "source": "template",
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    """Smoke test against the example explanations produced by explainability.py."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "explanation_examples.json"
    if not path.exists():
        raise SystemExit("Run `python -m src.explainability` first to generate examples.")

    examples = json.loads(path.read_text())

    print(f"Ollama reachable : {ollama_available()}")
    print(f"Models available : {list_models() or '(none / server down)'}\n")

    for name, ex in examples.items():
        print("=" * 76)
        print(f"{name}  |  predicted {ex['prediction']} @ {ex['confidence']:.1%}  "
              f"|  truth: {ex['ground_truth']} ({ex['actual_subtype']})")
        print("=" * 76)
        result = generate_finding(ex)
        print(f"[source: {result['source']}]"
              + (f"  [error: {result['error']}]" if result["error"] else ""))
        print(result["finding"])
        print()


if __name__ == "__main__":
    main()
