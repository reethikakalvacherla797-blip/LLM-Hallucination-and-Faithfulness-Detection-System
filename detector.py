from functools import lru_cache

import torch
from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def load_nli_model():
    """Load the pretrained NLI model once per app session/process."""
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        device=device
    )

    # Print the model's real label map for debugging / verification.
    print("Loaded NLI model label map:", classifier.model.config.id2label)
    return classifier


def _normalize_label(label):
    """Normalize common model label spellings."""
    label = label.strip().lower().replace(" ", "_")
    if label in {"entailment", "contradiction", "neutral"}:
        return label
    return label


def detect_hallucination(context, answer):
    """
    Use NLI to estimate whether answer is entailed by, contradicts,
    or is neutral with respect to the supplied context.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context cannot be empty.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    classifier = load_nli_model()

    # In NLI, the context is the premise and the answer is the hypothesis.
    raw = classifier(
        {"text": context, "text_pair": answer},
        top_k=None,
        truncation=True
    )

    # Transformers versions may return either a list of dictionaries
    # or a nested list for a single input.
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    raw_scores = {
        _normalize_label(item["label"]): float(item["score"])
        for item in raw
    }

    # Some checkpoints expose generic LABEL_0 / LABEL_1 / LABEL_2 labels.
    # This mapping is the commonly used mapping for this checkpoint;
    # if the output labels differ, inspect the printed label map above.
    if all(label.startswith("label_") for label in raw_scores):
        # Generic labels do not reveal their meaning. Use the known label order
        # for this checkpoint and verify the checkpoint config if it differs.
        fallback = {
            "label_0": "contradiction",
            "label_1": "entailment",
            "label_2": "neutral"
        }
        raw_scores = {
            fallback[label]: score
            for label, score in raw_scores.items()
            if label in fallback
        }

    # Keep only known NLI labels; fail clearly rather than silently guessing.
    required = {"entailment", "contradiction", "neutral"}
    if not required.issubset(raw_scores):
        raise ValueError(
            "Could not identify all NLI labels from this model. "
            f"Received labels: {list(raw_scores.keys())}. "
            "Check the model's printed label map."
        )

    top_label = max(required, key=lambda label: raw_scores[label])

    status_map = {
        "entailment": "SUPPORTED",
        "contradiction": "CONTRADICTED",
        "neutral": "INSUFFICIENT EVIDENCE"
    }

    return {
        "status": status_map[top_label],
        "model_label": top_label,
        "entailment_score": raw_scores["entailment"],
        "contradiction_score": raw_scores["contradiction"],
        "neutral_score": raw_scores["neutral"]
    }
