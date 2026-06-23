"""Clause classification model utilities."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable


PRIMARY_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
FALLBACK_MODEL_NAME = "distilbert-base-uncased"

CLASSIFIER_PREDICTION_COLUMNS = [
    "clause_id",
    "split",
    "clause_text",
    "true_clause_type",
    "predicted_clause_type",
    "confidence",
]


def normalize_split(value: object) -> str:
    """Normalize split labels to train, validation, or test."""

    split = str(value or "").strip().lower()
    aliases = {"valid": "validation", "val": "validation", "dev": "validation"}
    return aliases.get(split, split)


def build_label_mappings(labels: Iterable[object]) -> tuple[dict[str, int], dict[int, str]]:
    """Build stable label mappings for classification."""

    unique_labels = sorted({str(label).strip() for label in labels if str(label).strip()})
    if not unique_labels:
        raise ValueError("At least one non-empty label is required.")

    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def encode_labels(labels: Iterable[object], label2id: dict[str, int]) -> list[int]:
    """Encode string labels with a label2id mapping."""

    encoded: list[int] = []
    for label in labels:
        key = str(label).strip()
        if key not in label2id:
            raise KeyError(f"Unknown label: {key}")
        encoded.append(label2id[key])
    return encoded


def save_label_mapping(path: str | Path, label2id: dict[str, int], id2label: dict[int, str]) -> Path:
    """Save classifier label mappings to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label2id": label2id,
        "id2label": {str(index): label for index, label in id2label.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_label_mapping(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load classifier label mappings from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    label2id = {str(label): int(index) for label, index in payload["label2id"].items()}
    id2label = {int(index): str(label) for index, label in payload["id2label"].items()}
    return label2id, id2label


def load_tokenizer_and_model_with_fallback(
    *,
    primary_model_name: str = PRIMARY_MODEL_NAME,
    fallback_model_name: str = FALLBACK_MODEL_NAME,
    num_labels: int,
    label2id: dict[str, int],
    id2label: dict[int, str],
):
    """Load LegalBERT and fall back to DistilBERT if loading fails."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    errors: list[str] = []
    for model_name in [primary_model_name, fallback_model_name]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                label2id=label2id,
                id2label={index: label for index, label in id2label.items()},
            )
            return model_name, tokenizer, model
        except Exception as exc:  # pragma: no cover - depends on network/model cache
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError("Could not load primary or fallback classifier model. " + " | ".join(errors))


def compute_classification_metrics(eval_pred) -> dict[str, float]:
    """Compute accuracy, weighted precision, weighted recall, and weighted F1."""

    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_confusion_matrix_frame(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
    label_names: list[str],
):
    """Return a pandas confusion matrix DataFrame."""

    import pandas as pd
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(list(true_labels), list(predicted_labels), labels=label_names)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_names],
        columns=[f"pred_{label}" for label in label_names],
    )


def softmax_confidence(logits) -> list[float]:
    """Return max softmax confidence for each logits row."""

    import numpy as np

    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1).round(6).tolist()


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
