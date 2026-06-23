"""Evaluation utilities for simplification and clause classification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
from typing import Iterable


RESULT_COLUMNS = ["stage", "metric", "value", "notes"]

HUMAN_EVAL_COLUMNS = [
    "evaluation_id",
    "task",
    "clause_id",
    "split",
    "original_clause",
    "reference_output",
    "model_output",
    "clarity_score_1_5",
    "faithfulness_score_1_5",
    "label_correct",
    "notes",
]


def get_result_columns() -> list[str]:
    """Return canonical evaluation result columns."""

    return list(RESULT_COLUMNS)


def get_human_eval_columns() -> list[str]:
    """Return canonical human evaluation template columns."""

    return list(HUMAN_EVAL_COLUMNS)


def word_count(text: object) -> int:
    """Count word-like tokens."""

    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def compression_ratio(source_text: object, output_text: object) -> float:
    """Return output length divided by source length."""

    source_length = max(len(str(source_text or "")), 1)
    return len(str(output_text or "")) / source_length


def readability_scores(text: object) -> dict[str, float]:
    """Compute readability metrics with textstat when available."""

    value = str(text or "").strip()
    if not value:
        return {"flesch_reading_ease": math.nan, "flesch_kincaid_grade": math.nan}

    try:
        import textstat

        return {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(value)),
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(value)),
        }
    except Exception:
        words = max(word_count(value), 1)
        sentences = max(len(re.findall(r"[.!?]+", value)), 1)
        avg_words_per_sentence = words / sentences
        return {
            "flesch_reading_ease": math.nan,
            "flesch_kincaid_grade": float(avg_words_per_sentence),
        }


def summarize_simplification_quality(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    """Summarize readability and compression before and after simplification."""

    rows = list(rows)
    if not rows:
        return {
            "source_flesch_reading_ease": math.nan,
            "prediction_flesch_reading_ease": math.nan,
            "source_flesch_kincaid_grade": math.nan,
            "prediction_flesch_kincaid_grade": math.nan,
            "compression_ratio": math.nan,
        }

    source_readability = [readability_scores(row.get("clause_text", "")) for row in rows]
    output_readability = [readability_scores(row.get("predicted_simple_clause", "")) for row in rows]
    compression_values = [
        compression_ratio(row.get("clause_text", ""), row.get("predicted_simple_clause", ""))
        for row in rows
    ]

    return {
        "source_flesch_reading_ease": _mean(score["flesch_reading_ease"] for score in source_readability),
        "prediction_flesch_reading_ease": _mean(score["flesch_reading_ease"] for score in output_readability),
        "source_flesch_kincaid_grade": _mean(score["flesch_kincaid_grade"] for score in source_readability),
        "prediction_flesch_kincaid_grade": _mean(score["flesch_kincaid_grade"] for score in output_readability),
        "compression_ratio": _mean(compression_values),
    }


def compute_rouge_scores(predictions: list[str], references: list[str]) -> tuple[dict[str, float], str]:
    """Compute ROUGE scores, falling back to token-overlap ROUGE-1 when needed."""

    predictions, references = _paired_non_empty(predictions, references)
    if not predictions:
        return {"rouge1": math.nan, "rouge2": math.nan, "rougeL": math.nan, "rougeLsum": math.nan}, "no paired predictions"

    try:
        import evaluate

        metric = evaluate.load("rouge")
        scores = metric.compute(predictions=predictions, references=references)
        return {key: float(value) for key, value in scores.items()}, "evaluate.load('rouge')"
    except Exception as exc:
        return {
            "rouge1": _mean(_token_f1(prediction, reference) for prediction, reference in zip(predictions, references)),
            "rouge2": math.nan,
            "rougeL": math.nan,
            "rougeLsum": math.nan,
        }, f"fallback token ROUGE-1 because evaluate rouge failed: {exc}"


def compute_bertscore(predictions: list[str], references: list[str]) -> tuple[dict[str, float], str]:
    """Compute BERTScore with evaluate when available."""

    predictions, references = _paired_non_empty(predictions, references)
    if not predictions:
        return {"bertscore_precision": math.nan, "bertscore_recall": math.nan, "bertscore_f1": math.nan}, "no paired predictions"

    try:
        import evaluate

        metric = evaluate.load("bertscore")
        scores = metric.compute(predictions=predictions, references=references, lang="en")
        return {
            "bertscore_precision": _mean(scores["precision"]),
            "bertscore_recall": _mean(scores["recall"]),
            "bertscore_f1": _mean(scores["f1"]),
        }, "evaluate.load('bertscore')"
    except Exception as exc:
        return {
            "bertscore_precision": math.nan,
            "bertscore_recall": math.nan,
            "bertscore_f1": math.nan,
        }, f"BERTScore unavailable: {exc}"


def compute_classifier_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, float]:
    """Compute accuracy, weighted precision, weighted recall, and weighted F1."""

    true_labels, predicted_labels = _paired_non_empty(true_labels, predicted_labels)
    if not true_labels:
        return {"accuracy": math.nan, "precision": math.nan, "recall": math.nan, "f1": math.nan}

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    except Exception:
        total = len(true_labels)
        correct = sum(true == predicted for true, predicted in zip(true_labels, predicted_labels))
        return {
            "accuracy": correct / total,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
        }


def confusion_matrix_records(true_labels: list[str], predicted_labels: list[str]) -> tuple[list[str], list[list[int]]]:
    """Return labels and confusion matrix values without requiring sklearn."""

    true_labels, predicted_labels = _paired_non_empty(true_labels, predicted_labels)
    labels = sorted(set(true_labels) | set(predicted_labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[label_to_index[true_label]][label_to_index[predicted_label]] += 1

    return labels, matrix


def build_human_eval_template(
    simplification_rows: list[dict[str, object]],
    classifier_rows: list[dict[str, object]],
    *,
    sample_size: int = 20,
) -> list[dict[str, object]]:
    """Create a mixed human evaluation template from prediction rows."""

    candidates: list[dict[str, object]] = []

    for row in simplification_rows:
        candidates.append(
            {
                "task": "simplification",
                "clause_id": row.get("clause_id", ""),
                "split": row.get("split", ""),
                "original_clause": row.get("clause_text", ""),
                "reference_output": row.get("reference_simple_clause", ""),
                "model_output": row.get("predicted_simple_clause", ""),
                "label_correct": "",
            }
        )

    for row in classifier_rows:
        candidates.append(
            {
                "task": "classification",
                "clause_id": row.get("clause_id", ""),
                "split": row.get("split", ""),
                "original_clause": row.get("clause_text", ""),
                "reference_output": row.get("true_clause_type", ""),
                "model_output": row.get("predicted_clause_type", ""),
                "label_correct": "",
            }
        )

    candidates = [candidate for candidate in candidates if str(candidate.get("original_clause", "")).strip()]
    ordered = sorted(candidates, key=lambda row: (str(row["task"]), str(row["clause_id"])))
    sampled = ordered[:sample_size]

    output_rows = []
    for index, row in enumerate(sampled, start=1):
        output_rows.append(
            {
                "evaluation_id": f"eval_{index:03d}",
                "task": row["task"],
                "clause_id": row["clause_id"],
                "split": row["split"],
                "original_clause": row["original_clause"],
                "reference_output": row["reference_output"],
                "model_output": row["model_output"],
                "clarity_score_1_5": "",
                "faithfulness_score_1_5": "",
                "label_correct": row["label_correct"],
                "notes": "",
            }
        )
    return output_rows


def write_csv(path: str | Path, rows: list[dict[str, object]], columns: list[str]) -> Path:
    """Write dictionaries to CSV with a fixed schema."""

    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _paired_non_empty(left_values: list[object], right_values: list[object]) -> tuple[list[str], list[str]]:
    pairs = [
        (str(left or "").strip(), str(right or "").strip())
        for left, right in zip(left_values, right_values)
        if str(left or "").strip() and str(right or "").strip()
    ]
    if not pairs:
        return [], []
    left, right = zip(*pairs)
    return list(left), list(right)


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = re.findall(r"\w+", prediction.lower())
    reference_tokens = re.findall(r"\w+", reference.lower())
    if not prediction_tokens or not reference_tokens:
        return 0.0

    pred_counts = Counter(prediction_tokens)
    ref_counts = Counter(reference_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _mean(values: Iterable[float]) -> float:
    clean_values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean_values:
        return math.nan
    return sum(clean_values) / len(clean_values)
