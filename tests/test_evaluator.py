import math

from src.evaluator import (
    build_human_eval_template,
    compression_ratio,
    compute_classifier_metrics,
    compute_rouge_scores,
    confusion_matrix_records,
    summarize_simplification_quality,
)


def test_compression_ratio_uses_output_over_source_length():
    assert compression_ratio("1234567890", "12345") == 0.5


def test_compute_classifier_metrics_accuracy():
    metrics = compute_classifier_metrics(["a", "b", "b"], ["a", "a", "b"])

    assert round(metrics["accuracy"], 3) == 0.667


def test_confusion_matrix_records_contains_all_labels():
    labels, matrix = confusion_matrix_records(["a", "b"], ["b", "b"])

    assert labels == ["a", "b"]
    assert matrix == [[0, 1], [0, 1]]


def test_compute_rouge_scores_returns_values_or_nan():
    scores, _ = compute_rouge_scores(["plain text"], ["plain text"])

    assert "rouge1" in scores
    assert math.isnan(scores["rouge1"]) or 0 <= scores["rouge1"] <= 1


def test_summarize_simplification_quality_handles_empty_rows():
    summary = summarize_simplification_quality([])

    assert math.isnan(summary["compression_ratio"])


def test_build_human_eval_template_mixes_tasks():
    rows = build_human_eval_template(
        [
            {
                "clause_id": "s1",
                "split": "test",
                "clause_text": "Original",
                "reference_simple_clause": "Reference",
                "predicted_simple_clause": "Prediction",
            }
        ],
        [
            {
                "clause_id": "c1",
                "split": "test",
                "clause_text": "Clause",
                "true_clause_type": "payment",
                "predicted_clause_type": "payment",
            }
        ],
        sample_size=20,
    )

    assert {row["task"] for row in rows} == {"classification", "simplification"}
