from src.dataset_builder import (
    build_classification_dataset,
    build_simplification_dataset,
    class_distribution,
    find_missing_values,
    label_clause_with_rules,
    split_distribution,
)


def test_label_clause_with_rules_detects_financial_penalty():
    labels = label_clause_with_rules(
        "If payment is more than five days late, the landlord may charge a late fee."
    )

    assert labels["clause_type"] == "payment"
    assert labels["risk_level"] == "medium"
    assert labels["risk_type"] == "financial_penalty"


def test_label_clause_with_rules_detects_high_liability_risk():
    labels = label_clause_with_rules(
        "The provider shall not be liable for indirect or consequential damages."
    )

    assert labels["clause_type"] == "liability"
    assert labels["risk_level"] == "high"
    assert labels["risk_type"] == "liability_limitation"


def test_build_simplification_dataset_leaves_manual_field_blank():
    rows = build_simplification_dataset(
        [{"clause_id": "c1", "document_id": "d1", "clause_number": 1, "clause_text": "The tenant shall pay rent."}]
    )

    assert rows[0]["simple_clause"] == ""
    assert rows[0]["needs_manual_simplification"] is True


def test_build_classification_dataset_has_no_missing_labels():
    rows = build_classification_dataset(
        [
            {"clause_id": "c1", "clause_text": "The tenant shall pay rent."},
            {"clause_id": "c2", "clause_text": "The landlord may charge a late fee."},
            {"clause_id": "c3", "clause_text": "The provider shall not be liable for damages."},
        ]
    )

    missing = find_missing_values(rows, ["clause_type", "risk_level", "risk_type", "split"])

    assert missing == {"clause_type": 0, "risk_level": 0, "risk_type": 0, "split": 0}
    assert class_distribution(rows, "risk_level")["high"] == 1
    assert set(split_distribution(rows)) == {"train", "validation", "test"}
