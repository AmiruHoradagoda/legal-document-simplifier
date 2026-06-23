from src.dataset_builder import (
    build_public_ledgar_datasets,
    class_distribution,
    find_missing_values,
    get_classification_columns,
    get_simplification_columns,
    label_clause_with_rules,
    split_distribution,
    weak_simplify_legal_text,
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


def test_build_public_ledgar_datasets_maps_hugging_face_rows(monkeypatch):
    class FakeLabelFeature:
        def int2str(self, value):
            return {0: "Payment", 1: "Termination"}[value]

    class FakeSplit:
        features = {"label": FakeLabelFeature()}

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    class FakeDataset(dict):
        pass

    fake_dataset = FakeDataset(
        {
            "train": FakeSplit(
                [
                    {"text": "The tenant shall pay rent on time.", "label": 0},
                    {"text": "Either party may terminate after breach.", "label": 1},
                    {"text": "The landlord may charge a late fee.", "label": 0},
                ]
            ),
            "validation": FakeSplit(
                [
                    {"text": "Payment must be made before the due date.", "label": 0},
                    {"text": "Termination requires written notice.", "label": 1},
                ]
            ),
            "test": FakeSplit(
                [
                    {"text": "Rent payment is due monthly.", "label": 0},
                    {"text": "The agreement may terminate for default.", "label": 1},
                ]
            ),
        }
    )

    def fake_load_dataset(dataset_name, config_name):
        assert dataset_name == "coastalcph/lex_glue"
        assert config_name == "ledgar"
        return fake_dataset

    import sys
    import types

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    simplification_rows, classification_rows = build_public_ledgar_datasets(
        max_train_rows=3,
        max_validation_rows=2,
        max_test_rows=2,
        max_clause_types=2,
        seed=42,
    )

    assert simplification_rows
    assert classification_rows
    assert set(simplification_rows[0]) == set(get_simplification_columns())
    assert set(classification_rows[0]) == set(get_classification_columns())
    assert all(row["clause_id"].startswith("ledgar_") for row in classification_rows)
    assert find_missing_values(classification_rows, ["clause_type", "risk_level", "risk_type", "split"]) == {
        "clause_type": 0,
        "risk_level": 0,
        "risk_type": 0,
        "split": 0,
    }
    assert find_missing_values(simplification_rows, ["clause_id", "clause_text", "simple_clause", "split"]) == {
        "clause_id": 0,
        "clause_text": 0,
        "simple_clause": 0,
        "split": 0,
    }
    assert set(split_distribution(classification_rows)) == {"train", "validation", "test"}
    assert set(class_distribution(classification_rows, "clause_type")) == {"Payment", "Termination"}


def test_weak_simplify_legal_text_replaces_common_legalese():
    simplified = weak_simplify_legal_text(
        "The lessee shall remit payment pursuant to this agreement prior to termination."
    )

    assert "renter" in simplified.lower()
    assert "must" in simplified.lower()
    assert "under" in simplified.lower()
    assert "before" in simplified.lower()
