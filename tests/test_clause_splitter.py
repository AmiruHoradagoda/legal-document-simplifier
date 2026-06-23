from src.clause_splitter import build_clause_records, split_clauses


def test_split_clauses_preserves_number_labels():
    text = (
        "1. Payment. The tenant shall pay rent on the first day of each month. "
        "2. Late Fees. The landlord may charge a reasonable late fee after notice."
    )

    clauses = split_clauses(text, min_words=3, min_chars=10)

    assert [clause["clause_label"] for clause in clauses] == ["1", "2"]
    assert clauses[0]["clause_text"].startswith("Payment.")


def test_build_clause_records_adds_sequential_numbers():
    records = build_clause_records(
        "doc-1",
        "sample.txt",
        "The tenant shall pay rent on the first day of each month. "
        "If payment is late, the landlord may charge a reasonable fee.",
        min_words=5,
        min_chars=20,
    )

    assert [record["clause_number"] for record in records] == [1, 2]
    assert all(record["clause_id"].startswith("doc_1_clause_") for record in records)


def test_split_clauses_filters_very_short_segments():
    clauses = split_clauses(
        "Definitions. 1. Payment. The tenant shall pay rent on the first day of each month.",
        min_words=5,
        min_chars=20,
    )

    assert len(clauses) == 1
    assert clauses[0]["clause_label"] == "1"
