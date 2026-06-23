from src.rag_qa import LEGAL_DISCLAIMER, answer_question, generate_rule_based_answer, retrieve_top_k


SAMPLE_CLAUSES = [
    {
        "clause_id": "c1",
        "document_id": "d1",
        "clause_number": 1,
        "clause_text": "The tenant shall pay rent on the first day of each month.",
    },
    {
        "clause_id": "c2",
        "document_id": "d1",
        "clause_number": 2,
        "clause_text": "If payment is more than five days late, the landlord may charge a reasonable late fee.",
    },
    {
        "clause_id": "c3",
        "document_id": "d1",
        "clause_number": 3,
        "clause_text": "Either party may terminate this agreement after a material breach and written notice.",
    },
]


def test_retrieve_top_k_uses_lexical_fallback_without_model():
    results = retrieve_top_k("What is the late payment fee?", SAMPLE_CLAUSES, top_k=2)

    assert len(results) == 2
    assert results[0]["clause_id"] == "c2"
    assert results[0]["score"] >= results[1]["score"]


def test_generate_rule_based_answer_uses_retrieved_clause_only():
    answer = generate_rule_based_answer("Can the landlord charge a fee?", [SAMPLE_CLAUSES[1] | {"score": 0.9}])

    assert LEGAL_DISCLAIMER in answer
    assert "late fee" in answer
    assert "Clause 2" in answer


def test_answer_question_returns_retrieved_clauses():
    result = answer_question("Can either party terminate?", SAMPLE_CLAUSES, top_k=1)

    assert result["question"] == "Can either party terminate?"
    assert len(result["retrieved_clauses"]) == 1
    assert result["retrieved_clauses"][0]["clause_id"] == "c3"


def test_answer_question_handles_empty_clause_list():
    result = answer_question("What happens?", [], top_k=3)

    assert result["retrieved_clauses"] == []
    assert "could not find" in result["answer"]
