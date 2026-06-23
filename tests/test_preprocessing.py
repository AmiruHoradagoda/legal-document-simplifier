from src.preprocessing import clean_legal_text, count_words, flatten_text


def test_clean_legal_text_preserves_inline_clause_numbers():
    text = "1. Payment. The tenant shall pay rent.\n\nPage 1 of 3\n\n2. Term. This agreement lasts one year."

    cleaned = clean_legal_text(text)

    assert "1. Payment" in cleaned
    assert "2. Term" in cleaned
    assert "Page 1 of 3" not in cleaned


def test_flatten_text_collapses_whitespace():
    assert flatten_text("A   clause\nwith\tspacing.") == "A clause with spacing."


def test_count_words_handles_basic_legal_text():
    assert count_words("Tenant's thirty-day notice period.") == 4
