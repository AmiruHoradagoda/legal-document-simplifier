from src.simplifier import (
    INPUT_PREFIX,
    build_simplification_prompt,
    normalize_split,
    validate_simplification_rows,
)


def test_build_simplification_prompt_uses_required_prefix():
    prompt = build_simplification_prompt("The tenant shall pay rent.")

    assert prompt == f"{INPUT_PREFIX}The tenant shall pay rent."


def test_normalize_split_aliases_validation():
    assert normalize_split("val") == "validation"
    assert normalize_split("DEV") == "validation"
    assert normalize_split("test") == "test"


def test_validate_simplification_rows_reports_blank_targets():
    issues = validate_simplification_rows(
        [{"clause_id": "c1", "clause_text": "Original", "simple_clause": "", "split": "train"}]
    )

    assert any("blank simple_clause" in issue for issue in issues)
