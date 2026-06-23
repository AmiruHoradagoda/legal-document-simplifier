from src.risk_rules import apply_risk_rules, apply_risk_rules_to_rows


def test_apply_risk_rules_detects_high_liability_clause():
    result = apply_risk_rules("The landlord shall not be liable for consequential damages.")

    assert result["rule_risk_level"] == "high"
    assert result["rule_risk_type"] == "liability_limitation"


def test_apply_risk_rules_defaults_to_low_general():
    result = apply_risk_rules("The tenant shall keep the property clean.")

    assert result["rule_risk_level"] == "low"
    assert result["rule_risk_type"] == "general"


def test_apply_risk_rules_to_rows_preserves_original_fields():
    rows = apply_risk_rules_to_rows([{"clause_id": "c1", "clause_text": "A late fee may be charged."}])

    assert rows[0]["clause_id"] == "c1"
    assert rows[0]["rule_risk_level"] == "medium"
