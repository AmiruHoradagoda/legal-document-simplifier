"""Rule-based legal risk helpers."""

from __future__ import annotations

import re
from typing import Iterable


RISK_RULES = [
    {
        "risk_type": "liability_limitation",
        "risk_level": "high",
        "keywords": ["not be liable", "consequential damages", "indirect damages", "punitive damages"],
    },
    {
        "risk_type": "indemnification",
        "risk_level": "high",
        "keywords": ["indemnify", "indemnification", "hold harmless", "defend"],
    },
    {
        "risk_type": "unilateral_change",
        "risk_level": "high",
        "keywords": ["sole discretion", "without notice", "change these terms", "modify these terms"],
    },
    {
        "risk_type": "dispute_resolution",
        "risk_level": "high",
        "keywords": ["waive jury", "arbitration", "binding arbitration"],
    },
    {
        "risk_type": "financial_penalty",
        "risk_level": "medium",
        "keywords": ["late fee", "penalty", "interest", "charge", "forfeit"],
    },
    {
        "risk_type": "termination",
        "risk_level": "medium",
        "keywords": ["terminate", "termination", "material breach", "default"],
    },
    {
        "risk_type": "confidentiality",
        "risk_level": "medium",
        "keywords": ["confidential", "non-disclosure", "nondisclosure", "trade secret"],
    },
    {
        "risk_type": "data_privacy",
        "risk_level": "medium",
        "keywords": ["personal data", "privacy", "data protection", "share data"],
    },
]

RISK_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


def apply_risk_rules(clause_text: object) -> dict[str, object]:
    """Apply transparent keyword risk rules to one clause."""

    text = _normalize_text(clause_text)
    matches = []

    for rule in RISK_RULES:
        hits = _matching_keywords(text, rule["keywords"])
        if hits:
            matches.append(
                {
                    "risk_type": rule["risk_type"],
                    "risk_level": rule["risk_level"],
                    "matched_keywords": hits,
                }
            )

    if not matches:
        return {
            "rule_risk_level": "low",
            "rule_risk_type": "general",
            "rule_matches": "",
        }

    highest = max(matches, key=lambda item: RISK_LEVEL_RANK[item["risk_level"]])
    match_summary = "; ".join(
        f"{match['risk_type']}({match['risk_level']}): {', '.join(match['matched_keywords'])}"
        for match in matches
    )
    return {
        "rule_risk_level": highest["risk_level"],
        "rule_risk_type": highest["risk_type"],
        "rule_matches": match_summary,
    }


def apply_risk_rules_to_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Apply risk rules to rows that include `clause_text`."""

    output = []
    for row in rows:
        enriched = dict(row)
        enriched.update(apply_risk_rules(enriched.get("clause_text", "")))
        output.append(enriched)
    return output


def _matching_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    hits = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text):
            hits.append(keyword)
    return hits


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()
