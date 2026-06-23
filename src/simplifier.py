"""Clause simplification model utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


INPUT_PREFIX = "simplify legal text: "

SIMPLIFIER_PREDICTION_COLUMNS = [
    "clause_id",
    "split",
    "clause_text",
    "reference_simple_clause",
    "predicted_simple_clause",
]


def build_simplification_prompt(text: object, *, prefix: str = INPUT_PREFIX) -> str:
    """Build the text-to-text prompt used for simplification training."""

    return f"{prefix}{str(text or '').strip()}"


def validate_simplification_rows(rows: Iterable[dict[str, object]]) -> list[str]:
    """Return validation issues for simplification dataset rows."""

    issues: list[str] = []
    rows = list(rows)
    if not rows:
        return ["No rows found in the simplification dataset."]

    required_columns = {"clause_id", "clause_text", "simple_clause", "split"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        issues.append(f"Missing required columns: {sorted(missing_columns)}")

    missing_targets = sum(1 for row in rows if not str(row.get("simple_clause", "") or "").strip())
    if missing_targets:
        issues.append(f"{missing_targets} row(s) have blank simple_clause targets.")

    return issues


def normalize_split(value: object) -> str:
    """Normalize split labels to train, validation, or test."""

    split = str(value or "").strip().lower()
    aliases = {"valid": "validation", "val": "validation", "dev": "validation"}
    return aliases.get(split, split)


def generate_simplifications(
    texts: Iterable[str],
    *,
    model,
    tokenizer,
    prefix: str = INPUT_PREFIX,
    max_new_tokens: int = 96,
    num_beams: int = 4,
) -> list[str]:
    """Generate simplified clauses with a trained seq2seq model."""

    prompts = [build_simplification_prompt(text, prefix=prefix) for text in texts]
    if not prompts:
        return []

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
