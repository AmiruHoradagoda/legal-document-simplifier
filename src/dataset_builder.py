"""Dataset construction utilities for simplification and classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Iterable


SIMPLIFICATION_COLUMNS = [
    "clause_id",
    "clause_text",
    "simple_clause",
    "needs_manual_simplification",
    "split",
]

CLASSIFICATION_COLUMNS = [
    "clause_id",
    "clause_text",
    "clause_type",
    "risk_level",
    "risk_type",
    "weak_label_reason",
    "split",
]


CLAUSE_TYPE_RULES = [
    ("payment", ["payment", "pay", "rent", "fee", "invoice", "charge", "late fee", "deposit"]),
    ("termination", ["terminate", "termination", "cancel", "end the agreement", "breach"]),
    ("liability", ["liable", "liability", "damages", "indemnify", "indemnification", "hold harmless"]),
    ("confidentiality", ["confidential", "non-disclosure", "nondisclosure", "trade secret"]),
    ("privacy", ["personal data", "privacy", "data protection", "process data"]),
    ("intellectual_property", ["intellectual property", "copyright", "trademark", "license", "ownership"]),
    ("dispute_resolution", ["arbitration", "jurisdiction", "venue", "governing law", "court"]),
    ("renewal", ["renew", "renewal", "extension", "auto-renew"]),
]

RISK_TYPE_RULES = [
    ("financial_penalty", ["penalty", "late fee", "interest", "charge", "forfeit", "liquidated damages"]),
    ("termination", ["terminate", "termination", "cancel", "breach", "default"]),
    ("liability_limitation", ["not be liable", "limit liability", "indirect damages", "consequential damages"]),
    ("indemnification", ["indemnify", "indemnification", "hold harmless", "defend"]),
    ("unilateral_change", ["sole discretion", "without notice", "modify", "change these terms"]),
    ("data_privacy", ["personal data", "privacy", "data protection", "share data"]),
    ("dispute_resolution", ["arbitration", "waive jury", "jurisdiction", "venue", "governing law"]),
    ("confidentiality", ["confidential", "non-disclosure", "trade secret"]),
]

HIGH_RISK_KEYWORDS = [
    "indemnify",
    "hold harmless",
    "not be liable",
    "sole discretion",
    "without notice",
    "waive",
    "arbitration",
    "consequential damages",
    "punitive damages",
    "personal data",
    "terminate immediately",
]

MEDIUM_RISK_KEYWORDS = [
    "terminate",
    "breach",
    "default",
    "late fee",
    "penalty",
    "interest",
    "confidential",
    "governing law",
    "jurisdiction",
    "renewal",
    "charge",
]

PUBLIC_DATASET_NAME = "coastalcph/lex_glue"
PUBLIC_DATASET_CONFIG = "ledgar"

LEGAL_SIMPLIFICATION_REPLACEMENTS = [
    (r"\bshall\b", "must"),
    (r"\bhereby\b", ""),
    (r"\bthereof\b", "of it"),
    (r"\bherein\b", "in this agreement"),
    (r"\bhereto\b", "to this agreement"),
    (r"\bprovided that\b", "if"),
    (r"\bprior to\b", "before"),
    (r"\bpursuant to\b", "under"),
    (r"\bin accordance with\b", "under"),
    (r"\bnotwithstanding\b", "despite"),
    (r"\bcommence\b", "start"),
    (r"\bterminate\b", "end"),
    (r"\btermination\b", "ending"),
    (r"\bremit\b", "pay"),
    (r"\blessee\b", "renter"),
    (r"\blessor\b", "owner"),
]


@dataclass(frozen=True)
class WeakClassificationLabel:
    """Weak labels assigned from transparent keyword rules."""

    clause_type: str
    risk_level: str
    risk_type: str
    weak_label_reason: str


def get_simplification_columns() -> list[str]:
    """Return canonical simplification dataset columns."""

    return list(SIMPLIFICATION_COLUMNS)


def get_classification_columns() -> list[str]:
    """Return canonical classification dataset columns."""

    return list(CLASSIFICATION_COLUMNS)


def label_clause_with_rules(clause_text: object) -> dict[str, str]:
    """Assign weak clause labels using deterministic keyword rules."""

    text = _normalize_text(clause_text)
    clause_type, type_hits = _first_matching_label(text, CLAUSE_TYPE_RULES, "general")
    risk_type, risk_hits = _first_matching_label(text, RISK_TYPE_RULES, "general")

    high_hits = _matching_keywords(text, HIGH_RISK_KEYWORDS)
    medium_hits = _matching_keywords(text, MEDIUM_RISK_KEYWORDS)

    if high_hits:
        risk_level = "high"
        level_hits = high_hits
    elif medium_hits:
        risk_level = "medium"
        level_hits = medium_hits
    else:
        risk_level = "low"
        level_hits = []

    reason_parts = []
    if type_hits:
        reason_parts.append(f"type={clause_type}: {', '.join(type_hits)}")
    if risk_hits:
        reason_parts.append(f"risk_type={risk_type}: {', '.join(risk_hits)}")
    if level_hits:
        reason_parts.append(f"risk_level={risk_level}: {', '.join(level_hits)}")
    if not reason_parts:
        reason_parts.append("default low risk general clause")

    return asdict(
        WeakClassificationLabel(
            clause_type=clause_type,
            risk_level=risk_level,
            risk_type=risk_type,
            weak_label_reason="; ".join(reason_parts),
        )
    )


def build_simplification_dataset(
    clauses: Iterable[dict[str, object]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create manual-fill simplification rows from clause records."""

    rows: list[dict[str, object]] = []
    for clause in clauses:
        clause_text = str(clause.get("clause_text", "") or "").strip()
        if not clause_text:
            continue
        rows.append(
            {
                "clause_id": str(clause.get("clause_id", "") or _fallback_id(clause_text)),
                "clause_text": clause_text,
                "simple_clause": "",
                "needs_manual_simplification": True,
                "split": "",
            }
        )

    splits = assign_splits(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for row, split in zip(rows, splits):
        row["split"] = split
    return rows


def weak_simplify_legal_text(text: object) -> str:
    """Create a weak plain-language target for demo training.

    This is not a gold simplification. It gives the model a runnable public-data
    training target when no human-written simplifications are available.
    """

    simplified = re.sub(r"\s+", " ", str(text or "")).strip()
    for pattern, replacement in LEGAL_SIMPLIFICATION_REPLACEMENTS:
        simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    return simplified


def build_weak_simplification_dataset(
    clauses: Iterable[dict[str, object]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create simplification rows with weak auto-generated targets."""

    rows = build_simplification_dataset(
        clauses,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for row in rows:
        row["simple_clause"] = weak_simplify_legal_text(row["clause_text"])
        row["needs_manual_simplification"] = False
    return rows


def build_classification_dataset(
    clauses: Iterable[dict[str, object]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create weakly labeled classification rows from clause records."""

    rows: list[dict[str, object]] = []
    for clause in clauses:
        clause_text = str(clause.get("clause_text", "") or "").strip()
        if not clause_text:
            continue

        labels = label_clause_with_rules(clause_text)
        rows.append(
            {
                "clause_id": str(clause.get("clause_id", "") or _fallback_id(clause_text)),
                "clause_text": clause_text,
                "clause_type": labels["clause_type"],
                "risk_level": labels["risk_level"],
                "risk_type": labels["risk_type"],
                "weak_label_reason": labels["weak_label_reason"],
                "split": "",
            }
        )

    splits = assign_splits(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_key="risk_level",
    )
    for row, split in zip(rows, splits):
        row["split"] = split
    return rows


def build_public_ledgar_datasets(
    *,
    max_train_rows: int = 600,
    max_validation_rows: int = 100,
    max_test_rows: int = 100,
    max_clause_types: int = 10,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Download LexGLUE LEDGAR and build simplification/classification rows.

    Returns `(simplification_rows, classification_rows)`.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Public dataset loading requires the `datasets` package.") from exc

    dataset = load_dataset(PUBLIC_DATASET_NAME, PUBLIC_DATASET_CONFIG)
    split_limits = {
        "train": max_train_rows,
        "validation": max_validation_rows,
        "test": max_test_rows,
    }

    classification_candidates: list[dict[str, object]] = []

    for split_name, limit in split_limits.items():
        if split_name not in dataset or limit <= 0:
            continue
        split_dataset = dataset[split_name]
        indexes = _stable_order(range(len(split_dataset)), seed=seed)[:limit]
        for position, dataset_index in enumerate(indexes, start=1):
            example = split_dataset[int(dataset_index)]
            clause_text = _extract_public_text(example)
            if not clause_text:
                continue
            clause_type = _extract_public_label(split_dataset, example)
            clause_id = f"ledgar_{split_name}_{int(dataset_index):06d}"
            labels = label_clause_with_rules(clause_text)

            classification_candidates.append(
                {
                    "clause_id": clause_id,
                    "clause_text": clause_text,
                    "clause_type": clause_type,
                    "risk_level": labels["risk_level"],
                    "risk_type": labels["risk_type"],
                    "weak_label_reason": f"LEDGAR label for clause_type; {labels['weak_label_reason']}",
                    "split": split_name,
                }
            )

    train_label_counts = Counter(
        row["clause_type"] for row in classification_candidates if row["split"] == "train"
    )
    if max_clause_types > 0:
        kept_labels = {label for label, _ in train_label_counts.most_common(max_clause_types)}
        classification_rows = [
            row for row in classification_candidates if row["clause_type"] in kept_labels
        ]
    else:
        classification_rows = classification_candidates

    source_clause_rows = [
        {
            "clause_id": row["clause_id"],
            "clause_text": row["clause_text"],
        }
        for row in classification_rows
    ]

    simplification_rows = build_weak_simplification_dataset(
        source_clause_rows,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
    )
    split_by_clause_id = {row["clause_id"]: row["split"] for row in classification_rows}
    for row in simplification_rows:
        row["split"] = split_by_clause_id.get(row["clause_id"], row["split"])

    return simplification_rows, classification_rows


def assign_splits(
    rows: list[dict[str, object]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: str | None = None,
) -> list[str]:
    """Assign deterministic train, validation, and test split labels."""

    if not rows:
        return []
    _validate_ratios(train_ratio, validation_ratio, test_ratio)

    if stratify_key and _can_stratify(rows, stratify_key):
        return _assign_stratified_splits(
            rows,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
            stratify_key=stratify_key,
        )

    ordered_indexes = _stable_order(range(len(rows)), seed=seed)
    return _labels_for_ordered_indexes(
        ordered_indexes,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )


def find_missing_values(rows: Iterable[dict[str, object]], columns: Iterable[str]) -> dict[str, int]:
    """Count blank values for required columns."""

    missing: dict[str, int] = {}
    for column in columns:
        count = 0
        for row in rows:
            value = row.get(column)
            if value is None or str(value).strip() == "":
                count += 1
        missing[column] = count
    return missing


def class_distribution(rows: Iterable[dict[str, object]], column: str) -> dict[str, int]:
    """Return class counts for a label column."""

    counts = Counter(str(row.get(column, "") or "").strip() for row in rows)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def split_distribution(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    """Return split counts."""

    return class_distribution(rows, "split")


def _first_matching_label(
    text: str,
    rules: list[tuple[str, list[str]]],
    default: str,
) -> tuple[str, list[str]]:
    for label, keywords in rules:
        hits = _matching_keywords(text, keywords)
        if hits:
            return label, hits
    return default, []


def _matching_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    hits = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text):
            hits.append(keyword)
    return hits


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _fallback_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"clause_{digest}"


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def _can_stratify(rows: list[dict[str, object]], stratify_key: str) -> bool:
    counts = Counter(str(row.get(stratify_key, "") or "") for row in rows)
    return len(rows) >= 9 and len(counts) > 1 and all(count >= 3 for count in counts.values())


def _assign_stratified_splits(
    rows: list[dict[str, object]],
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    stratify_key: str,
) -> list[str]:
    labels = ["train"] * len(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get(stratify_key, "") or "")].append(index)

    for indexes in groups.values():
        ordered = _stable_order(indexes, seed=seed)
        split_counts = _split_counts(
            len(ordered),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        position_to_split = (
            ["train"] * split_counts["train"]
            + ["validation"] * split_counts["validation"]
            + ["test"] * split_counts["test"]
        )
        for position, original_index in enumerate(ordered):
            labels[original_index] = position_to_split[position]
    return labels


def _labels_for_ordered_indexes(
    ordered_indexes: list[int],
    *,
    train_ratio: float,
    validation_ratio: float,
) -> list[str]:
    labels = ["train"] * len(ordered_indexes)
    split_counts = _split_counts(len(ordered_indexes), train_ratio, validation_ratio)

    position_to_split: list[str] = (
        ["train"] * split_counts["train"]
        + ["validation"] * split_counts["validation"]
        + ["test"] * split_counts["test"]
    )
    for position, original_index in enumerate(ordered_indexes):
        labels[original_index] = position_to_split[position]
    return labels


def _split_counts(total: int, train_ratio: float, validation_ratio: float) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if total == 1:
        return {"train": 1, "validation": 0, "test": 0}
    if total == 2:
        return {"train": 1, "validation": 0, "test": 1}
    if total == 3:
        return {"train": 1, "validation": 1, "test": 1}

    train_count = max(1, round(total * train_ratio))
    validation_count = max(1, round(total * validation_ratio))
    test_count = total - train_count - validation_count

    if test_count < 1:
        test_count = 1
        train_count = max(1, train_count - 1)
    while train_count + validation_count + test_count > total:
        train_count = max(1, train_count - 1)

    return {"train": train_count, "validation": validation_count, "test": test_count}


def _stable_order(indexes: Iterable[int], *, seed: int) -> list[int]:
    return sorted(
        indexes,
        key=lambda index: hashlib.sha1(f"{seed}:{index}".encode("utf-8")).hexdigest(),
    )


def _extract_public_text(example: dict[str, object]) -> str:
    for key in ["text", "clause_text", "provision", "context"]:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip()
    return ""


def _extract_public_label(split_dataset, example: dict[str, object]) -> str:
    label_value = example.get("label")
    features = getattr(split_dataset, "features", {})
    label_feature = features.get("label") if hasattr(features, "get") else None
    if label_feature is not None and hasattr(label_feature, "int2str"):
        try:
            return str(label_feature.int2str(int(label_value))).strip()
        except Exception:
            pass

    if isinstance(label_value, list):
        return "_".join(str(item).strip() for item in label_value if str(item).strip()) or "unknown"
    return str(label_value).strip() or "unknown"
