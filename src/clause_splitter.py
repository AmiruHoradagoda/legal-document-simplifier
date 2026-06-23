"""Clause splitting utilities for legal documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re

from src.preprocessing import clean_legal_text, count_words, flatten_text, is_short_text


CLAUSE_COLUMNS = [
    "clause_id",
    "document_id",
    "source_path",
    "clause_number",
    "clause_label",
    "original_clause_text",
    "clause_text",
    "word_count",
    "char_count",
]


@dataclass(frozen=True)
class ClauseRecord:
    """Normalized clause record written to data/processed/clauses.csv."""

    clause_id: str
    document_id: str
    source_path: str
    clause_number: int
    clause_label: str
    original_clause_text: str
    clause_text: str
    word_count: int
    char_count: int


_CLAUSE_MARKER_RE = re.compile(
    r"""
    (?P<marker>
        (?<!\w)(?:section|sec\.?|article|clause|paragraph)\s+
            [IVXLCDM]+(?:\.[A-Z0-9]+)*\.?
        |
        (?<!\w)(?:section|sec\.?|article|clause|paragraph)\s+
            \d+(?:\.\d+)*[A-Za-z]?\.?
        |
        (?<!\w)\d+(?:\.\d+)+\.?
        |
        (?<!\w)\d+\. (?=\s*[A-Z])
        |
        \([a-zA-Z0-9]+\)
        |
        (?m:^[a-zA-Z]\)\s+)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;:])\s+(?=(?:[A-Z]|\([a-zA-Z0-9]+\)|\d+\.))")


def get_clause_columns() -> list[str]:
    """Return the canonical column order for clause CSV files."""

    return list(CLAUSE_COLUMNS)


def split_clauses(text: object, *, min_words: int = 8, min_chars: int = 30) -> list[dict[str, str | int]]:
    """Split document text into clause dictionaries.

    The function first looks for explicit legal numbering. If no useful
    numbering is found, it falls back to sentence-style segmentation.
    """

    cleaned = clean_legal_text(text)
    if not cleaned:
        return []

    candidates = _split_by_markers(cleaned)
    if len(candidates) <= 1:
        candidates = _split_by_sentences(cleaned)

    results: list[dict[str, str | int]] = []
    for label, original in candidates:
        clause_text = flatten_text(_strip_leading_marker(original, label))
        original_text = flatten_text(original)
        if is_short_text(clause_text, min_words=min_words, min_chars=min_chars):
            continue

        results.append(
            {
                "clause_label": label,
                "original_clause_text": original_text,
                "clause_text": clause_text,
                "word_count": count_words(clause_text),
                "char_count": len(clause_text),
            }
        )

    return results


def build_clause_records(
    document_id: str,
    source_path: str,
    text: object,
    *,
    min_words: int = 8,
    min_chars: int = 30,
) -> list[dict[str, str | int]]:
    """Build CSV-ready clause records for a single document."""

    clauses = split_clauses(text, min_words=min_words, min_chars=min_chars)
    records: list[dict[str, str | int]] = []

    for index, clause in enumerate(clauses, start=1):
        clause_text = str(clause["clause_text"])
        records.append(
            asdict(
                ClauseRecord(
                    clause_id=_make_clause_id(document_id, index, clause_text),
                    document_id=str(document_id),
                    source_path=str(source_path),
                    clause_number=index,
                    clause_label=str(clause.get("clause_label", "")),
                    original_clause_text=str(clause["original_clause_text"]),
                    clause_text=clause_text,
                    word_count=int(clause["word_count"]),
                    char_count=int(clause["char_count"]),
                )
            )
        )

    return records


def build_clause_records_for_documents(
    documents: list[dict[str, object]],
    *,
    min_words: int = 8,
    min_chars: int = 30,
) -> list[dict[str, str | int]]:
    """Build clause records from document dictionaries.

    Each input document should include `document_id`, `source_path`, and `text`.
    Missing values are treated defensively as empty strings.
    """

    records: list[dict[str, str | int]] = []
    for document in documents:
        records.extend(
            build_clause_records(
                document_id=str(document.get("document_id", "")),
                source_path=str(document.get("source_path", "")),
                text=document.get("text", ""),
                min_words=min_words,
                min_chars=min_chars,
            )
        )
    return records


def _split_by_markers(text: str) -> list[tuple[str, str]]:
    matches = list(_CLAUSE_MARKER_RE.finditer(text))
    if not matches:
        return [("", text)]

    segments: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append(("", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = _normalize_label(match.group("marker"))
        segment = text[start:end].strip()
        if segment:
            segments.append((label, segment))

    return segments


def _split_by_sentences(text: str) -> list[tuple[str, str]]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    segments: list[tuple[str, str]] = []

    for paragraph in paragraphs or [text]:
        pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY_RE.split(paragraph) if piece.strip()]
        if not pieces:
            continue
        if len(pieces) == 1:
            segments.append(("", pieces[0]))
        else:
            segments.extend(("", piece) for piece in pieces)

    return segments


def _strip_leading_marker(text: str, label: str) -> str:
    if not label:
        return text
    escaped = re.escape(label)
    suffix = r"\.?\s*" if not label.endswith(")") else r"\s*"
    pattern = re.compile(rf"^\s*{escaped}{suffix}", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


def _normalize_label(label: str) -> str:
    return flatten_text(label).rstrip(".")


def _make_clause_id(document_id: str, clause_number: int, clause_text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{clause_number}:{clause_text}".encode("utf-8")).hexdigest()[:10]
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", document_id).strip("_").lower() or "document"
    return f"{prefix}_clause_{clause_number:04d}_{digest}"
