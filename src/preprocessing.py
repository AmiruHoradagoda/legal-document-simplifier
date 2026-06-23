"""Text preprocessing utilities for extracted legal documents."""

from __future__ import annotations

import re
import unicodedata


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def normalize_unicode(text: object) -> str:
    """Normalize text to a predictable Unicode representation."""

    if text is None:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def normalize_whitespace(text: object, *, preserve_newlines: bool = True) -> str:
    """Normalize repeated whitespace while optionally preserving line breaks."""

    normalized = normalize_unicode(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHARS_RE.sub(" ", normalized)
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)

    if preserve_newlines:
        lines = [_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
        normalized = "\n".join(lines)
        normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    else:
        normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def remove_repeated_page_artifacts(text: object, *, max_line_length: int = 120) -> str:
    """Remove short lines that repeat often, such as headers or footers."""

    normalized = normalize_whitespace(text, preserve_newlines=True)
    lines = normalized.split("\n")
    counts: dict[str, int] = {}

    for line in lines:
        key = line.strip().lower()
        if key and len(key) <= max_line_length:
            counts[key] = counts.get(key, 0) + 1

    repeated = {line for line, count in counts.items() if count >= 3}
    if not repeated:
        return normalized

    kept_lines = [line for line in lines if line.strip().lower() not in repeated]
    return normalize_whitespace("\n".join(kept_lines), preserve_newlines=True)


def clean_legal_text(text: object, *, remove_repeated_artifacts: bool = True) -> str:
    """Clean extracted legal text while keeping useful clause boundaries."""

    cleaned = normalize_whitespace(text, preserve_newlines=True)
    cleaned = re.sub(r"(?im)^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "", cleaned)

    if remove_repeated_artifacts:
        cleaned = remove_repeated_page_artifacts(cleaned)

    return normalize_whitespace(cleaned, preserve_newlines=True)


def flatten_text(text: object) -> str:
    """Collapse text to one line for CSV previews and model inputs."""

    return normalize_whitespace(text, preserve_newlines=False)


def count_words(text: object) -> int:
    """Count word-like tokens."""

    return len(_WORD_RE.findall(normalize_unicode(text)))


def is_short_text(text: object, *, min_words: int = 5, min_chars: int = 20) -> bool:
    """Return True when text is too short to be useful as a clause."""

    flattened = flatten_text(text)
    return count_words(flattened) < min_words or len(flattened) < min_chars
