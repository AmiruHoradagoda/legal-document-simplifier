"""Retrieval-augmented document Q&A utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_DISCLAIMER = "Educational assistance only; this system does not provide legal advice."


@dataclass(frozen=True)
class RetrievedClause:
    """A retrieved clause and its relevance score."""

    clause_id: str
    document_id: str
    clause_number: int
    clause_text: str
    score: float


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the sentence-transformer embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "RAG embeddings require sentence-transformers. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return SentenceTransformer(model_name)


def embed_texts(model, texts: Iterable[str]) -> np.ndarray:
    """Encode text into normalized float32 embeddings."""

    import numpy as np

    text_list = [str(text or "") for text in texts]
    if not text_list:
        return np.empty((0, 0), dtype="float32")

    embeddings = model.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    """Build an inner-product FAISS index for normalized embeddings."""

    import numpy as np

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("FAISS indexing requires faiss-cpu. Install dependencies with `pip install -r requirements.txt`.") from exc

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.asarray(embeddings, dtype="float32"))
    return index


def retrieve_top_k(
    question: str,
    clauses: list[dict[str, object]],
    *,
    model=None,
    index=None,
    embeddings: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Retrieve top-k clauses with FAISS when available, otherwise lexical scoring."""

    if top_k <= 0:
        return []
    usable_clauses = [clause for clause in clauses if str(clause.get("clause_text", "")).strip()]
    if not usable_clauses or not str(question or "").strip():
        return []

    top_k = min(top_k, len(usable_clauses))

    if model is not None and index is not None:
        query_embedding = embed_texts(model, [question])
        scores, indexes = index.search(query_embedding, top_k)
        results = []
        for score, row_index in zip(scores[0], indexes[0]):
            if row_index < 0 or row_index >= len(usable_clauses):
                continue
            results.append(_result_row(usable_clauses[int(row_index)], float(score)))
        return results

    if embeddings is not None and index is not None:
        raise ValueError("A model is required to embed the question for FAISS retrieval.")

    scored = [
        _result_row(clause, lexical_relevance_score(question, str(clause.get("clause_text", ""))))
        for clause in usable_clauses
    ]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:top_k]


def lexical_relevance_score(question: str, clause_text: str) -> float:
    """Compute a simple token-overlap relevance score."""

    query_tokens = set(_tokens(question))
    clause_tokens = set(_tokens(clause_text))
    if not query_tokens or not clause_tokens:
        return 0.0

    overlap = len(query_tokens & clause_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(clause_tokens))


def generate_rule_based_answer(question: str, retrieved_clauses: list[dict[str, object]]) -> str:
    """Generate a grounded answer summary using only retrieved clauses."""

    if not retrieved_clauses:
        return (
            f"{LEGAL_DISCLAIMER} I could not find a relevant clause in the loaded document. "
            "Review the source document directly."
        )

    best_clause = retrieved_clauses[0]
    clause_text = str(best_clause.get("clause_text", "")).strip()
    clause_number = best_clause.get("clause_number", "")

    summary = _first_sentence(clause_text)
    source_label = f"Clause {clause_number}" if str(clause_number).strip() else "Top retrieved clause"

    return (
        f"{LEGAL_DISCLAIMER} Based on {source_label}, {summary} "
        "This answer is limited to the retrieved clauses and should be verified against the full document."
    )


def answer_question(
    question: str,
    clauses: list[dict[str, object]],
    *,
    model=None,
    index=None,
    embeddings: np.ndarray | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    """Retrieve clauses and return a grounded rule-based answer."""

    retrieved = retrieve_top_k(
        question,
        clauses,
        model=model,
        index=index,
        embeddings=embeddings,
        top_k=top_k,
    )
    return {
        "question": question,
        "answer": generate_rule_based_answer(question, retrieved),
        "retrieved_clauses": retrieved,
    }


def _result_row(clause: dict[str, object], score: float) -> dict[str, object]:
    return {
        "clause_id": str(clause.get("clause_id", "")),
        "document_id": str(clause.get("document_id", "")),
        "clause_number": _safe_int(clause.get("clause_number")),
        "clause_text": str(clause.get("clause_text", "")),
        "score": float(score),
    }


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _tokens(text: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _first_sentence(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return pieces[0].strip() if pieces and pieces[0].strip() else text.strip()
