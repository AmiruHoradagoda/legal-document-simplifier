"""Streamlit demo for LegalEase."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.clause_splitter import build_clause_records
from src.dataset_builder import label_clause_with_rules
from src.document_loader import SUPPORTED_EXTENSIONS, extract_text_from_file
from src.preprocessing import clean_legal_text
from src.rag_qa import (
    DEFAULT_EMBEDDING_MODEL,
    answer_question,
    build_faiss_index,
    embed_texts,
    load_embedding_model,
)
from src.risk_rules import apply_risk_rules_to_rows
from src.simplifier import INPUT_PREFIX


DISCLAIMER = (
    "LegalEase is for educational assistance only and does not provide legal advice. "
    "Always review important legal questions with a qualified professional."
)

PROJECT_ROOT = Path(__file__).resolve().parent
SIMPLIFIER_DIR = PROJECT_ROOT / "models" / "simplifier"
CLASSIFIER_DIR = PROJECT_ROOT / "models" / "clause_classifier"


@st.cache_resource(show_spinner=False)
def load_simplifier_resources(model_dir: str):
    """Load the saved simplifier model and tokenizer."""

    path = Path(model_dir)
    if not (path / "config.json").exists():
        return None

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return {"tokenizer": tokenizer, "model": model}
    except Exception as exc:  # pragma: no cover - depends on local model files
        return {"error": str(exc)}


@st.cache_resource(show_spinner=False)
def load_classifier_resources(model_dir: str):
    """Load the saved classifier model, tokenizer, and label mapping."""

    path = Path(model_dir)
    if not (path / "config.json").exists():
        return None

    try:
        from src.classifier import load_label_mapping
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        label_mapping_path = path / "label_mapping.json"
        _, id2label = load_label_mapping(label_mapping_path) if label_mapping_path.exists() else ({}, {})
        return {"tokenizer": tokenizer, "model": model, "id2label": id2label}
    except Exception as exc:  # pragma: no cover - depends on local model files
        return {"error": str(exc)}


@st.cache_resource(show_spinner=False)
def load_rag_embedding_model():
    """Load the RAG embedding model."""

    try:
        return load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover - depends on local model files
        return {"error": str(exc)}


def initialize_state() -> None:
    defaults = {
        "raw_text": "",
        "document_id": "",
        "source_name": "",
        "clauses_df": pd.DataFrame(),
        "report_df": pd.DataFrame(),
        "rag_mode": "not_built",
        "rag_index": None,
        "rag_embeddings": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploaded_file(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())
        extracted_rows = extract_text_from_file(temp_path, base_dir=temp_dir)

    text_parts = [
        str(row.get("text", "")).strip()
        for row in extracted_rows
        if str(row.get("text", "")).strip() and not str(row.get("error", "")).strip()
    ]
    if not text_parts:
        errors = "; ".join(str(row.get("error", "")) for row in extracted_rows if row.get("error"))
        raise ValueError(errors or "No text could be extracted from the uploaded document.")

    raw_text = "\n\n".join(text_parts)
    cleaned_text = clean_legal_text(raw_text)
    first_row = extracted_rows[0]
    document_id = str(first_row.get("document_id", Path(uploaded_file.name).stem))
    clause_rows = build_clause_records(
        document_id=document_id,
        source_path=uploaded_file.name,
        text=cleaned_text,
        min_words=5,
        min_chars=20,
    )

    clauses_df = pd.DataFrame(clause_rows)
    st.session_state.raw_text = cleaned_text
    st.session_state.document_id = document_id
    st.session_state.source_name = uploaded_file.name
    st.session_state.clauses_df = clauses_df
    st.session_state.report_df = build_base_report(clauses_df)
    st.session_state.rag_mode = "not_built"
    st.session_state.rag_index = None
    st.session_state.rag_embeddings = None


def build_base_report(clauses_df: pd.DataFrame) -> pd.DataFrame:
    if clauses_df.empty:
        return pd.DataFrame()

    rows = apply_risk_rules_to_rows(clauses_df.to_dict(orient="records"))
    report_df = pd.DataFrame(rows)
    for column in ["simplified_clause", "predicted_clause_type", "classifier_confidence"]:
        if column not in report_df.columns:
            report_df[column] = ""
    return report_df


def run_simplifier(clauses: list[str]) -> tuple[list[str], str]:
    resources = load_simplifier_resources(str(SIMPLIFIER_DIR))
    if not resources:
        return clauses, "No saved simplifier model found. Showing original clauses as fallback."
    if "error" in resources:
        return clauses, f"Simplifier could not be loaded: {resources['error']}"

    try:
        tokenizer = resources["tokenizer"]
        model = resources["model"]
        prompts = [f"{INPUT_PREFIX}{clause}" for clause in clauses]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        generated_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True), "Simplifier model used."
    except Exception as exc:  # pragma: no cover - depends on local model files
        return clauses, f"Simplifier inference failed: {exc}"


def run_classifier(clauses: list[str]) -> tuple[list[str], list[float], str]:
    resources = load_classifier_resources(str(CLASSIFIER_DIR))
    if not resources:
        labels = [label_clause_with_rules(clause)["clause_type"] for clause in clauses]
        return labels, [0.0] * len(labels), "No saved classifier model found. Used keyword fallback labels."
    if "error" in resources:
        labels = [label_clause_with_rules(clause)["clause_type"] for clause in clauses]
        return labels, [0.0] * len(labels), f"Classifier could not be loaded: {resources['error']}"

    try:
        import numpy as np

        tokenizer = resources["tokenizer"]
        model = resources["model"]
        id2label = resources["id2label"]
        inputs = tokenizer(clauses, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        predicted_ids = probabilities.argmax(axis=1)
        labels = [id2label.get(int(index), str(index)) for index in predicted_ids]
        confidences = probabilities.max(axis=1).round(4).tolist()
        return labels, confidences, "Classifier model used."
    except Exception as exc:  # pragma: no cover - depends on local model files
        labels = [label_clause_with_rules(clause)["clause_type"] for clause in clauses]
        return labels, [0.0] * len(labels), f"Classifier inference failed: {exc}"


def build_rag_resources() -> None:
    clauses_df = st.session_state.clauses_df
    if clauses_df.empty:
        return

    embedding_model = load_rag_embedding_model()
    if isinstance(embedding_model, dict) and "error" in embedding_model:
        st.session_state.rag_mode = f"lexical fallback: {embedding_model['error']}"
        st.session_state.rag_index = None
        st.session_state.rag_embeddings = None
        return

    try:
        embeddings = embed_texts(embedding_model, clauses_df["clause_text"].tolist())
        index = build_faiss_index(embeddings)
        st.session_state.rag_mode = "sentence-transformers + FAISS"
        st.session_state.rag_index = index
        st.session_state.rag_embeddings = embeddings
    except Exception as exc:  # pragma: no cover - depends on local dependencies
        st.session_state.rag_mode = f"lexical fallback: {exc}"
        st.session_state.rag_index = None
        st.session_state.rag_embeddings = None


def report_to_txt(report_df: pd.DataFrame) -> str:
    lines = [DISCLAIMER, ""]
    for _, row in report_df.iterrows():
        lines.append(f"Clause {row.get('clause_number', '')}")
        lines.append(f"Original: {row.get('clause_text', '')}")
        lines.append(f"Simplified: {row.get('simplified_clause', '')}")
        lines.append(f"Clause type: {row.get('predicted_clause_type', '')}")
        lines.append(f"Rule risk: {row.get('rule_risk_level', '')} / {row.get('rule_risk_type', '')}")
        lines.append("")
    return "\n".join(lines)


def render_upload_page() -> None:
    st.header("Upload and Extract")
    uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX document", type=["pdf", "txt", "docx"])
    if uploaded_file and st.button("Process Document", type="primary"):
        try:
            with st.spinner("Extracting text and splitting clauses..."):
                process_uploaded_file(uploaded_file)
            st.success(f"Processed {len(st.session_state.clauses_df)} clause(s).")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    if st.session_state.raw_text:
        st.subheader("Extracted Text Preview")
        st.text_area("Cleaned text", st.session_state.raw_text[:5000], height=220)
        st.subheader("Clauses")
        st.dataframe(st.session_state.clauses_df, use_container_width=True, hide_index=True)


def render_simplification_page() -> None:
    st.header("Simplification")
    report_df = st.session_state.report_df
    if report_df.empty:
        st.info("Upload and process a document first.")
        return

    if st.button("Run Simplifier", type="primary"):
        with st.spinner("Running simplifier..."):
            simplified, message = run_simplifier(report_df["clause_text"].tolist())
            report_df = report_df.copy()
            report_df["simplified_clause"] = simplified
            st.session_state.report_df = report_df
        st.info(message)

    columns = ["clause_number", "clause_text", "simplified_clause"]
    st.dataframe(st.session_state.report_df[columns], use_container_width=True, hide_index=True)


def render_risk_page() -> None:
    st.header("Clause Classification and Risk Rules")
    report_df = st.session_state.report_df
    if report_df.empty:
        st.info("Upload and process a document first.")
        return

    if st.button("Run Classifier and Risk Rules", type="primary"):
        with st.spinner("Classifying clauses..."):
            labels, confidences, message = run_classifier(report_df["clause_text"].tolist())
            updated = pd.DataFrame(apply_risk_rules_to_rows(report_df.to_dict(orient="records")))
            updated["predicted_clause_type"] = labels
            updated["classifier_confidence"] = confidences
            st.session_state.report_df = updated
        st.info(message)

    columns = [
        "clause_number",
        "clause_text",
        "predicted_clause_type",
        "classifier_confidence",
        "rule_risk_level",
        "rule_risk_type",
        "rule_matches",
    ]
    st.dataframe(st.session_state.report_df[columns], use_container_width=True, hide_index=True)


def render_qa_page() -> None:
    st.header("Document Q&A")
    if st.session_state.clauses_df.empty:
        st.info("Upload and process a document first.")
        return

    if st.button("Build Q&A Index", type="primary"):
        with st.spinner("Building retrieval index..."):
            build_rag_resources()
        st.info(f"Retrieval mode: {st.session_state.rag_mode}")

    question = st.text_input("Ask a question about the uploaded document", value="What happens if payment is late?")
    top_k = st.slider("Clauses to retrieve", min_value=1, max_value=10, value=3)
    if st.button("Answer Question"):
        embedding_model = None
        if st.session_state.rag_index is not None:
            loaded = load_rag_embedding_model()
            embedding_model = None if isinstance(loaded, dict) else loaded

        result = answer_question(
            question,
            st.session_state.clauses_df.to_dict(orient="records"),
            model=embedding_model,
            index=st.session_state.rag_index,
            embeddings=st.session_state.rag_embeddings,
            top_k=top_k,
        )
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Retrieved Clauses")
        st.dataframe(pd.DataFrame(result["retrieved_clauses"]), use_container_width=True, hide_index=True)


def render_downloads_page() -> None:
    st.header("Downloads")
    report_df = st.session_state.report_df
    if report_df.empty:
        st.info("Upload and process a document first.")
        return

    st.dataframe(report_df, use_container_width=True, hide_index=True)
    csv_data = report_df.to_csv(index=False).encode("utf-8")
    txt_data = report_to_txt(report_df).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="legalease_simplified_report.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download TXT Report",
            data=txt_data,
            file_name="legalease_simplified_report.txt",
            mime="text/plain",
        )


def main() -> None:
    st.set_page_config(page_title="LegalEase", layout="wide")
    initialize_state()

    st.title("LegalEase: AI Legal Document Simplifier")
    st.warning(DISCLAIMER)

    page = st.sidebar.radio(
        "Page",
        ["Upload", "Simplification", "Classification and Risk", "Q&A", "Downloads"],
    )

    if page == "Upload":
        render_upload_page()
    elif page == "Simplification":
        render_simplification_page()
    elif page == "Classification and Risk":
        render_risk_page()
    elif page == "Q&A":
        render_qa_page()
    else:
        render_downloads_page()


if __name__ == "__main__":
    main()
