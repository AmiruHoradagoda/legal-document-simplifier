"""Streamlit demo for the Hugging Face dataset workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dataset_builder import label_clause_with_rules
from src.rag_qa import (
    DEFAULT_EMBEDDING_MODEL,
    answer_question,
    build_faiss_index,
    embed_texts,
    load_embedding_model,
)
from src.simplifier import INPUT_PREFIX


DISCLAIMER = (
    "LegalEase is for educational assistance only and does not provide legal advice. "
    "Always review important legal questions with a qualified lawyer or legal professional."
)

PROJECT_ROOT = Path(__file__).resolve().parent
SIMPLIFICATION_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "simplification_dataset.csv"
CLASSIFICATION_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "classification_dataset.csv"
SIMPLIFIER_DIR = PROJECT_ROOT / "models" / "simplifier"
CLASSIFIER_DIR = PROJECT_ROOT / "models" / "clause_classifier"


@st.cache_data(show_spinner=False)
def load_dataset_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Notebook 03 outputs."""

    simplification_columns = ["clause_id", "clause_text", "simple_clause", "split"]
    classification_columns = [
        "clause_id",
        "clause_text",
        "clause_type",
        "risk_level",
        "risk_type",
        "weak_label_reason",
        "split",
    ]

    simplification_df = _read_csv_or_empty(SIMPLIFICATION_DATASET_PATH, simplification_columns)
    classification_df = _read_csv_or_empty(CLASSIFICATION_DATASET_PATH, classification_columns)
    return simplification_df, classification_df


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
        "selected_df": pd.DataFrame(),
        "report_df": pd.DataFrame(),
        "rag_mode": "not_built",
        "rag_index": None,
        "rag_embeddings": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_demo_dataset(
    simplification_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    *,
    split: str,
    max_rows: int,
) -> pd.DataFrame:
    """Build the app working set from Hugging Face-derived CSV rows."""

    if classification_df.empty:
        return pd.DataFrame()

    filtered = classification_df.copy()
    if split != "all":
        filtered = filtered[filtered["split"] == split].copy()

    filtered = filtered.head(max_rows).reset_index(drop=True)
    if filtered.empty:
        return filtered

    simple_lookup = simplification_df.set_index("clause_id")["simple_clause"].to_dict()
    filtered["reference_simple_clause"] = filtered["clause_id"].map(simple_lookup).fillna("")
    filtered["document_id"] = "huggingface_ledgar"
    filtered["clause_number"] = range(1, len(filtered) + 1)
    filtered["simplified_clause"] = ""
    filtered["predicted_clause_type"] = ""
    filtered["classifier_confidence"] = ""
    return filtered


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
    selected_df = st.session_state.selected_df
    if selected_df.empty:
        return

    embedding_model = load_rag_embedding_model()
    if isinstance(embedding_model, dict) and "error" in embedding_model:
        st.session_state.rag_mode = f"lexical fallback: {embedding_model['error']}"
        st.session_state.rag_index = None
        st.session_state.rag_embeddings = None
        return

    try:
        embeddings = embed_texts(embedding_model, selected_df["clause_text"].tolist())
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
        lines.append(f"Reference simplification: {row.get('reference_simple_clause', '')}")
        lines.append(f"Model simplification: {row.get('simplified_clause', '')}")
        lines.append(f"Dataset clause type: {row.get('clause_type', '')}")
        lines.append(f"Predicted clause type: {row.get('predicted_clause_type', '')}")
        lines.append(f"Dataset risk: {row.get('risk_level', '')} / {row.get('risk_type', '')}")
        lines.append("")
    return "\n".join(lines)


def render_dataset_page(simplification_df: pd.DataFrame, classification_df: pd.DataFrame) -> None:
    st.header("Hugging Face Dataset")
    if simplification_df.empty or classification_df.empty:
        st.error("Run Notebook 03 first to create the Hugging Face-derived dataset CSVs.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        split = st.selectbox("Split", ["all", "train", "validation", "test"], index=0)
    with col2:
        max_rows = st.number_input("Rows", min_value=5, max_value=200, value=25, step=5)
    with col3:
        st.metric("Classification rows", len(classification_df))

    if st.button("Load Rows", type="primary"):
        selected_df = build_demo_dataset(
            simplification_df,
            classification_df,
            split=split,
            max_rows=int(max_rows),
        )
        st.session_state.selected_df = selected_df
        st.session_state.report_df = selected_df.copy()
        st.session_state.rag_mode = "not_built"
        st.session_state.rag_index = None
        st.session_state.rag_embeddings = None

    if st.session_state.selected_df.empty:
        st.info("Load rows from the Hugging Face-derived dataset to start the demo.")
    else:
        st.dataframe(st.session_state.selected_df, use_container_width=True, hide_index=True)


def render_simplification_page() -> None:
    st.header("Simplification")
    report_df = st.session_state.report_df
    if report_df.empty:
        st.info("Load dataset rows first.")
        return

    if st.button("Run Simplifier", type="primary"):
        with st.spinner("Running simplifier..."):
            simplified, message = run_simplifier(report_df["clause_text"].tolist())
            report_df = report_df.copy()
            report_df["simplified_clause"] = simplified
            st.session_state.report_df = report_df
        st.info(message)

    columns = ["clause_number", "clause_text", "reference_simple_clause", "simplified_clause"]
    st.dataframe(st.session_state.report_df[columns], use_container_width=True, hide_index=True)


def render_classification_page() -> None:
    st.header("Classification and Risk Labels")
    report_df = st.session_state.report_df
    if report_df.empty:
        st.info("Load dataset rows first.")
        return

    if st.button("Run Classifier", type="primary"):
        with st.spinner("Classifying clauses..."):
            labels, confidences, message = run_classifier(report_df["clause_text"].tolist())
            updated = report_df.copy()
            updated["predicted_clause_type"] = labels
            updated["classifier_confidence"] = confidences
            st.session_state.report_df = updated
        st.info(message)

    columns = [
        "clause_number",
        "clause_text",
        "clause_type",
        "predicted_clause_type",
        "classifier_confidence",
        "risk_level",
        "risk_type",
        "weak_label_reason",
    ]
    st.dataframe(st.session_state.report_df[columns], use_container_width=True, hide_index=True)


def render_qa_page() -> None:
    st.header("Dataset Q&A")
    if st.session_state.selected_df.empty:
        st.info("Load dataset rows first.")
        return

    if st.button("Build Q&A Index", type="primary"):
        with st.spinner("Building retrieval index..."):
            build_rag_resources()
        st.info(f"Retrieval mode: {st.session_state.rag_mode}")

    question = st.text_input("Ask a question about the loaded dataset rows", value="What happens if payment is late?")
    top_k = st.slider("Clauses to retrieve", min_value=1, max_value=10, value=3)
    if st.button("Answer Question"):
        embedding_model = None
        if st.session_state.rag_index is not None:
            loaded = load_rag_embedding_model()
            embedding_model = None if isinstance(loaded, dict) else loaded

        result = answer_question(
            question,
            st.session_state.selected_df.to_dict(orient="records"),
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
        st.info("Load dataset rows first.")
        return

    st.dataframe(report_df, use_container_width=True, hide_index=True)
    csv_data = report_df.to_csv(index=False).encode("utf-8")
    txt_data = report_to_txt(report_df).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="legalease_huggingface_report.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download TXT Report",
            data=txt_data,
            file_name="legalease_huggingface_report.txt",
            mime="text/plain",
        )


def _read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns].fillna("")


def main() -> None:
    st.set_page_config(page_title="LegalEase", layout="wide")
    initialize_state()
    simplification_df, classification_df = load_dataset_csvs()

    st.title("LegalEase: Hugging Face Legal Dataset Demo")
    st.warning(DISCLAIMER)

    page = st.sidebar.radio(
        "Page",
        ["Dataset", "Simplification", "Classification and Risk", "Q&A", "Downloads"],
    )

    if page == "Dataset":
        render_dataset_page(simplification_df, classification_df)
    elif page == "Simplification":
        render_simplification_page()
    elif page == "Classification and Risk":
        render_classification_page()
    elif page == "Q&A":
        render_qa_page()
    else:
        render_downloads_page()


if __name__ == "__main__":
    main()
