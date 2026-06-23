# LegalEase: AI Legal Document Simplifier

LegalEase is a Jupyter-notebook-first Python machine learning project for educational legal document assistance. It extracts text from legal documents, cleans and splits clauses, builds training datasets, fine-tunes lightweight NLP models, evaluates outputs, adds retrieval-augmented document Q&A, and provides a Streamlit demo.

> Legal disclaimer: LegalEase is for educational assistance only. It does not provide legal advice, does not replace a lawyer, and should not be used as the sole basis for legal or financial decisions.

## Project Overview

The project supports an end-to-end legal document simplification workflow:

- Import PDF, TXT, and DOCX files.
- Extract raw document text with PyMuPDF, pdfplumber, plain text readers, and python-docx.
- Clean extracted text and split it into legal clauses.
- Create simplification and clause classification datasets.
- Fine-tune a small text-to-text simplifier with `google/flan-t5-small`.
- Fine-tune a clause type classifier with `nlpaueb/legal-bert-base-uncased`, with fallback to `distilbert-base-uncased`.
- Evaluate simplification and classification outputs.
- Build document Q&A using SentenceTransformers and FAISS.
- Run a Streamlit app for upload, simplification, classification, risk rules, Q&A, and report downloads.

## Setup With Conda

Use Python 3.10. This project is intended to run in a Conda environment.

```bash
conda create -n legal-ai python=3.10
conda activate legal-ai
pip install -r requirements.txt
python -m ipykernel install --user --name legal-ai --display-name "Python (legal-ai)"
```

If PyTorch installation needs a platform-specific wheel, install the correct PyTorch build first from the official PyTorch selector, then rerun:

```bash
pip install -r requirements.txt
```

## Notebook Execution Order

Run notebooks from top to bottom in this order:

1. `notebooks/01_data_import_and_extraction.ipynb`
2. `notebooks/02_preprocessing_and_clause_split.ipynb`
3. `notebooks/03_build_training_datasets.ipynb`
4. `notebooks/04_train_simplifier_flan_t5.ipynb`
5. `notebooks/05_train_clause_classifier_legalbert.ipynb`
6. `notebooks/06_evaluation.ipynb`
7. `notebooks/07_rag_document_qa.ipynb`
8. `notebooks/08_end_to_end_demo_test.ipynb`

The notebooks are designed to run on small sample data first. Add larger datasets only after the full workflow works.

## Data Layout

```text
data/
  raw/
    pdfs/
      sample_lease_clause.txt
    sample_clauses.csv
  processed/
    extracted_text.csv
    clauses.csv
    simplification_dataset.csv
    classification_dataset.csv
  evaluation/
    human_eval_template.csv
    results.csv
```

Input files go in `data/raw/pdfs/`. The folder name is historical; the loader supports `.pdf`, `.txt`, and `.docx`.

Processed files:

- `extracted_text.csv`: document text extracted from uploaded/source files.
- `clauses.csv`: cleaned clause-level records with numbering and text.
- `simplification_dataset.csv`: original clause text plus manually filled `simple_clause` targets.
- `classification_dataset.csv`: clause text plus weak labels for `clause_type`, `risk_level`, and `risk_type`.
- `results.csv`: automatic evaluation metrics.
- `human_eval_template.csv`: 20-row template for manual review.

## Model Training

### Simplifier

Run:

```text
notebooks/04_train_simplifier_flan_t5.ipynb
```

Model:

- Primary model: `google/flan-t5-small`
- Prompt prefix: `simplify legal text: `
- Input data: `data/processed/simplification_dataset.csv`
- Required target column: `simple_clause`
- Output model path: `models/simplifier/`
- Prediction output: `outputs/predictions/simplifier_predictions.csv`

The included sample targets are only for smoke testing. Replace or expand them with manually reviewed simplifications before reporting model quality.

### Clause Classifier

Run:

```text
notebooks/05_train_clause_classifier_legalbert.ipynb
```

Model:

- Primary model: `nlpaueb/legal-bert-base-uncased`
- Fallback model: `distilbert-base-uncased`
- Classification target: `clause_type`
- Input data: `data/processed/classification_dataset.csv`
- Output model path: `models/clause_classifier/`
- Label mapping: `models/clause_classifier/label_mapping.json`
- Prediction output: `outputs/predictions/clause_classifier_predictions.csv`

The current labels are weak labels from keyword rules. Review and improve labels before serious training.

## Evaluation

Run:

```text
notebooks/06_evaluation.ipynb
```

Metrics included:

- Simplification: ROUGE, BERTScore, readability before/after, compression ratio.
- Classification: accuracy, precision, recall, F1-score, confusion matrix.

Outputs:

- `data/evaluation/results.csv`
- `data/evaluation/human_eval_template.csv`
- `outputs/charts/simplifier_readability_before_after.png`
- `outputs/charts/simplifier_compression_ratio.png`
- `outputs/charts/simplifier_rouge_bertscore.png`
- `outputs/charts/clause_classifier_metrics.png`
- `outputs/charts/clause_classifier_confusion_matrix.csv`
- `outputs/charts/clause_classifier_confusion_matrix.png`

Current evaluation files may contain placeholders until notebooks 04 and 05 are run and prediction CSVs are populated.

## RAG Document Q&A

Run:

```text
notebooks/07_rag_document_qa.ipynb
```

The Q&A pipeline:

- Loads `data/processed/clauses.csv`.
- Embeds clauses with `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a FAISS index with `faiss-cpu`.
- Retrieves top-k relevant clauses for a question.
- Returns a grounded answer using only retrieved clauses.

If SentenceTransformers or FAISS is unavailable, the code falls back to lexical retrieval.

## Streamlit App

After installing dependencies and optionally training models, run:

```bash
conda activate legal-ai
streamlit run app.py
```

The app supports:

- PDF/TXT/DOCX upload.
- Text extraction, cleaning, and clause splitting.
- Saved simplifier model inference.
- Saved clause classifier inference.
- Rule-based risk detection.
- RAG-style document Q&A.
- CSV and TXT report downloads.

If trained models are missing, the app falls back to original clauses for simplification and keyword-based labels for classification.

## Repository Layout

```text
legal-document-simplifier/
  app.py
  requirements.txt
  README.md
  .env.example
  data/
  docs/
  models/
    simplifier/
    clause_classifier/
  notebooks/
  outputs/
    charts/
    predictions/
    simplified_reports/
  src/
  tests/
```

## Limitations

- The included sample data is intentionally tiny and is only suitable for smoke testing.
- Weak classification labels are rule-generated and may be noisy.
- Simplification quality depends on manually reviewed `simple_clause` targets.
- Small models may miss legal nuance and can produce incomplete or inaccurate simplifications.
- RAG answers are limited to retrieved clauses and may miss information in unretrieved sections.
- This project does not perform legal reasoning or jurisdiction-specific legal validation.
- Outputs require human review before any real-world use.

## Testing

After installing dependencies:

```bash
conda activate legal-ai
python -m pytest tests/
```

Some tests use lightweight fallback paths. Full model training and Streamlit testing require the ML dependencies and downloaded model files.

## Legal Disclaimer

LegalEase is for educational assistance only and does not provide legal advice. The generated simplifications, classifications, risk labels, and Q&A responses may be incomplete or incorrect. Consult a qualified legal professional for advice about specific documents, rights, duties, or legal decisions.
