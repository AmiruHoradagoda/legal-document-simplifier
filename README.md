# LegalEase: AI Legal Document Simplifier

LegalEase is a Jupyter-notebook-first Python machine learning project for educational legal document assistance. It extracts text from legal documents, cleans and splits clauses, builds training datasets, fine-tunes lightweight NLP models, evaluates outputs, adds retrieval-augmented document Q&A, and provides a Streamlit demo.

> Legal disclaimer: LegalEase is for educational assistance only. It does not provide legal advice, does not replace a lawyer, and should not be used as the sole basis for legal or financial decisions.

## Project Overview

The project supports an end-to-end legal document simplification workflow:

- Import PDF, TXT, and DOCX files.
- Extract raw document text with PyMuPDF, pdfplumber, plain text readers, and python-docx.
- Clean extracted text and split it into legal clauses.
- Create simplification and clause classification datasets from public Hugging Face data.
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

For Google Colab, use the Colab-specific dependency file to avoid replacing Colab's built-in Jupyter packages:

```bash
pip install -r requirements-colab.txt
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

Notebook 03 downloads public training data by default, so you do not need to provide a dataset. Uploaded documents are still useful for demo extraction, RAG, and app testing.

If you only want to train with public datasets in Colab, install requirements and start at `notebooks/03_build_training_datasets.ipynb`, then run notebooks 04, 05, and 06.

## Data Layout

```text
data/
  raw/
    pdfs/
      .gitkeep
  processed/
    extracted_text.csv
    clauses.csv
    simplification_dataset.csv
    classification_dataset.csv
  evaluation/
    human_eval_template.csv
    results.csv
```

Input demo files can go in `data/raw/pdfs/`. The folder name is historical; the loader supports `.pdf`, `.txt`, and `.docx`.

Training data is created by `notebooks/03_build_training_datasets.ipynb` from the public Hugging Face dataset `coastalcph/lex_glue` with the `ledgar` subset.

Processed files:

- `extracted_text.csv`: document text extracted from uploaded/source files.
- `clauses.csv`: cleaned clause-level records with numbering and text.
- `simplification_dataset.csv`: `clause_id`, `clause_text`, weak auto-generated `simple_clause`, manual-review flag, and `split`.
- `classification_dataset.csv`: `clause_id`, `clause_text`, LEDGAR `clause_type`, weak `risk_level`/`risk_type`, rule reason, and `split`.
- `results.csv`: automatic evaluation metrics.
- `human_eval_template.csv`: 20-row template for manual review.

See `data/README.md` for the full column-by-column schema. `clauses.csv` is for optional uploaded-document processing and RAG, not the Hugging Face training dataset.

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

Notebook 03 creates weak simplification targets automatically because public expert-written legal simplification pairs are limited. These targets are suitable for demonstrating the pipeline, but human-written simplifications are still better for final quality.

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

The `clause_type` labels come from LEDGAR. `risk_level` and `risk_type` are still weak keyword-rule labels.

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
    classifier.py
    clause_splitter.py
    dataset_builder.py
    document_loader.py
    evaluator.py
    preprocessing.py
    rag_qa.py
    risk_rules.py
    simplifier.py
  tests/
```

## Limitations

- Notebook 03 samples the public LEDGAR dataset to stay Colab-friendly; increase row limits for stronger training.
- `risk_level` and `risk_type` labels are rule-generated and may be noisy.
- Simplification targets are weak auto-generated targets, not expert-written plain-language rewrites.
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
