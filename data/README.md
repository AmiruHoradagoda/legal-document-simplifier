# Data Directory Schema

This project has two different data flows:

1. Public Hugging Face training flow: starts at Notebook 03 and writes model-ready CSV files.
2. Optional uploaded-document flow: starts at Notebook 01 and writes extracted document/clauses for demos and RAG.

If you are training only with Hugging Face LEDGAR, the most important files are:

- `data/processed/simplification_dataset.csv`
- `data/processed/classification_dataset.csv`

## `processed/simplification_dataset.csv`

Used by Notebook 04 to train the FLAN-T5 simplifier.

Columns:

- `clause_id`: stable row id for tracking predictions and evaluation rows.
- `clause_text`: original legal clause input to the simplifier.
- `simple_clause`: target simplified clause used as the training label.
- `needs_manual_simplification`: whether the target should be reviewed or replaced by a human-written simplification.
- `split`: train, validation, or test.

This is a model-training CSV.

## `processed/classification_dataset.csv`

Used by Notebook 05 to train the LegalBERT/DistilBERT classifier.

Columns:

- `clause_id`: stable row id for tracking predictions and evaluation rows.
- `clause_text`: legal clause input to the classifier.
- `clause_type`: classifier target label from LEDGAR.
- `risk_level`: weak rule-based risk label for supporting risk analysis.
- `risk_type`: weak rule-based risk category.
- `weak_label_reason`: explanation of how risk labels were assigned.
- `split`: train, validation, or test.

This is a model-training CSV.

## `processed/extracted_text.csv`

Used by Notebook 01 and Notebook 02 for optional uploaded-document processing.

Columns:

- `document_id`: generated id for the uploaded/source document.
- `source_path`: original local file path or uploaded filename.
- `page_number`: PDF page number when available.
- `text`: extracted page or document text.
- `word_count`: extracted text word count.
- `char_count`: extracted text character count.
- `extraction_method`: extractor used, such as PyMuPDF, pdfplumber, txt, or docx.
- `error`: extraction error message if a file failed.

This is not used for Hugging Face training unless you choose the local-document fallback path.

## `processed/clauses.csv`

Used by Notebook 02 and Notebook 07 for optional uploaded-document clause processing and RAG.

Columns:

- `clause_id`: stable id for one extracted clause.
- `document_id`: id of the uploaded/source document.
- `source_path`: original local file path or uploaded filename.
- `clause_number`: sequential clause number after splitting.
- `clause_label`: detected legal numbering such as `1`, `1.2`, `(a)`, or `Section 4`.
- `original_clause_text`: clause text before final cleaning/flattening.
- `clause_text`: cleaned clause text.
- `word_count`: clause word count.
- `char_count`: clause character count.

This is not the Hugging Face training dataset. It is for uploaded documents, local fallback training, and RAG demos.

## `evaluation/results.csv`

Used by Notebook 06.

Columns:

- `stage`: simplifier or classifier.
- `metric`: metric name, such as ROUGE, BERTScore, accuracy, precision, recall, F1, readability, or compression ratio.
- `value`: metric value.
- `notes`: how the metric was computed or why it is unavailable.

## `evaluation/human_eval_template.csv`

Used by Notebook 06 for manual review.

Columns:

- `evaluation_id`: review row id.
- `task`: simplification or classification.
- `clause_id`: id of the evaluated clause.
- `split`: train, validation, or test.
- `original_clause`: source legal clause.
- `reference_output`: target simplification or true label.
- `model_output`: model prediction.
- `clarity_score_1_5`: human clarity score.
- `faithfulness_score_1_5`: human faithfulness score.
- `label_correct`: human yes/no correctness value for classifier output.
- `notes`: reviewer notes.

## `raw/`

Optional folder for uploaded demo files used by Notebooks 01 and 02.

Uploaded private documents are ignored by Git. The `.gitkeep` files only preserve the empty folder structure.
