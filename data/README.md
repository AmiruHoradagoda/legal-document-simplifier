# Data Directory Schema

This project uses a Hugging Face dataset flow. Notebook 03 downloads public LEDGAR clauses from `coastalcph/lex_glue` and writes model-ready CSV files.

Important files:

- `data/processed/simplification_dataset.csv`
- `data/processed/classification_dataset.csv`
- `data/evaluation/results.csv`
- `data/evaluation/human_eval_template.csv`

## `processed/simplification_dataset.csv`

Used by Notebook 04 to train the FLAN-T5 simplifier.

Columns:

- `clause_id`: stable row id for tracking predictions and evaluation rows.
- `clause_text`: original legal clause input to the simplifier.
- `simple_clause`: target simplified clause used as the training label.
- `needs_manual_simplification`: whether the target should be reviewed or replaced by a human-written simplification.
- `split`: train, validation, or test.

The simplification target is weak and automatically generated for pipeline training.

## `processed/classification_dataset.csv`

Used by Notebook 05 to train the LegalBERT/DistilBERT classifier and by Notebook 07 for dataset Q&A.

Columns:

- `clause_id`: stable row id for tracking predictions and evaluation rows.
- `clause_text`: legal clause input to the classifier.
- `clause_type`: classifier target label from LEDGAR.
- `risk_level`: weak rule-based risk label for supporting risk analysis.
- `risk_type`: weak rule-based risk category.
- `weak_label_reason`: explanation of how risk labels were assigned.
- `split`: train, validation, or test.

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
