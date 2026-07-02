# Bio Tag Analysis: Slot Filling and Intent Detection

> A compact spoken-language-understanding pipeline for predicting **intent labels** and **BIO slot tags** from airline travel queries.

## Project Summary

This project analyzes and models a labelled airline-query dataset where each row contains:

```text
word:slot word:slot ... <=> intent
```

For example, a sentence such as “flights from Boston to Denver” is represented as token-level BIO slot labels plus one utterance-level intent. The notebook, `Assignment_2_Task_2.ipynb`, turns that raw format into a full modelling pipeline:

1. Load and parse the provided data with `SLUDataLoader`.
2. Explore slot distributions, sequence lengths, common tokens, BIO transitions, and predictive words.
3. Check data quality issues such as malformed examples, invalid BIO sequences, duplicate labels, and near-duplicate utterances.
4. Create a leakage-aware train/validation split that keeps duplicate examples together.
5. Train a classical baseline using:
   - CRF for slot filling;
   - Linear SVM for intent classification.
6. Train and tune a neural **BiLSTM-CRF** slot tagger.
7. Generate final predictions for `data/student_test.txt`.

The final prediction file is:

```text
A2_Task_2_pred.txt
```

## Why Two Models?

The task has two related but different prediction problems:

- **Intent detection:** one label for the whole utterance.
- **Slot filling:** one BIO tag for each token.

The SVM intent classifier performs strongly and is simple to interpret, so it is kept for final intent predictions. Slot filling is more sequence-sensitive, so the final slot model uses a **BiLSTM-CRF**: the BiLSTM learns token context, and the CRF decodes globally consistent BIO tag sequences.

## Latest Results

The most recent full notebook execution produced:

| Model | Metric | Score |
|---|---:|---:|
| CRF/SVM baseline | Intent Accuracy | 0.9611 |
| CRF/SVM baseline | Slot F1 | 0.6700 |
| CRF/SVM baseline | Entity F1 | 0.9351 |
| Untuned BiLSTM-CRF | Token F1 | 0.7799 |
| Untuned BiLSTM-CRF | Entity F1 | 0.9662 |
| Tuned BiLSTM-CRF | Token F1 | 0.8068 |
| Tuned BiLSTM-CRF | Entity F1 | 0.9716 |

The tuned BiLSTM-CRF is the strongest slot model and is used for final slot predictions. The SVM pipeline is used for final intent predictions.

## Model Artifacts

The notebook writes two PyTorch checkpoint files:

- `bilstm_crf_slots.pt`: weights only for the earlier untuned BiLSTM-CRF checkpoint. This file requires the matching architecture and vocabularies to be recreated manually.
- `bilstm_crf_bundle.pt`: the preferred reusable artifact. It contains the tuned model weights, model config, word vocabulary, slot vocabulary, and intent vocabulary.

For future inference or handoff, keep `bilstm_crf_bundle.pt`.

## Reproducing the Pipeline

Run the notebook 

The notebook now sets a global seed near the top:

```python
GLOBAL_SEED = 42
```

The split also uses `random_state=42`, and the hyperparameter search uses a fixed random iterator seed. This makes future runs much more reproducible. Small numerical differences can still occur across hardware, PyTorch versions, or GPU backends.

## Important Files

| File | Purpose |
|---|---|
| `Assignment_2_Task_2.ipynb` | Main analysis, modelling, tuning, and inference notebook |
| `data/train.txt` | Labelled training data |
| `data/student_test.txt` | Unlabelled test utterances |
| `templates/data_loader_template.py` | Parser and vocabulary helper |
| `templates/evaluation_template.py` | Intent and slot evaluation metrics |
| `A2_Task_2_pred.txt` | Final generated predictions |
| `bilstm_crf_bundle.pt` | Best tuned BiLSTM-CRF checkpoint bundle |

## Notes

During validation, a few rare slot labels appeared only in the validation split. The notebook adds those labels to the slot vocabulary for safe evaluation encoding while keeping the word vocabulary train-only, so validation still reflects realistic out-of-vocabulary words.
