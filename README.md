# AI-generated text detection

Classification of English and German documents into `human`, `ai`, and `fusion`, evaluated with macro-F1. The study compares word TF-IDF, character TF-IDF, multilingual DistilBERT, three-window inference, and probability ensembles.

## Latest result

The ensemble trained on all 272,352 labeled documents achieved **0.91147 public macro-F1** on Kaggle, compared with 0.87518 for the earlier ensemble. It combines word TF-IDF (18.75%), character TF-IDF (56.25%), and multilingual DistilBERT (25%). The character vocabulary came from the earlier 40,000-document model; its IDF weights and classifier were refitted on all labeled data. DistilBERT was trained for one epoch on a Colab Tesla T4. The full-data fit has no separate cross-validation or holdout score.

## Evaluation

The original model comparisons use the same 40,000 development documents and three group-aware CV folds. A separate fixed holdout contains 54,469 documents. Detected similar-text groups never cross folds or the holdout boundary. Hyperparameters and ensemble weights are selected using development CV, not holdout labels. Selected CV scores are model-selection estimates, not nested unbiased estimates.

Related texts are approximated using 64-permutation MinHash LSH candidates followed by exact word-trigram Jaccard verification at 0.65. This can miss substantial paraphrases and some candidates. Rare language/domain/class combinations cannot be balanced perfectly across folds.

TF-IDF is fitted independently within each training fold and reads full documents. DistilBERT is fine-tuned for one fixed epoch on the first 254 content tokens plus special tokens. Chunking uses the same weights and averages probabilities across unique start/middle/end windows; it does not retrain on chunks or cover every long text completely.

## Reproduce

Python is managed with `uv`. The Kaggle CLI and Google Colab CLI require the user's own authentication; no credentials are included. Use a CUDA Colab runtime for the encoder. The measured run used a Tesla T4. Colab's CUDA-enabled PyTorch is retained; its exact version and the model revision are recorded in `results/`.

Prepare the fixed data and baseline locally:

```bash
uv sync --locked
kaggle competitions download -c ai-generated-text-detection-hw3-ucu-ai-2026 -p data
unzip -n data/ai-generated-text-detection-hw3-ucu-ai-2026.zip -d data
uv run --locked python eda.py
uv run --locked python prepare.py
uv run --locked python baseline.py
uv run --locked python prepare_colab.py
```

Run the classical and encoder experiments in Colab:

```bash
colab new -s hw3-nlp --gpu T4
colab install -s hw3-nlp -r requirements-colab.txt
uv run --locked python transfer_colab.py
uv run --locked python orchestrate_colab.py
uv run --locked python build_final_deliverables.py
```

The transfer helper uploads data chunks, frozen metadata, scripts, and the recorded pretrained-model revision, then verifies the transfer checksum. The local orchestrator retrieves incremental prediction checkpoints, finalizes the experiment, downloads and validates the final archive, and only then stops the Colab runtime. Do not submit to Kaggle again when reproducing results unless a new submission is intended. `submit_kaggle.py` submits up to two validated candidates and records their scores.

Execute the results notebook after building it. `verify_final_results.py` checks executed cells, recomputes saved CV/holdout metrics, checks group isolation and ID ordering, verifies submission checksums, and checks exact linear attributions. Notebook execution requires Jupyter/nbclient; use a Jupyter environment with the project's packages installed.

## Deliverables

- `REPORT_UK.pdf`: final report (added separately).
- `ai-generated-text-detection-hw3-ucu-ai-2026.ipynb`: executed analysis and results.
- `results/final_metrics.json`: model comparisons, selection, class metrics, and subgroup results.
- `results/transformer_config.json`: model revision and training configuration.
- `results/character_attributions.json`: exact local logit attributions for the character model.
- `results/final_verification.json`: artifact and metric verification.
- `submissions/` and `SUBMISSIONS.md`: validated prediction files and submission history.

Data and checkpoints are excluded from the reproducibility archive. Regenerate them from the competition files. Cached CV predictions in the archive are valid only for the frozen data and configuration; remove prediction/model/token caches before changing an experiment. Generated figures and aggregate results are retained.

## References

- [Competition and dataset description](https://www.kaggle.com/competitions/ai-generated-text-detection-hw3-ucu-ai-2026)
- [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Stratified group folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- [MinHash and LSH](https://ekzhu.com/datasketch/documentation.html)

The dataset's stated license is CC BY-NC-SA 4.0, for course use. No external training data or proprietary-model labeling/features are used.
