"""Tune a bounded classical baseline using development CV, then evaluate holdout."""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
MAX_TRAIN_ROWS = 40000
LABELS = ["human", "ai", "fusion"]
C_VALUES = [1.0, 4.0]


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=80000, min_df=3,
                           sublinear_tf=True, dtype=np.float32)


def classifier(c):
    return LogisticRegression(C=c, solver="saga", max_iter=200, tol=0.002,
                              random_state=SEED)


def main():
    started = time.monotonic()
    df = pd.read_csv(ROOT / "data" / "train.csv")
    manifest = pd.read_csv(ROOT / "data" / "split_manifest.csv")
    assert df.id.tolist() == manifest.id.tolist()
    development = manifest.loc[manifest.fold != 0]
    sizes = development.groupby("group").size()
    rng = np.random.default_rng(SEED)
    selected_groups, count = [], 0
    for group in rng.permutation(sizes.index):
        if count + sizes[group] <= MAX_TRAIN_ROWS:
            selected_groups.append(group)
            count += int(sizes[group])
        if count == MAX_TRAIN_ROWS:
            break
    selected = manifest.group.isin(selected_groups)
    dev = df.loc[selected].reset_index(drop=True)
    groups = manifest.loc[selected, "group"].to_numpy()
    held = df.loc[manifest.fold == 0].reset_index(drop=True)
    manifest["baseline_training"] = selected
    manifest.to_csv(ROOT / "data" / "experiment_manifest.csv", index=False)
    print(f"Development training: {len(dev)}; fixed holdout: {len(held)}", flush=True)
    strata = dev.label + "|" + dev.lang + "|" + dev.domain
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    records = []
    out_of_fold = {c: np.full(len(dev), "", dtype=object) for c in C_VALUES}
    for fold, (tr, va) in enumerate(cv.split(dev, strata, groups)):
        assert not (set(groups[tr]) & set(groups[va]))
        vec = vectorizer()
        print(f"CV fold {fold + 1}: fitting TF-IDF on {len(tr)} rows", flush=True)
        xtr = vec.fit_transform(dev.text.iloc[tr])
        xva = vec.transform(dev.text.iloc[va])
        for c in C_VALUES:
            model = classifier(c)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(xtr, dev.label.iloc[tr])
            predicted = model.predict(xva)
            out_of_fold[c][va] = predicted
            score = f1_score(dev.label.iloc[va], predicted, labels=LABELS, average="macro")
            records.append({"fold": fold + 1, "C": c, "macro_f1": score,
                            "iterations": int(model.n_iter_.max()),
                            "convergence_warning": any(isinstance(w.message, ConvergenceWarning) for w in caught)})
            print(f"CV fold {fold + 1}, C={c}: macro-F1={score:.5f}", flush=True)
            pd.DataFrame(records).to_csv(ROOT / "results" / "baseline_cv.csv", index=False)
    results = pd.DataFrame(records)
    means = results.groupby("C").macro_f1.mean()
    best_c = float(means.idxmax())
    print(f"Selected C={best_c}; fitting on {len(dev)} development rows", flush=True)
    pipeline = Pipeline([("tfidf", vectorizer()), ("classifier", classifier(best_c))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(dev.text, dev.label)
    predicted = pipeline.predict(held.text)
    report = classification_report(held.label, predicted, labels=LABELS, output_dict=True)
    pd.DataFrame({"id": held.id, "true_label": held.label, "prediction": predicted,
                  "lang": held.lang, "domain": held.domain}).to_csv(
                      ROOT / "data" / "baseline_holdout_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(held.label, predicted, labels=LABELS,
                                            normalize="true", ax=ax, cmap="Blues", values_format=".2f")
    ax.set_title("TF-IDF + logistic regression: fixed holdout")
    fig.savefig(ROOT / "results" / "baseline_confusion.png", dpi=160)
    plt.close(fig)
    slices = []
    for column in ("lang", "domain"):
        for value in sorted(held[column].unique()):
            mask = held[column].eq(value).to_numpy()
            slices.append({"slice": column, "value": value, "rows": int(mask.sum()),
                "macro_f1": f1_score(held.label[mask], predicted[mask], labels=LABELS,
                                      average="macro", zero_division=0)})
    pd.DataFrame(slices).to_csv(ROOT / "results" / "baseline_slices.csv", index=False)
    summary = {
        "model": "Word unigram/bigram TF-IDF + logistic regression",
        "seed": SEED, "training_rows": len(dev), "holdout_rows": len(held),
        "tfidf": {"ngram_range": [1, 2], "max_features": 80000, "min_df": 3, "sublinear_tf": True},
        "selected_C": best_c, "selection": "Mean macro-F1 across 3 group-aware development folds",
        "cv": {str(c): {"mean_macro_f1": float(means[c]),
                         "std_macro_f1": float(results.loc[results.C == c, "macro_f1"].std(ddof=1)),
                         "pooled_oof_report": classification_report(dev.label, out_of_fold[c], labels=LABELS, output_dict=True)}
               for c in C_VALUES},
        "holdout_report": report,
        "holdout_is_cv": False,
        "final_convergence_warning": any(isinstance(w.message, ConvergenceWarning) for w in caught),
        "elapsed_seconds": time.monotonic() - started,
        "split_manifest_sha256": hashlib.sha256((ROOT / "data" / "split_manifest.csv").read_bytes()).hexdigest(),
    }
    (ROOT / "results" / "baseline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipeline, ROOT / "models" / "word_tfidf_logreg.joblib", compress=3)
    print(json.dumps({"cv_means": means.to_dict(), "holdout_macro_f1": report["macro avg"]["f1-score"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
