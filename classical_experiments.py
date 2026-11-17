"""Compare word and character representations on the frozen Colab bundle."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import gc
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ.get("HW3_ROOT", Path(__file__).resolve().parent))
LABELS = np.array(["human", "ai", "fusion"])


def write(name, content):
    (ROOT / "results" / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def score(y, probability):
    return float(f1_score(y, LABELS[probability.argmax(axis=1)], average="macro"))


def ordered(model, matrix):
    return model.predict_proba(matrix)[:, [list(model.classes_).index(k) for k in LABELS]].astype(np.float32)


def predict_batches(pipe, texts):
    return np.concatenate([ordered(pipe, texts.iloc[i:i+1000]) for i in range(0, len(texts), 1000)])


def vec(kind):
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), max_features=80000, min_df=3,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=100000,
                           min_df=5, lowercase=False, sublinear_tf=True, dtype=np.float32)


def clf():
    return LogisticRegression(C=4.0, solver="saga", max_iter=300, tol=0.002, random_state=42)


def attribution(pipe, held):
    """Exact additive logit differences for the character model, not causal claims."""
    vocabulary = pipe.named_steps["tfidf"].get_feature_names_out()
    model = pipe.named_steps["classifier"]
    examples = []
    candidates = held.groupby("label", sort=True).head(5)
    for row in candidates.itertuples():
        x = pipe.named_steps["tfidf"].transform([row.text])
        logits = model.decision_function(x)[0]
        ranking = np.argsort(logits)[::-1]
        a, b = ranking[:2]
        values = x.data * (model.coef_[a, x.indices] - model.coef_[b, x.indices])
        bias = float(model.intercept_[a] - model.intercept_[b])
        assert np.isclose(values.sum() + bias, logits[a] - logits[b], rtol=1e-4, atol=1e-4)
        selected = np.argsort(np.abs(values))[-12:]
        examples.append({"id": row.id, "true_label": row.label,
            "prediction": str(model.classes_[a]), "runner_up": str(model.classes_[b]),
            "logit_difference": float(logits[a]-logits[b]), "bias_difference": bias,
            "sum_feature_contributions": float(values.sum()),
            "top_features": [{"feature": str(vocabulary[x.indices[j]]), "contribution": float(values[j])}
                             for j in selected]})
    write("character_attributions.json", {"method": "Exact TF-IDF contribution to predicted-minus-runner-up logit",
          "scope": "Character model only; correlated lexical evidence, not causal authorship evidence",
          "examples": examples})
    ex = examples[0]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    features = ex["top_features"]
    values = [f["contribution"] for f in features]
    ax.barh([repr(f["feature"]) for f in features], values,
            color=["#386789" if x >= 0 else "#c67c45" for x in values])
    ax.axvline(0, color="#53646b", lw=0.8)
    ax.set(title=f"Character model: {ex['prediction']} versus {ex['runner_up']}",
           xlabel="Exact contribution to logit difference")
    fig.savefig(ROOT / "results/character_attribution.png", dpi=160)
    plt.close(fig)


def main():
    started = time.monotonic()
    for folder in ("results", "models", "predictions"):
        (ROOT / folder).mkdir(exist_ok=True)
    dev = pd.read_csv(ROOT / "data/development.csv")
    held = pd.read_csv(ROOT / "data/holdout.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    record = {"class_order": LABELS.tolist(), "training_rows": len(dev), "holdout_rows": len(held),
              "models": {}, "folds": []}
    all_prob = {}
    for kind in ("word", "character"):
        oof = np.zeros((len(dev), 3), dtype=np.float32)
        for fold in range(3):
            cached = ROOT / f"predictions/{kind}_fold{fold}.npz"
            tr = np.flatnonzero(dev.cv_fold.ne(fold))
            va = np.flatnonzero(dev.cv_fold.eq(fold))
            if cached.exists():
                oof[va] = np.load(cached)["probabilities"]
            else:
                print(f"{kind} fold {fold+1}: fitting {len(tr)} documents", flush=True)
                pipe = Pipeline([("tfidf", vec(kind)), ("classifier", clf())])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipe.fit(dev.text.iloc[tr], dev.label.iloc[tr])
                oof[va] = predict_batches(pipe, dev.text.iloc[va])
                np.savez_compressed(cached, probabilities=oof[va], ids=dev.id.iloc[va].to_numpy())
                if any(isinstance(w.message, ConvergenceWarning) for w in caught):
                    raise RuntimeError(f"{kind} fold {fold}: optimizer did not converge")
                del pipe
                gc.collect()
            value = score(dev.label.iloc[va], oof[va])
            record["folds"].append({"model": kind, "fold": fold, "macro_f1": value})
            write("classical_progress.json", record)
            print(f"{kind} fold {fold+1}: macro-F1={value:.5f}", flush=True)
        print(f"{kind}: final 40000-document fit", flush=True)
        model_path = ROOT / f"models/{kind}.joblib"
        if model_path.exists():
            pipe = joblib.load(model_path)
        else:
            pipe = Pipeline([("tfidf", vec(kind)), ("classifier", clf())]).fit(dev.text, dev.label)
            joblib.dump(pipe, model_path, compress=3)
        held_probability = predict_batches(pipe, held.text)
        test_probability = predict_batches(pipe, test.text)
        np.savez_compressed(ROOT / f"predictions/{kind}.npz", oof=oof, holdout=held_probability,
                            test=test_probability, development_ids=dev.id.to_numpy(),
                            holdout_ids=held.id.to_numpy(), test_ids=test.id.to_numpy())
        all_prob[kind] = (oof, held_probability, test_probability)
        fold_scores = [x["macro_f1"] for x in record["folds"] if x["model"] == kind]
        record["models"][kind] = {"cv_mean": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores, ddof=1)),
             "holdout_macro_f1": score(held.label, held_probability),
             "holdout_report": classification_report(held.label, LABELS[held_probability.argmax(1)], labels=LABELS, output_dict=True),
             "holdout_confusion": confusion_matrix(held.label, LABELS[held_probability.argmax(1)], labels=LABELS).tolist()}
        if kind == "character":
            attribution(pipe, held)
        print(f"{kind} complete: CV={np.mean(fold_scores):.5f}", flush=True)
        write("classical_metrics.json", record)
        del pipe
        gc.collect()
    grid = []
    for alpha in [0., .25, .5, .75, 1.]:
        p = (1-alpha) * all_prob["word"][0] + alpha * all_prob["character"][0]
        fold_scores = [score(dev.label[dev.cv_fold.eq(f)], p[dev.cv_fold.eq(f)]) for f in range(3)]
        grid.append({"character_weight": alpha, "cv_mean": float(np.mean(fold_scores)),
                     "cv_std": float(np.std(fold_scores, ddof=1)), "fold_scores": fold_scores})
    best = max(grid, key=lambda x: x["cv_mean"])
    a = best["character_weight"]
    probabilities = [(1-a)*x+a*y for x,y in zip(all_prob["word"], all_prob["character"])]
    np.savez_compressed(ROOT / "predictions/classical_blend.npz", oof=probabilities[0], holdout=probabilities[1],
                        test=probabilities[2], development_ids=dev.id.to_numpy(), holdout_ids=held.id.to_numpy(), test_ids=test.id.to_numpy())
    record["blend_grid"] = grid
    record["models"]["classical_blend"] = {**best, "holdout_macro_f1": score(held.label, probabilities[1]),
        "holdout_report": classification_report(held.label, LABELS[probabilities[1].argmax(1)], labels=LABELS, output_dict=True),
        "holdout_confusion": confusion_matrix(held.label, LABELS[probabilities[1].argmax(1)], labels=LABELS).tolist()}
    record["elapsed_seconds"] = time.monotonic()-started
    write("classical_metrics.json", record)
    print("CLASSICAL COMPLETE " + json.dumps({k: v.get("cv_mean") for k,v in record["models"].items()}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
