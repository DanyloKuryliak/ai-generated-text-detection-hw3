"""Create figures and a fixed split with approximate related-text groups."""
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
SEED = 42
PERMUTATIONS = 64
THRESHOLD = 0.65
LABELS = ["human", "ai", "fusion"]


def shingles(text):
    words = re.findall(r"\w+", text.casefold())
    return {" ".join(words[i:i + 3]).encode() for i in range(len(words) - 2)}


def signature(text):
    sketch = MinHash(num_perm=PERMUTATIONS, seed=SEED, scheme="affine32")
    sketch.update_batch(list(shingles(text)))
    return sketch.hashvalues


def save_json(name, obj):
    (ROOT / "results" / name).write_text(json.dumps(obj, indent=2) + "\n")


def make_figures(df):
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ["#406b8a", "#cc8250", "#639e86"]
    df.label.value_counts().reindex(LABELS).plot.bar(ax=axes[0, 0], color=colors, rot=0)
    axes[0, 0].set(title="Training class balance", ylabel="Documents", xlabel="")
    pd.crosstab(df.lang, df.label, normalize="index").reindex(columns=LABELS).plot.bar(
        ax=axes[0, 1], stacked=True, color=colors, rot=0)
    axes[0, 1].set(title="Class proportions by language", ylabel="Proportion", xlabel="")
    pd.crosstab(df.domain, df.label, normalize="index").reindex(columns=LABELS).plot.barh(
        ax=axes[1, 0], stacked=True, color=colors, legend=False)
    axes[1, 0].set(title="Class proportions by domain", xlabel="Proportion", ylabel="")
    for label, color in zip(LABELS, colors):
        axes[1, 1].hist(df.loc[df.label == label, "word_count"], bins=50,
                        histtype="step", density=True, label=label, color=color)
    axes[1, 1].set(title="Text length by class", xlabel="Whitespace-separated words",
                   ylabel="Density")
    axes[1, 1].legend()
    fig.savefig(ROOT / "results" / "eda.png")
    plt.close(fig)
    df.groupby(["lang", "domain", "label"]).size().rename("count").to_csv(
        ROOT / "results" / "class_balance.csv")
    df.groupby("label").word_count.describe().to_csv(ROOT / "results" / "lengths_by_class.csv")
    audit = json.loads((ROOT / "results" / "data_audit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, column in zip(axes, ["lang", "domain"]):
        proportions = pd.DataFrame({
            part: pd.Series(audit[part]["counts"][column]) / audit[part]["rows"]
            for part in ["train", "test"]
        }).fillna(0)
        proportions.plot.barh(ax=ax, color=["#406b8a", "#cc8250"])
        ax.set(xlabel="Proportion", ylabel="", title=f"Train/test distribution: {column}")
    fig.savefig(ROOT / "results" / "train_test_shift.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    df["word_count"] = df.text.map(lambda text: len(text.split()))
    make_figures(df)
    print("EDA figures saved.", flush=True)
    cache = ROOT / "data" / "minhash.npy"
    if cache.exists():
        signatures = np.load(cache)
        if signatures.shape != (len(df), PERMUTATIONS):
            raise ValueError("Signature cache does not match the dataset")
    else:
        signatures = np.empty((len(df), PERMUTATIONS), dtype=np.uint64)
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i, values in enumerate(pool.map(signature, df.text, chunksize=128)):
                signatures[i] = values
                if (i + 1) % 10000 == 0:
                    print(f"Signatures: {i + 1}/{len(df)}", flush=True)
        np.save(cache, signatures)
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    @lru_cache(maxsize=256)
    def tokens(i):
        return shingles(df.at[i, "text"])

    index = MinHashLSH(threshold=0.5, num_perm=PERMUTATIONS)
    comparisons = edges = cross_domain_edges = cross_language_edges = 0
    # Search across all languages/domains: metadata must not hide duplicate families.
    for i in range(len(df)):
        sketch = MinHash(num_perm=PERMUTATIONS, seed=SEED, hashvalues=signatures[i], scheme="affine32")
        candidates = sorted(index.query(sketch))
        for j in candidates:
            a, b = tokens(i), tokens(j)
            comparisons += 1
            similarity = len(a & b) / max(1, len(a | b))
            if similarity >= THRESHOLD:
                parent[find(i)] = find(j)
                edges += 1
                cross_domain_edges += int(df.at[i, "domain"] != df.at[j, "domain"])
                cross_language_edges += int(df.at[i, "lang"] != df.at[j, "lang"])
        index.insert(i, sketch)
        if (i + 1) % 25000 == 0:
            print(f"Grouping: {i + 1}/{len(df)}; verified pairs={edges}", flush=True)
    groups = np.array([find(i) for i in range(len(df))])
    counts = Counter(groups)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    strata = df.label + "|" + df.lang + "|" + df.domain
    folds = np.full(len(df), -1)
    for fold, (_, val) in enumerate(split.split(df, strata, groups)):
        folds[val] = fold
    assert (folds >= 0).all()
    manifest = df[["id", "label", "lang", "domain"]].copy()
    manifest["group"] = groups
    manifest["fold"] = folds
    # A group is wholly in one fold; fold 0 is held out from parameter selection.
    assert manifest.groupby("group").fold.nunique().max() == 1
    manifest.to_csv(ROOT / "data" / "split_manifest.csv", index=False)
    digest = hashlib.sha256((ROOT / "data" / "split_manifest.csv").read_bytes()).hexdigest()
    save_json("split_audit.json", {
        "seed": SEED, "folds": 5, "holdout_fold": 0, "manifest_sha256": digest,
        "grouping": "64-permutation MinHash LSH candidates; exact word-trigram Jaccard >= 0.65",
        "limitations": "Approximate lexical grouping; original source IDs are unavailable. Rewrites with little lexical overlap and some LSH misses may remain in different groups.",
        "candidate_pairs_checked": comparisons, "verified_pairs": edges,
        "cross_domain_pairs": cross_domain_edges, "cross_language_pairs": cross_language_edges,
        "groups": len(counts), "groups_with_multiple_rows": sum(x > 1 for x in counts.values()),
        "rows_in_multirow_groups": sum(x for x in counts.values() if x > 1),
        "largest_group": max(counts.values()),
        "fold_counts": manifest.groupby("fold").size().to_dict(),
        "group_overlap_between_folds": 0,
    })
    print((ROOT / "results" / "split_audit.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
