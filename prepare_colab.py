"""Package the fixed development/holdout/test data for a Colab run."""
import hashlib
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    frame = pd.read_csv(DATA / "train.csv")
    manifest = pd.read_csv(DATA / "experiment_manifest.csv")
    assert frame.id.tolist() == manifest.id.tolist()
    dev = frame.loc[manifest.baseline_training].reset_index(drop=True)
    dev["group"] = manifest.loc[manifest.baseline_training, "group"].to_numpy()
    dev["cv_fold"] = -1
    strata = dev.label + "|" + dev.lang + "|" + dev.domain
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    with warnings.catch_warnings(record=True):
        for fold, (tr, va) in enumerate(cv.split(dev, strata, dev.group)):
            assert not set(dev.group.iloc[tr]) & set(dev.group.iloc[va])
            dev.loc[va, "cv_fold"] = fold
    assert (dev.cv_fold >= 0).all()
    held = frame.loc[manifest.fold == 0].reset_index(drop=True)
    assert not set(dev.group) & set(manifest.loc[manifest.fold == 0, "group"])
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert test.id.tolist() == sample.id.tolist()
    metadata = {
        "seed": 42, "development_rows": len(dev), "holdout_rows": len(held),
        "test_rows": len(test), "class_order": ["human", "ai", "fusion"],
        "development_ids_sha256": hashlib.sha256("\n".join(dev.id).encode()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256((DATA / "split_manifest.csv").read_bytes()).hexdigest(),
        "cv_folds": dev.groupby("cv_fold").size().to_dict(),
        "development_group_overlap_with_holdout": 0,
        "experiment_plan": {
            "reference": "word TF-IDF (1,2), 80000 features, C=4",
            "hypothesis_1": "character TF-IDF char_wb (3,5), 100000 features, C=4",
            "encoder": "distilbert/distilbert-base-multilingual-cased, 1 epoch, 256 tokens",
            "hypothesis_2": "same fine-tuned encoder: first chunk vs mean probabilities over start/middle/end chunks",
            "selection": "Mean macro-F1 across identical three development folds; holdout excluded from selection",
            "ensemble_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
    }
    path = DATA / "colab_bundle.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, df in [("development.csv", dev), ("holdout.csv", held), ("test.csv", test), ("sample_submission.csv", sample)]:
            with archive.open(name, "w", force_zip64=True) as out:
                df.to_csv(out, index=False)
        archive.writestr("bundle_metadata.json", json.dumps(metadata, indent=2))
    metadata["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "results/colab_bundle_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    chunks = DATA / "transfer"
    chunks.mkdir(exist_ok=True)
    with path.open("rb") as handle:
        for i in range(1000):
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            (chunks / f"bundle.part{i:03}").write_bytes(chunk)
    print(json.dumps({"bytes": path.stat().st_size, "sha256": metadata["bundle_sha256"],
                      "rows": [len(dev), len(held), len(test)]}))


if __name__ == "__main__":
    main()
