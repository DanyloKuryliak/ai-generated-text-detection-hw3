"""Recompute CV/holdout scores and verify IDs, group isolation, and submissions."""
import hashlib
import json
from pathlib import Path
import zipfile

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
LABELS=np.array(["human","ai","fusion"])


def score(y,p):
    return float(f1_score(y,LABELS[p.argmax(1)],average="macro"))


def main():
    m=json.loads((ROOT/"results/final_metrics.json").read_text())
    metadata=json.loads((ROOT/"results/colab_bundle_metadata.json").read_text())
    with zipfile.ZipFile(ROOT/"data/colab_bundle.zip") as archive:
        dev=pd.read_csv(archive.open("development.csv"),usecols=["id","label","group","cv_fold"])
        held=pd.read_csv(archive.open("holdout.csv"),usecols=["id","label"])
        sample=pd.read_csv(archive.open("sample_submission.csv"))
    manifest=pd.read_csv(ROOT/"data/experiment_manifest.csv")
    assert dev.id.tolist()==manifest.loc[manifest.baseline_training,"id"].tolist()
    assert held.id.tolist()==manifest.loc[manifest.fold==0,"id"].tolist()
    assert not set(dev.group) & set(manifest.loc[manifest.fold==0,"group"])
    assert dev.groupby("group").cv_fold.nunique().max()==1
    assert hashlib.sha256("\n".join(dev.id).encode()).hexdigest()==metadata["development_ids_sha256"]
    for name,row in m["models"].items():
        file="final_selected" if name=="ensemble" else name
        with np.load(ROOT/f"predictions/{file}.npz",allow_pickle=True) as saved:
            assert dev.id.tolist()==saved["development_ids"].tolist()
            assert held.id.tolist()==saved["holdout_ids"].tolist()
            assert sample.id.tolist()==saved["test_ids"].tolist()
            oof=saved["oof"]; hold=saved["holdout"]
            values=[score(dev.label[dev.cv_fold.eq(f)],oof[dev.cv_fold.eq(f)]) for f in range(3)]
            assert np.allclose(values,row["cv_fold_scores"])
            assert np.isclose(np.mean(values),row["cv_mean"])
            assert np.isclose(np.std(values,ddof=1),row["cv_std"])
            assert np.isclose(score(held.label,hold),row["holdout_macro_f1"])
            for part in ["oof","holdout","test"]:
                assert np.isfinite(saved[part]).all()
                assert np.allclose(saved[part].sum(1),1,atol=1e-5)
    for item in m["submission_files"]:
        path=ROOT/"submissions"/item["file"]
        df=pd.read_csv(path)
        assert df.columns.tolist()==["id","label"]
        assert len(df)==25000 and df.id.is_unique
        assert df.id.tolist()==sample.id.tolist()
        assert df.label.isin(LABELS).all()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
    with np.load(ROOT/"predictions/final_selected.npz",allow_pickle=True) as saved:
        final=pd.read_csv(ROOT/"submissions"/m["submission_files"][0]["file"])
        assert final.label.tolist()==LABELS[saved["test"].argmax(1)].tolist()
    explanations=json.loads((ROOT/"results/character_attributions.json").read_text())
    for ex in explanations["examples"]:
        assert np.isclose(ex["bias_difference"]+ex["sum_feature_contributions"],ex["logit_difference"],rtol=1e-4,atol=1e-4)
    nb=nbformat.read(ROOT/"ai-generated-text-detection-hw3-ucu-ai-2026.ipynb",as_version=4)
    nbformat.validate(nb)
    assert all(cell.execution_count is not None for cell in nb.cells if cell.cell_type=="code")
    assert not any(out.output_type=="error" for cell in nb.cells for out in cell.get("outputs",[]))
    result={"status":"passed","model_scores_recomputed":len(m["models"]),
            "training_rows":len(dev),"holdout_rows":len(held),"test_rows":len(sample),
            "group_overlap":0,"cv_isolation_checked":True,"submission_ids_order_and_checksums_checked":True,
            "exact_linear_attributions_checked":len(explanations["examples"]),"executed_notebook_has_errors":False}
    (ROOT/"results/final_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
