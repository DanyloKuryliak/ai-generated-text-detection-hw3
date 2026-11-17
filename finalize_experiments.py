"""Select by development CV, evaluate fixed holdout, and validate submissions."""
import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT=Path(os.environ.get("HW3_ROOT",Path(__file__).resolve().parent))
LABELS=np.array(["human","ai","fusion"])


def score(y,p):
    return float(f1_score(y,LABELS[p.argmax(1)],average="macro"))


def main():
    dev=pd.read_csv(ROOT/"data/development.csv")
    held=pd.read_csv(ROOT/"data/holdout.csv")
    test=pd.read_csv(ROOT/"data/test.csv")
    sample=pd.read_csv(ROOT/"data/sample_submission.csv")
    candidates={}
    metrics={}
    for name in ["word","character","classical_blend","transformer_head","transformer_chunks"]:
        with np.load(ROOT/f"predictions/{name}.npz",allow_pickle=True) as saved:
            for key,frame in [("development",dev),("holdout",held),("test",test)]:
                assert frame.id.tolist()==saved[f"{key}_ids"].tolist()
            arrays=tuple(saved[key].copy() for key in ("oof","holdout","test"))
        for array,frame in zip(arrays,[dev,held,test]):
            assert array.shape==(len(frame),3)
            assert np.isfinite(array).all()
            assert np.allclose(array.sum(1),1,atol=1e-5)
        candidates[name]=arrays
    # All weights and model choices are selected using development OOF labels only.
    def cv_values(p):
        return [score(dev.label[dev.cv_fold.eq(f)],p[dev.cv_fold.eq(f)]) for f in range(3)]
    def cv_mean(name):
        return float(np.mean(cv_values(candidates[name][0])))
    classical=max(["word","character","classical_blend"],key=cv_mean)
    encoder=max(["transformer_head","transformer_chunks"],key=cv_mean)
    grid=[]
    for a in [.0,.25,.5,.75,1.]:
        p=(1-a)*candidates[classical][0]+a*candidates[encoder][0]
        values=cv_values(p)
        grid.append({"transformer_weight":a,"cv_mean":float(np.mean(values)),"fold_scores":values})
    best=max(grid,key=lambda row:row["cv_mean"])
    a=best["transformer_weight"]
    chosen=tuple((1-a)*x+a*y for x,y in zip(candidates[classical],candidates[encoder]))
    if a==0:
        selected=classical
    elif a==1:
        selected=encoder
    else:
        selected="ensemble"
        candidates[selected]=chosen
    np.savez_compressed(ROOT/"predictions/final_selected.npz",oof=chosen[0],holdout=chosen[1],test=chosen[2],
         development_ids=dev.id.to_numpy(),holdout_ids=held.id.to_numpy(),test_ids=test.id.to_numpy())
    for name,arrays in candidates.items():
        values=cv_values(arrays[0])
        predicted=LABELS[arrays[1].argmax(1)]
        slices=[]
        for column in ["lang","domain"]:
            for value in sorted(held[column].unique()):
                mask=held[column].eq(value).to_numpy()
                slices.append({"slice":column,"value":value,"rows":int(mask.sum()),
                               "macro_f1":score(held.label[mask],arrays[1][mask])})
        metrics[name]={"cv_mean":float(np.mean(values)),"cv_std":float(np.std(values,ddof=1)),
            "cv_fold_scores":values,"holdout_macro_f1":score(held.label,arrays[1]),
            "holdout_report":classification_report(held.label,predicted,labels=LABELS,output_dict=True),
            "holdout_confusion":confusion_matrix(held.label,predicted,labels=LABELS).tolist(),"holdout_slices":slices}
    (ROOT/"submissions").mkdir(exist_ok=True)
    files=[]
    standalone=max(["word","character","transformer_head","transformer_chunks"],key=cv_mean)
    for name in dict.fromkeys([selected,standalone]):
        arrays=chosen if name==selected else candidates[name]
        output=pd.DataFrame({"id":test.id,"label":LABELS[arrays[2].argmax(1)]}).set_index("id").loc[sample.id].reset_index()
        assert output.columns.tolist()==["id","label"]
        assert len(output)==25000 and output.id.is_unique
        assert output.id.tolist()==sample.id.tolist()
        assert output.label.isin(LABELS).all() and not output.isna().any().any()
        path=ROOT/f"submissions/{name}-cv-selected.csv"
        output.to_csv(path,index=False)
        files.append({"model":name,"file":path.name,"rows":len(output),
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"cv_macro_f1":metrics[name]["cv_mean"]})
    predicted=LABELS[chosen[1].argmax(1)]
    pd.DataFrame({"id":held.id,"true_label":held.label,"prediction":predicted,"lang":held.lang,"domain":held.domain}).to_csv(
        ROOT/"predictions/final_holdout_predictions.csv",index=False)
    record={"selected_model":selected,"selection_metric":"Mean macro-F1 across three frozen group-aware development folds",
        "selection_used_holdout":False,"training_rows":len(dev),"holdout_rows":len(held),"test_rows":len(test),
        "class_order":LABELS.tolist(),"classical_component":classical,"transformer_component":encoder,
        "transformer_weight":a,"ensemble_grid":grid,"models":metrics,"submission_files":files,
        "limitations":["CV is used for model/weight selection; selected CV scores are not nested unbiased estimates.",
                        "Holdout is a shared comparison set; original source families may remain undetected by lexical grouping.",
                        "Transformer has a fixed one-epoch budget; three 254-token windows do not cover every long document."]}
    (ROOT/"results/final_metrics.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"selected":selected,"transformer_weight":a,"scores":{k:{"cv":v["cv_mean"],"holdout":v["holdout_macro_f1"]} for k,v in metrics.items()},"submissions":files}),flush=True)


if __name__=="__main__":
    main()
