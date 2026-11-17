"""Fine-tune multilingual DistilBERT and compare head versus document chunks."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import gc
import json
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ROOT = Path(os.environ.get("HW3_ROOT", Path(__file__).resolve().parent))
MODEL = "distilbert/distilbert-base-multilingual-cased"
LABELS = np.array(["human", "ai", "fusion"])
LENGTH = 256
BATCH = 16
ACCUMULATION = 2
EPOCHS = 1
LEARNING_RATE = 3e-5
DEVICE = os.environ.get("HW3_DEVICE", "cuda")


def gpu_name():
    if DEVICE == "cuda": return torch.cuda.get_device_name(0)
    output=subprocess.run(["system_profiler","SPHardwareDataType"],capture_output=True,text=True,timeout=15).stdout
    chip=re.search(r"Chip:\s*(.+)",output)
    return (chip.group(1) if chip else "Apple GPU")+" (MPS)"


def memory_used():
    return torch.cuda.max_memory_allocated() if DEVICE=="cuda" else torch.mps.current_allocated_memory()


def clear_memory():
    gc.collect()
    if DEVICE=="cuda": torch.cuda.empty_cache()
    else: torch.mps.empty_cache()


def write(name, value):
    path = ROOT / "results" / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def score(y, p):
    return float(f1_score(y, LABELS[p.argmax(1)], average="macro"))


def encode(frame, tokenizer, name):
    path = ROOT / f"data/{name}_tokens.npy"
    count_path = ROOT / f"data/{name}_chunk_counts.npy"
    if path.exists() and count_path.exists():
        return np.load(path, mmap_mode="r"), np.load(count_path)
    arrays = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(len(frame), 3, LENGTH))
    arrays[:] = tokenizer.pad_token_id
    counts = np.empty(len(frame), dtype=np.int8)
    lengths = np.empty(len(frame), dtype=np.int32)
    width = LENGTH - 2
    for first in range(0, len(frame), 256):
        batch = tokenizer(frame.text.iloc[first:first+256].tolist(), add_special_tokens=False,
                          truncation=False, return_attention_mask=False, return_token_type_ids=False,
                          verbose=False)["input_ids"]
        for offset, tokens in enumerate(batch):
            i = first+offset
            lengths[i] = len(tokens)
            starts = sorted({0, max(0, (len(tokens)-width)//2), max(0, len(tokens)-width)})
            counts[i] = len(starts)
            for k, start in enumerate(starts):
                sequence = [tokenizer.cls_token_id] + tokens[start:start+width] + [tokenizer.sep_token_id]
                arrays[i, k, :len(sequence)] = sequence
        if first % 10240 == 0:
            print(f"Tokenization {name}: {first}/{len(frame)}", flush=True)
    arrays.flush()
    np.save(count_path, counts)
    np.save(ROOT / f"data/{name}_token_lengths.npy", lengths)
    coverage = np.minimum(lengths, width*3) / np.maximum(lengths, 1)
    # Exact union coverage, accounting for overlap of start/middle/end windows.
    exact = []
    for n in lengths:
        starts = sorted({0, max(0, (int(n)-width)//2), max(0, int(n)-width)})
        covered, end = 0, 0
        for start in starts:
            stop = min(int(n), start+width)
            covered += max(0, stop-max(end, start))
            end = max(end, stop)
        exact.append(covered/max(1, int(n)))
    write(f"{name}_tokenization.json", {
        "rows": len(frame), "sequence_length": LENGTH, "content_tokens_per_chunk": width,
        "median_full_tokens": float(np.median(lengths)), "max_full_tokens": int(lengths.max()),
        "fraction_exceeding_one_chunk": float((lengths > width).mean()),
        "mean_head_coverage": float((np.minimum(lengths, width)/np.maximum(lengths,1)).mean()),
        "mean_three_chunk_coverage": float(np.mean(exact)),
        "limitations": "Up to three non-overlapping/overlapping windows; very long texts are not fully covered."})
    return arrays, counts


class Heads(Dataset):
    def __init__(self, tokens, labels, indices):
        self.tokens, self.labels, self.indices = tokens, labels, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return torch.from_numpy(self.tokens[i, 0].copy()).long(), int(self.labels[i])


class Chunks(Dataset):
    def __init__(self, tokens, counts, indices):
        self.tokens = tokens
        self.mapping = [(int(i), k) for i in indices for k in range(int(counts[i]))]

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, item):
        i, k = self.mapping[item]
        return torch.from_numpy(self.tokens[i,k].copy()).long(), i, k


def train(tokens, labels, indices, tokenizer, revision, name, trial=False):
    target = ROOT / "models" / name
    if (target / "complete.json").exists():
        return AutoModelForSequenceClassification.from_pretrained(target).to(DEVICE)
    value = 42 + (int(name[-1]) if name.startswith("fold") else 3)
    seed(value)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=revision, num_labels=3,
        id2label=dict(enumerate(LABELS.tolist())), label2id={k:i for i,k in enumerate(LABELS)},
        attn_implementation="sdpa").to(DEVICE)
    loader = DataLoader(Heads(tokens, labels, indices), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(value), pin_memory=DEVICE=="cuda")
    groups = [
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight", "layer_norm.weight"))], "weight_decay": .01},
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight", "layer_norm.weight"))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    steps = (len(loader)+ACCUMULATION-1)//ACCUMULATION*EPOCHS
    schedule = get_linear_schedule_with_warmup(optimizer, max(1,int(.06*steps)), max(1,steps))
    scaler = torch.amp.GradScaler(DEVICE,init_scale=128 if DEVICE=="mps" else 65536)
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for epoch in range(EPOCHS):
        for step, (ids, y) in enumerate(loader):
            ids, y = ids.to(DEVICE,non_blocking=DEVICE=="cuda"), y.to(DEVICE,non_blocking=DEVICE=="cuda")
            group_size = min(ACCUMULATION, len(loader)-(step//ACCUMULATION)*ACCUMULATION)
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                output = model(input_ids=ids, attention_mask=ids.ne(tokenizer.pad_token_id), labels=y)
                loss = output.loss/group_size
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            loss_sum += output.loss.item()
            if (step+1) % ACCUMULATION == 0 or step+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            if step % 100 == 0:
                progress = {"stage":name, "epoch":epoch+1, "batch":step+1, "batches":len(loader),
                            "mean_loss":loss_sum/(step+1), "elapsed_seconds":time.monotonic()-started,
                            "gpu_allocated_bytes":memory_used()}
                write("transformer_training_progress.json",progress)
                print(json.dumps(progress),flush=True)
    if DEVICE=="cuda": torch.cuda.synchronize()
    else: torch.mps.synchronize()
    if trial:
        write("transformer_smoke.json", {"passed":True,"documents":len(indices),
              "elapsed_seconds":time.monotonic()-started,"loss":loss_sum/max(1,len(loader)),
              "gpu":gpu_name(),"gpu_allocated_bytes":memory_used()})
    else:
        model.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        (target/"complete.json").write_text(json.dumps({"training_rows":len(indices), "epochs":EPOCHS,
             "model_revision":revision, "elapsed_seconds":time.monotonic()-started, "seed":value}))
    del optimizer, schedule, scaler, loader, groups
    clear_memory()
    return model


@torch.inference_mode()
def infer(model, tokens, counts, tokenizer, indices, name):
    indices = np.asarray(indices)
    dataset = Chunks(tokens, counts, indices)
    loader = DataLoader(dataset,batch_size=64,shuffle=False,pin_memory=DEVICE=="cuda")
    predictions = np.zeros((len(tokens),3,3),dtype=np.float32)
    model.eval()
    for step, (ids, rows, chunks) in enumerate(loader):
        ids = ids.to(DEVICE,non_blocking=DEVICE=="cuda")
        with torch.amp.autocast(DEVICE,dtype=torch.float16):
            logits = model(input_ids=ids,attention_mask=ids.ne(tokenizer.pad_token_id)).logits
        p = logits.float().softmax(-1).cpu().numpy()
        predictions[rows.numpy(),chunks.numpy()] = p
        if step % 200 == 0:
            print(f"Inference {name}: {step+1}/{len(loader)} batches",flush=True)
    head = predictions[indices,0]
    chunks = predictions[indices].sum(axis=1)/counts[indices,None]
    assert np.allclose(head.sum(1),1,atol=1e-5)
    assert np.allclose(chunks.sum(1),1,atol=1e-5)
    return head,chunks


def main():
    assert (DEVICE=="cuda" and torch.cuda.is_available()) or (DEVICE=="mps" and torch.backends.mps.is_available())
    torch.set_num_threads(1)
    for name in ("data","results","models","predictions"):
        (ROOT/name).mkdir(exist_ok=True)
    started=time.monotonic()
    revision_path=ROOT/"results/transformer_config.json"
    revision=json.loads(revision_path.read_text())["revision"] if revision_path.exists() else HfApi().model_info(MODEL).sha
    config={"model":MODEL,"revision":revision,"epochs":EPOCHS,"sequence_length":LENGTH,
            "batch_size":BATCH,"gradient_accumulation":ACCUMULATION,"learning_rate":LEARNING_RATE,
            "class_order":LABELS.tolist(),"gpu":gpu_name(),"torch":torch.__version__,"device":DEVICE,
            "initial_grad_scale":128 if DEVICE=="mps" else 65536,
            "training_text_view":"First 254 content tokens plus special tokens",
            "chunking":"Mean class probabilities over unique start/middle/end windows, 254 content tokens each",
            "compute_limit":"One fixed epoch per CV fold and final model; no holdout-based tuning"}
    write("transformer_config.json",config)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    dev=pd.read_csv(ROOT/"data/development.csv")
    labels=dev.label.map({k:i for i,k in enumerate(LABELS)}).to_numpy()
    tokens,counts=encode(dev,tokenizer,"development")
    if not (ROOT/"results/transformer_smoke.json").exists():
        model=train(tokens,labels,np.arange(32),tokenizer,revision,"smoke",trial=True)
        p=infer(model,tokens,counts,tokenizer,np.arange(32),"smoke")
        assert all(np.isfinite(x).all() for x in p)
        del model
        clear_memory()
        print("GPU SMOKE TEST PASSED",flush=True)
    oof={kind:np.zeros((len(dev),3),dtype=np.float32) for kind in ("head","chunks")}
    records=[]
    for fold in range(3):
        tr=np.flatnonzero(dev.cv_fold.ne(fold)); va=np.flatnonzero(dev.cv_fold.eq(fold))
        assert not set(dev.group.iloc[tr]) & set(dev.group.iloc[va])
        saved=ROOT/f"predictions/transformer_fold{fold}.npz"
        if saved.exists():
            cache=np.load(saved)
            probabilities=[cache["head"],cache["chunks"]]
        else:
            model=train(tokens,labels,tr,tokenizer,revision,f"fold{fold}")
            probabilities=infer(model,tokens,counts,tokenizer,va,f"fold{fold}")
            np.savez_compressed(saved,head=probabilities[0],chunks=probabilities[1],ids=dev.id.iloc[va].to_numpy())
            del model
            clear_memory()
        for kind,p in zip(("head","chunks"),probabilities):
            oof[kind][va]=p
            records.append({"model":kind,"fold":fold,"macro_f1":score(dev.label.iloc[va],p)})
        write("transformer_cv_progress.json",records)
        print("TRANSFORMER FOLD COMPLETE "+json.dumps(records[-2:]),flush=True)
    final=train(tokens,labels,np.arange(len(dev)),tokenizer,revision,"final")
    record={"config":config,"folds":records,"models":{}}
    holdout=pd.read_csv(ROOT/"data/holdout.csv")
    held_tokens,held_counts=encode(holdout,tokenizer,"holdout")
    held_prob=infer(final,held_tokens,held_counts,tokenizer,np.arange(len(holdout)),"holdout")
    test=pd.read_csv(ROOT/"data/test.csv")
    test_tokens,test_counts=encode(test,tokenizer,"test")
    test_prob=infer(final,test_tokens,test_counts,tokenizer,np.arange(len(test)),"test")
    for i,kind in enumerate(("head","chunks")):
        np.savez_compressed(ROOT/f"predictions/transformer_{kind}.npz",oof=oof[kind],holdout=held_prob[i],test=test_prob[i],
             development_ids=dev.id.to_numpy(),holdout_ids=holdout.id.to_numpy(),test_ids=test.id.to_numpy())
        values=[r["macro_f1"] for r in records if r["model"]==kind]
        record["models"][kind]={"cv_mean":float(np.mean(values)),"cv_std":float(np.std(values,ddof=1)),
             "holdout_macro_f1":score(holdout.label,held_prob[i]),
             "holdout_report":classification_report(holdout.label,LABELS[held_prob[i].argmax(1)],labels=LABELS,output_dict=True),
             "holdout_confusion":confusion_matrix(holdout.label,LABELS[held_prob[i].argmax(1)],labels=LABELS).tolist()}
    record["elapsed_seconds"]=time.monotonic()-started
    write("transformer_metrics.json",record)
    print("TRANSFORMER COMPLETE "+json.dumps({k:v["cv_mean"] for k,v in record["models"].items()}),flush=True)


if __name__=="__main__":
    main()
