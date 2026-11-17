"""Finalize predictions and package compact Colab artifacts for local verification."""
import importlib.metadata
import json
import runpy
import zipfile
from pathlib import Path

ROOT=Path("/content/hw3")
runpy.run_path(str(ROOT/"finalize_experiments.py"),run_name="__main__")
runtime={p:importlib.metadata.version(p) for p in ["torch","transformers","numpy","pandas","scikit-learn","matplotlib"]}
gpu=ROOT/"gpu.json"
if gpu.exists(): runtime["gpu_probe"]=json.loads(gpu.read_text())
(ROOT/"results/colab_environment.json").write_text(json.dumps(runtime,indent=2)+"\n")
with zipfile.ZipFile(ROOT/"experiment_artifacts.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for folder in ("results","predictions","submissions"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file(): archive.write(path,path.relative_to(ROOT))
print("Artifacts ready: /content/hw3/experiment_artifacts.zip",flush=True)
