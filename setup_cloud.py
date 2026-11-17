"""Verify and unpack a CLI-transferred bundle on a CUDA-enabled Colab runtime."""
import hashlib
import json
from pathlib import Path
import zipfile

import torch

ROOT=Path("/content/hw3")
ROOT.mkdir(exist_ok=True)
assert torch.cuda.is_available(),"A CUDA GPU is required for the encoder experiment"
metadata=json.loads((ROOT/"colab_bundle_metadata.json").read_text())
bundle=ROOT/"colab_bundle.zip"
checksum=hashlib.sha256()
with bundle.open("wb") as out:
    parts=sorted(ROOT.glob("bundle.part*"))
    if not parts: raise FileNotFoundError("No uploaded bundle parts")
    for path in parts:
        content=path.read_bytes(); checksum.update(content); out.write(content)
assert checksum.hexdigest()==metadata["bundle_sha256"]
for folder in ["data","results","predictions","models"]:
    (ROOT/folder).mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    allowed={"development.csv","holdout.csv","test.csv","sample_submission.csv","bundle_metadata.json"}
    assert set(archive.namelist())==allowed
    archive.extractall(ROOT/"data")
gpu={"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
     "memory_bytes":torch.cuda.get_device_properties(0).total_memory,"bundle_sha256":checksum.hexdigest()}
(ROOT/"results/runtime.json").write_text(json.dumps(gpu,indent=2)+"\n")
frozen=ROOT/"frozen_transformer_config.json"
if frozen.exists():
    (ROOT/"results/transformer_config.json").write_bytes(frozen.read_bytes())
print(json.dumps(gpu),flush=True)
