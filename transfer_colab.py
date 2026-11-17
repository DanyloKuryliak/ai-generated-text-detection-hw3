"""Upload a prepared data bundle in bounded chunks through the Colab CLI."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def upload(path):
    result = subprocess.run(["colab", "upload", "-s", "hw3-nlp", str(path),
                             f"/content/hw3/{path.name}"], capture_output=True, text=True,
                            timeout=180)
    if result.returncode:
        raise RuntimeError(f"Upload failed: {path.name}: {result.stderr[-1500:]}")
    return path.name


if __name__ == "__main__":
    subprocess.run(["colab", "exec", "-s", "hw3-nlp"],
        input="from pathlib import Path\nPath('/content/hw3').mkdir(exist_ok=True)\n",
        text=True, check=True, capture_output=True, timeout=120)
    paths = sorted((ROOT / "data/transfer").glob("bundle.part*"))
    if not paths:
        raise FileNotFoundError("Run prepare_colab.py first")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, name in enumerate(pool.map(upload, paths), 1):
            print(f"Uploaded {i}/{len(paths)}: {name}", flush=True)
    for name in ["classical_experiments.py", "transformer_experiments.py", "finalize_experiments.py",
                 "run_cloud.py", "collect_cloud.py"]:
        print("Uploaded code: " + upload(ROOT/name),flush=True)
    upload(ROOT/"results/colab_bundle_metadata.json")
    frozen=ROOT/"results/transformer_config.json"
    if frozen.exists():
        subprocess.run(["colab","upload","-s","hw3-nlp",str(frozen),
                        "/content/hw3/frozen_transformer_config.json"],check=True,timeout=120)
    subprocess.run(["colab","exec","-s","hw3-nlp","--timeout","180","-f",str(ROOT/"setup_cloud.py")],
                    check=True,timeout=240)
    recovered=ROOT/"data/classical_recovered.zip"
    if recovered.exists():
        upload(recovered)
        subprocess.run(["colab","exec","-s","hw3-nlp"],
            input="from pathlib import Path\nimport zipfile\nr=Path('/content/hw3')\nwith zipfile.ZipFile(r/'classical_recovered.zip') as z:\n assert all(Path(n).parts[0] in ['results','predictions'] and '..' not in Path(n).parts for n in z.namelist())\n z.extractall(r)\nprint('Classical artifacts restored.')\n",
            text=True,check=True,capture_output=True,timeout=180)
