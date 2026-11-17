"""Run the CPU and GPU experiments, stream logs, and collect compact artifacts."""
import json
import runpy
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

ROOT = Path("/content/hw3")
for folder in ("results", "predictions", "models"):
    (ROOT/folder).mkdir(exist_ok=True)


def job(script):
    log_path=ROOT/"results"/(Path(script).stem+".log")
    with log_path.open("a") as log:
        process=subprocess.Popen([sys.executable,"-u",str(ROOT/script)],cwd=ROOT,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush()
            print(f"[{Path(script).stem}] {line.rstrip()}",flush=True)
        code=process.wait()
    (ROOT/"results"/(Path(script).stem+"_status.json")).write_text(json.dumps({"exit_code":code}))


scripts=["classical_experiments.py","transformer_experiments.py"]
recovered=all((ROOT/f"predictions/{name}.npz").exists() for name in ["word","character","classical_blend"])
if recovered and (ROOT/"results/classical_metrics.json").exists():
    scripts.remove("classical_experiments.py")
    (ROOT/"results/classical_experiments_status.json").write_text(json.dumps({"exit_code":0,"source":"verified recovered classical artifacts"}))
    print("Using recovered classical artifacts; finalizer will verify all IDs and probabilities.",flush=True)
threads=[threading.Thread(target=job,args=(name,)) for name in scripts]
for thread in threads: thread.start()
for thread in threads: thread.join()
with zipfile.ZipFile(ROOT/"experiment_artifacts.zip","w",zipfile.ZIP_DEFLATED) as out:
    for folder in ("results","predictions"):
        for path in sorted((ROOT/folder).glob("*")):
            if path.is_file(): out.write(path,path.relative_to(ROOT))
failed=[]
for name in ["classical_experiments","transformer_experiments"]:
    status=json.loads((ROOT/"results"/(name+"_status.json")).read_text())
    if status["exit_code"]: failed.append(name)
if failed: raise RuntimeError("Failed jobs: "+", ".join(failed))
runpy.run_path(str(ROOT/"collect_cloud.py"),run_name="__main__")
print("ALL EXPERIMENTS COMPLETE: /content/hw3/experiment_artifacts.zip",flush=True)
