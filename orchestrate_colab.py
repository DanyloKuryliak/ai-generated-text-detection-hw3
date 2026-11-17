"""Run experiments with incremental local backups and verified final retrieval."""
import hashlib
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RECOVERY=ROOT/"data/colab_recovery"
RECOVERY.mkdir(parents=True,exist_ok=True)
(RECOVERY/"predictions").mkdir(exist_ok=True)
(RECOVERY/"results").mkdir(exist_ok=True)
SESSION="hw3-nlp"


def command(args,timeout=180):
    return subprocess.run(["colab",*args],capture_output=True,text=True,timeout=timeout)


def checkpoint():
    listing=command(["ls","-s",SESSION,"/content/hw3/predictions"],60)
    if listing.returncode==0:
        for name in listing.stdout.splitlines():
            name=name.strip()
            if re.fullmatch(r"[A-Za-z0-9_]+\.npz",name):
                target=RECOVERY/"predictions"/name
                if not target.exists():
                    temporary=target.with_suffix(".partial")
                    result=command(["download","-s",SESSION,f"/content/hw3/predictions/{name}",str(temporary)])
                    if result.returncode==0 and zipfile.is_zipfile(temporary):
                        with zipfile.ZipFile(temporary) as saved:
                            valid=saved.testzip() is None
                        if valid:
                            temporary.replace(target)
                            print("Checkpoint retrieved: "+name,flush=True)
    listing=command(["ls","-s",SESSION,"/content/hw3/results"],60)
    if listing.returncode==0:
        for name in listing.stdout.splitlines():
            name=name.strip()
            if re.fullmatch(r"[A-Za-z0-9_]+\.json",name):
                target=RECOVERY/"results"/name
                command(["download","-s",SESSION,f"/content/hw3/results/{name}",str(target)])


def main():
    log=ROOT/"results/cloud_recovery_run.log"
    status_path=ROOT/"results/orchestration_status.json"
    start=time.monotonic()
    with log.open("w") as handle:
        process=subprocess.Popen(["colab","exec","-s",SESSION,"--timeout","14400","-f",str(ROOT/"run_cloud.py")],
                                  cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        next_backup=start+60
        while process.poll() is None:
            if time.monotonic()>=next_backup:
                try: checkpoint()
                except Exception as error: print("Checkpoint retry later: "+type(error).__name__,flush=True)
                next_backup=time.monotonic()+90
            status_path.write_text(json.dumps({"phase":"running","elapsed_seconds":time.monotonic()-start,
                                              "local_checkpoint_files":len(list(RECOVERY.rglob("*.npz")))},indent=2)+"\n")
            time.sleep(10)
        code=process.returncode
    archive=ROOT/"data/experiment_artifacts.zip"
    result=command(["download","-s",SESSION,"/content/hw3/experiment_artifacts.zip",str(archive)])
    if result.returncode:
        checkpoint()
        raise RuntimeError("Final archive retrieval failed; runtime left active. "+result.stderr[-1000:])
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        for name in saved.namelist():
            p=Path(name)
            assert not p.is_absolute() and ".." not in p.parts
            assert p.parts[0] in ["results","predictions","submissions"]
        if code==0:
            assert "results/final_metrics.json" in saved.namelist()
            assert "predictions/final_selected.npz" in saved.namelist()
        saved.extractall(ROOT)
    status={"phase":"artifacts_retrieved","remote_exit_code":code,
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "elapsed_seconds":time.monotonic()-start}
    if code==0:
        stopped=command(["stop","-s",SESSION])
        status["runtime_stopped"]=stopped.returncode==0
    status_path.write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status),flush=True)
    if code: raise RuntimeError("Remote experiment failed; available artifacts retrieved")


if __name__=="__main__":
    main()
