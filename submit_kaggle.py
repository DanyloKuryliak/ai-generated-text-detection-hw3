"""Submit at most two validated candidates using the Kaggle CLI and log scores."""
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
SLUG="ai-generated-text-detection-hw3-ucu-ai-2026"


def cli(args):
    result=subprocess.run(["kaggle",*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout


def json_output(output):
    decoder=json.JSONDecoder()
    for i,char in enumerate(output):
        if char in "[{":
            try:
                value,end=decoder.raw_decode(output[i:])
                if not output[i+end:].strip(): return value
            except json.JSONDecodeError:
                pass
    raise ValueError("No complete JSON payload in CLI output")


def history():
    return json_output(cli(["competitions","submissions",SLUG,"--format","json"]))


def update_log(entries):
    path=ROOT/"SUBMISSIONS.md"
    lines=path.read_text().splitlines()
    lines=[line for line in lines if line!="No submissions have been sent to Kaggle."]
    for item in entries:
        value=item.get("public_score")
        public=str(value) if value is not None else "pending"
        row=f"| {item['date']} | {item['model']} | {item['cv_macro_f1']:.5f} | {public} | {item['file']} |"
        matches=[i for i,line in enumerate(lines) if item["file"] in line]
        if matches: lines[matches[0]]=row
        else: lines.append(row)
    path.write_text("\n".join(lines)+"\n")


def main():
    m=json.loads((ROOT/"results/final_metrics.json").read_text())
    assert len(m["submission_files"])<=2
    before=history()
    entries=[]
    for item in m["submission_files"]:
        path=ROOT/"submissions"/item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
        message=f"HW3; {item['model']}; grouped 3-fold CV macro-F1 {item['cv_macro_f1']:.5f}; 40k train"
        existing=[row for row in before if Path(str(row.get("fileName",""))).name==path.name and row.get("description")==message]
        entry={**item,"date":datetime.now(ZoneInfo("Europe/Kyiv")).date().isoformat(),
               "description":message,"public_score":None}
        entries.append(entry)
        update_log(entries)
        if not existing:
            print(cli(["competitions","submit",SLUG,"-f",str(path),"-m",message]),flush=True)
        else:
            print("Existing submission reused: "+path.name,flush=True)
    receipt={"status":"submitted","competition":SLUG,"submissions":entries}
    receipt_path=ROOT/"results/kaggle_submissions.json"
    for attempt in range(18):
        rows=history()
        for entry in entries:
            matched=[row for row in rows if Path(str(row.get("fileName",""))).name==entry["file"] and row.get("description")==entry["description"]]
            if matched:
                row=matched[0]
                entry["server_record"]=row
                value=row.get("publicScore")
                if value not in (None,""):
                    entry["public_score"]=float(value)
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
        update_log(entries)
        if all(item["public_score"] is not None for item in entries): break
        print(f"Waiting for Kaggle evaluation ({attempt+1}/18)",flush=True)
        time.sleep(10)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":
    main()
