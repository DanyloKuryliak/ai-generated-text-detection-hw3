"""Audit the supplied CSVs without displaying document contents."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
csv.field_size_limit(10_000_000)


def audit(name):
    counts = {key: Counter() for key in ("label", "lang", "domain")}
    missing = Counter()
    lengths = []
    ids = set()
    hashes = Counter()
    joint = Counter()
    with (ROOT / "data" / name).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            ids.add(row["id"])
            for key, value in row.items():
                if value is None or not value.strip():
                    missing[key] += 1
            for key in counts:
                if key in row:
                    counts[key][row[key]] += 1
            if "text" in row:
                text = row["text"]
                lengths.append(len(text.split()))
                hashes[hashlib.sha256(" ".join(text.split()).encode()).hexdigest()] += 1
            if "label" in row:
                joint[(row["lang"], row["domain"], row["label"])] += 1
    n = sum(counts["lang"].values())
    result = {"columns": columns, "rows": n, "unique_ids": len(ids),
              "missing": dict(missing),
              "counts": {key: dict(value) for key, value in counts.items()},
              "duplicate_text_extra_rows": sum(v - 1 for v in hashes.values()),
              "language_domain_label": [{"lang": k[0], "domain": k[1],
                  "label": k[2], "count": v} for k, v in sorted(joint.items())]}
    if lengths:
        result["word_lengths"] = {"min": min(lengths), "median": median(lengths),
                                  "max": max(lengths),
                                  "over_512_words": sum(x > 512 for x in lengths)}
    return result, ids, set(hashes)


if __name__ == "__main__":
    train, train_ids, train_hashes = audit("train.csv")
    test, test_ids, test_hashes = audit("test.csv")
    with (ROOT / "data" / "sample_submission.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        submission_columns = reader.fieldnames
        sample_ids = [row["id"] for row in reader]
    output = {"train": train, "test": test,
              "overlapping_ids": len(train_ids & test_ids),
              "overlapping_normalized_texts": len(train_hashes & test_hashes),
              "sample_submission": {"columns": submission_columns,
                  "rows": len(sample_ids), "unique_ids": len(set(sample_ids)),
                  "matches_test_ids": set(sample_ids) == test_ids}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    for name, result in (("train", train), ("test", test)):
        print(name, json.dumps({k: v for k, v in result.items()
                               if k != "language_domain_label"}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in output.items() if k not in ("train", "test")}))
