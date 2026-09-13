"""Shared helpers: logging, JSONL files that start with a header line, and
the results layout that every script reads and writes.

    <results>/<model>/<dataset>/workload.jsonl           workloads.py
    <results>/<model>/<dataset>/solo.json                solo.py
    <results>/<model>/<dataset>/<method>/rate_<r>.jsonl  run_sweep.py

<model> and <dataset> are the names given on the command line. <method> is
the method name with every character other than letters, digits and ._+-
replaced by "_"; the run header keeps the name itself, and readers take the
method and the rate from the header, not from the path.
"""
import json
import os
import re
import time
from pathlib import Path

DATASETS = {"sharegpt": "ShareGPT", "longbench": "LongBench v2"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _replace(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        write(f)
    os.replace(tmp, path)


def write_jsonl(path, header, records):
    """Write {"header": header} and then one record per line, atomically."""
    def write(f):
        f.write(json.dumps({"header": header}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    _replace(path, write)


def write_json(path, obj):
    _replace(path, lambda f: json.dump(obj, f, indent=1))


def read_header(path):
    with open(path) as f:
        return json.loads(f.readline())["header"]


def read_jsonl(path):
    """(header, records) of a file written by write_jsonl."""
    with open(path) as f:
        header = json.loads(f.readline())["header"]
        return header, [json.loads(line) for line in f if line.strip()]


def method_dir(method):
    return re.sub(r"[^A-Za-z0-9._+-]", "_", method)


def workload_path(results, model, dataset):
    return Path(results) / model / dataset / "workload.jsonl"


def solo_path(results, model, dataset):
    return Path(results) / model / dataset / "solo.json"


def run_path(results, model, dataset, method, rate):
    return (Path(results) / model / dataset / method_dir(method)
            / f"rate_{rate:g}.jsonl")


def find_runs(results, model, dataset):
    """{method: [(rate, path, header), ...] sorted by rate} for every run
    file of one model and dataset."""
    runs = {}
    for p in sorted((Path(results) / model / dataset).glob("*/rate_*.jsonl")):
        h = read_header(p)
        runs.setdefault(h["method"], []).append((h["rate"], p, h))
    return {m: sorted(v, key=lambda x: x[0]) for m, v in runs.items()}


def find_models(results):
    """Model directories under results that hold a dataset directory."""
    return sorted(p.name for p in Path(results).iterdir()
                  if p.is_dir() and any((p / d).is_dir() for d in DATASETS))
