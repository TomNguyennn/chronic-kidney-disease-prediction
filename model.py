"""Final single-seed CKD network; run with `python model.py`."""

import csv
import hashlib
import json
from pathlib import Path
import torch

from src.data import DEFAULT_DATA
from src.schema import load_data
from src.train import fit_pipeline, write_json
from src.evaluate import choose_threshold, metrics

DATA = DEFAULT_DATA
SPLITS = Path("data/splits_model")
RESULTS = Path("results/final_model")
CONFIG = {
    "experiment_id": "final-knn-onehot-nn",
    "model": "nn",
    "preprocessing": {"imputer": "knn", "scaler": "standard", "ordered_onehot": True},
    "model_params": {
        "hidden": [32, 16],
        "batch_size": 32,
        "weight_decay": 0.0,
        "epochs": 200,
        "loss": "bce",
    },
}
SEED = 44


def create_split(data=DATA, directory=SPLITS, seed=43):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with Path(data).open() as f:
        rows = list(csv.DictReader(f))
    labels = [r["class"].strip() for r in rows]
    if set(labels) != {"ckd", "notckd"}:
        raise ValueError("Invalid targets")
    g = torch.Generator().manual_seed(seed)
    parts = {"train": [], "validation": [], "test": []}
    folds = [[] for _ in range(5)]
    for label in ("notckd", "ckd"):
        ids = torch.tensor([i for i, v in enumerate(labels) if v == label])
        ids = ids[torch.randperm(len(ids), generator=g)].tolist()
        a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
        for key, subset in zip(parts, (ids[:a], ids[a:b], ids[b:])):
            parts[key] += subset
        for i, rid in enumerate(ids[:a]):
            folds[i % 5].append(rid)
    for key in parts:
        parts[key].sort()
    manifest = {
        "dataset_hash": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        "row_count": len(rows),
        "seed": seed,
        **parts,
        "cv_folds": folds,
    }
    manifest["split_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    for key, ids in parts.items():
        with (directory / f"{key}_ids.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["row_id"])
            w.writerows([[i] for i in ids])
    write_json(directory / "split_metadata.json", manifest)
    return manifest


def train(data=DATA, split_dir=SPLITS, results_dir=RESULTS):
    m = create_split(data, split_dir)
    old = json.loads(Path("data/splits/split_metadata.json").read_text())
    if set(m["test"]) == set(old["test"]):
        raise ValueError("New test membership matches existing split")
    X, y, _ = load_data(data, m["train"])
    vx, vy, _ = load_data(data, m["validation"])
    pipeline = fit_pipeline(X, y, CONFIG, SEED)
    p = pipeline.predict_proba(vx)
    threshold = choose_threshold(vy, p)
    if threshold is None:
        raise ValueError("No validation threshold meets precision >= 0.80")
    tx, ty, _ = load_data(data, m["test"])
    tp = pipeline.predict_proba(tx)
    result = Path(results_dir)
    result.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "pipeline": pipeline,
            "config": CONFIG,
            "seed": SEED,
            "threshold": threshold,
            "split_id": m["split_id"],
        },
        result / "model.pt",
    )
    write_json(
        result / "report.json",
        {
            "split_id": m["split_id"],
            "threshold": threshold,
            "validation": metrics(vy, p, threshold),
            "test": metrics(ty, tp, threshold),
            "test_ids": m["test"],
            "note": "Reshuffled holdout; dataset rows were examined in prior EDA/experiments.",
        },
    )
    return result


if __name__ == "__main__":
    print(train())
