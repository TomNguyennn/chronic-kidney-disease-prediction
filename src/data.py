"""Hash-bound stratified manifests; no preprocessing or model fitting."""

import csv
import hashlib
import json
from pathlib import Path
import torch

DEFAULT_DATA = Path("Chronic_Kidney_Disease/chronic_kidney_disease_full.csv")


def prepare_splits(path=DEFAULT_DATA, directory=Path("data/splits")):
    path, directory = Path(path), Path(directory)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    labels = [r["class"].strip() for r in rows]
    if set(labels) != {"ckd", "notckd"}:
        raise ValueError("Invalid targets")
    if min(labels.count(k) for k in ("ckd", "notckd")) < 10:
        raise ValueError(
            "Need at least 10 rows per class for this fixed five-fold protocol"
        )
    metadata = directory / "split_metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["dataset_hash"] != digest:
            raise ValueError("Dataset changed")
        ids = []
        for part in ["train", "validation", "test"]:
            with (directory / f"{part}_ids.csv").open() as f:
                current = [int(r["row_id"]) for r in csv.DictReader(f)]
            if current != saved[part]:
                raise ValueError("Manifest changed")
            ids.extend(current)
        if sorted(ids) != list(range(len(rows))):
            raise ValueError("Invalid membership")
        folds = saved["cv_folds"]
        if (
            len(folds) != 5
            or sorted(sum(folds, [])) != saved["train"]
            or any(not f for f in folds)
        ):
            raise ValueError("Invalid CV membership")
        expected = hashlib.sha256(
            json.dumps(
                {k: v for k, v in saved.items() if k != "split_id"}, sort_keys=True
            ).encode()
        ).hexdigest()
        if saved.get("split_id") != expected:
            raise ValueError("Split metadata changed")
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Partial split directory; inspect before recovery")
    g = torch.Generator().manual_seed(42)
    result = {
        "dataset_hash": digest,
        "row_count": len(rows),
        "seed": 42,
        "train": [],
        "validation": [],
        "test": [],
        "cv_folds": [[] for _ in range(5)],
    }
    for label in ["notckd", "ckd"]:
        ids = torch.tensor([i for i, v in enumerate(labels) if v == label])
        ids = ids[torch.randperm(len(ids), generator=g)].tolist()
        a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
        for part, subset in zip(
            ["train", "validation", "test"], [ids[:a], ids[a:b], ids[b:]]
        ):
            result[part].extend(subset)
        for i, row_id in enumerate(ids[:a]):
            result["cv_folds"][i % 5].append(row_id)
    for part in ["train", "validation", "test"]:
        result[part].sort()
        with (directory / f"{part}_ids.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id"])
            writer.writerows([[i] for i in result[part]])
    result["split_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    prepare_splits()
