"""Hash-bound stratified manifests; no preprocessing or model fitting."""

import csv
import hashlib
import json
from pathlib import Path
import torch

DEFAULT_DATA = Path("Chronic_Kidney_Disease/chronic_kidney_disease_full.csv")


def prepare_splits(path=DEFAULT_DATA, directory=Path("data/splits")):
    path, directory = Path(path), Path(directory)
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as data_file:
        rows = list(csv.DictReader(data_file))
    labels = [row["class"].strip() for row in rows]
    if set(labels) != {"ckd", "notckd"}:
        raise ValueError("Invalid targets")
    if min(labels.count(k) for k in ("ckd", "notckd")) < 10:
        raise ValueError(
            "Need at least 10 rows per class for this fixed five-fold protocol"
        )
    metadata = directory / "split_metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["dataset_hash"] != dataset_hash:
            raise ValueError("Dataset changed")
        all_ids = []
        for part in ["train", "validation", "test"]:
            with (directory / f"{part}_ids.csv").open() as manifest_file:
                current_ids = [
                    int(row["row_id"]) for row in csv.DictReader(manifest_file)
                ]
            if current_ids != saved[part]:
                raise ValueError("Manifest changed")
            all_ids.extend(current_ids)
        if sorted(all_ids) != list(range(len(rows))):
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
                {key: value for key, value in saved.items() if key != "split_id"},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if saved.get("split_id") != expected:
            raise ValueError("Split metadata changed")
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Partial split directory; inspect before recovery")
    generator = torch.Generator().manual_seed(42)
    result = {
        "dataset_hash": dataset_hash,
        "row_count": len(rows),
        "seed": 42,
        "train": [],
        "validation": [],
        "test": [],
        "cv_folds": [[] for _ in range(5)],
    }
    for label in ["notckd", "ckd"]:
        label_ids = torch.tensor(
            [row_id for row_id, row_label in enumerate(labels) if row_label == label]
        )
        label_ids = label_ids[
            torch.randperm(len(label_ids), generator=generator)
        ].tolist()
        train_end = int(0.6 * len(label_ids))
        validation_end = int(0.8 * len(label_ids))
        for part, subset in zip(
            ["train", "validation", "test"],
            [
                label_ids[:train_end],
                label_ids[train_end:validation_end],
                label_ids[validation_end:],
            ],
        ):
            result[part].extend(subset)
        for position, row_id in enumerate(label_ids[:train_end]):
            result["cv_folds"][position % 5].append(row_id)
    for part in ["train", "validation", "test"]:
        result[part].sort()
        with (directory / f"{part}_ids.csv").open("w") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(["row_id"])
            writer.writerows([[row_id] for row_id in result[part]])
    result["split_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    prepare_splits()
