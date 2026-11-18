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
    with Path(data).open() as data_file:
        rows = list(csv.DictReader(data_file))
    labels = [row["class"].strip() for row in rows]
    if set(labels) != {"ckd", "notckd"}:
        raise ValueError("Invalid targets")
    generator = torch.Generator().manual_seed(seed)
    parts = {"train": [], "validation": [], "test": []}
    folds = [[] for _ in range(5)]
    for label in ("notckd", "ckd"):
        label_ids = torch.tensor(
            [row_id for row_id, row_label in enumerate(labels) if row_label == label]
        )
        label_ids = label_ids[
            torch.randperm(len(label_ids), generator=generator)
        ].tolist()
        train_end = int(0.6 * len(label_ids))
        validation_end = int(0.8 * len(label_ids))
        subsets = (
            label_ids[:train_end],
            label_ids[train_end:validation_end],
            label_ids[validation_end:],
        )
        for key, subset in zip(parts, subsets):
            parts[key] += subset
        for position, row_id in enumerate(label_ids[:train_end]):
            folds[position % 5].append(row_id)
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
    for key, row_ids in parts.items():
        with (directory / f"{key}_ids.csv").open("w") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(["row_id"])
            writer.writerows([[row_id] for row_id in row_ids])
    write_json(directory / "split_metadata.json", manifest)
    return manifest


def train(data=DATA, split_dir=SPLITS, results_dir=RESULTS):
    manifest = create_split(data, split_dir)
    previous_manifest = json.loads(
        Path("data/splits/split_metadata.json").read_text()
    )
    if set(manifest["test"]) == set(previous_manifest["test"]):
        raise ValueError("New test membership matches existing split")
    train_features, train_labels, _ = load_data(data, manifest["train"])
    validation_features, validation_labels, _ = load_data(
        data, manifest["validation"]
    )
    pipeline = fit_pipeline(train_features, train_labels, CONFIG, SEED)
    validation_probabilities = pipeline.predict_proba(validation_features)
    threshold = choose_threshold(validation_labels, validation_probabilities)
    if threshold is None:
        raise ValueError("No validation threshold meets precision >= 0.80")
    test_features, test_labels, _ = load_data(data, manifest["test"])
    test_probabilities = pipeline.predict_proba(test_features)
    result_directory = Path(results_dir)
    result_directory.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "pipeline": pipeline,
            "config": CONFIG,
            "seed": SEED,
            "threshold": threshold,
            "split_id": manifest["split_id"],
        },
        result_directory / "model.pt",
    )
    write_json(
        result_directory / "report.json",
        {
            "split_id": manifest["split_id"],
            "threshold": threshold,
            "validation": metrics(
                validation_labels, validation_probabilities, threshold
            ),
            "test": metrics(test_labels, test_probabilities, threshold),
            "test_ids": manifest["test"],
            "note": "Reshuffled holdout; dataset rows were examined in prior EDA/experiments.",
        },
    )
    return result_directory


if __name__ == "__main__":
    print(train())
