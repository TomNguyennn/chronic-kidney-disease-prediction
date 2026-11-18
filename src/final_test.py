"""Explicit one-shot test stage. Reporting can be repeated from cached predictions."""
import json
from pathlib import Path
import shutil
import torch

from .data import DEFAULT_DATA, prepare_splits
from .evaluate import bootstrap_intervals, metrics
from .schema import load_data
from .train import digest, provenance, write_json


def freeze_finalist(run_dir, final_dir="results/final_test"):
    run_dir = Path(run_dir)
    final_dir = Path(final_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("status") != "complete" or not summary.get("eligible"):
        raise ValueError("Finalist must have a feasible validation threshold")
    if final_dir.exists():
        raise FileExistsError(
            "Finalist already frozen; no replacement in this protocol"
        )
    bundle = torch.load(
        run_dir / "pipeline.pt", weights_only=False, map_location="cpu"
    )
    if bundle["threshold"] != summary["threshold"]:
        raise ValueError("Inconsistent run artifacts")
    final_dir.mkdir(parents=True)
    pipeline_path = final_dir / "pipeline.pt"
    shutil.copy2(run_dir / "pipeline.pt", pipeline_path)
    write_json(
        final_dir / "selection.json",
        {
            "run_id": summary["run_id"],
            "pipeline_hash": digest(pipeline_path),
            "threshold": bundle["threshold"],
            "dataset_hash": bundle["manifest"]["dataset_hash"],
            "split_id": bundle["manifest"]["split_id"],
            "code_hash": bundle["provenance"]["code_hash"],
        },
    )
    return final_dir


def evaluate_final(
    data_path=DEFAULT_DATA,
    split_dir="data/splits",
    final_dir="results/final_test",
    n_resamples=2000,
):
    final_dir = Path(final_dir)
    selection = json.loads((final_dir / "selection.json").read_text())
    manifest = prepare_splits(data_path, split_dir)
    if (
        selection["dataset_hash"] != manifest["dataset_hash"]
        or selection["split_id"] != manifest["split_id"]
    ):
        raise ValueError("Dataset/split changed")
    pipeline_path = final_dir / "pipeline.pt"
    if digest(pipeline_path) != selection["pipeline_hash"]:
        raise ValueError("Frozen pipeline changed")
    if provenance()["code_hash"] != selection["code_hash"]:
        raise ValueError(
            "Code changed since fitting; final-test protocol requires the "
            "recorded code version"
        )
    # Exclusive marker protects against concurrent or accidental repeat test access.
    with (final_dir / "test_started.json").open("x") as marker_file:
        json.dump(selection, marker_file)
    bundle = torch.load(pipeline_path, weights_only=False, map_location="cpu")
    features, labels, _ = load_data(data_path, manifest["test"])
    probabilities = torch.stack(
        [
            pipeline.predict_proba(features)
            for pipeline in bundle["pipelines"]
        ]
    ).mean(0)
    write_json(
        final_dir / "predictions.json",
        [
            {
                "row_id": row_id,
                "y": int(label),
                "probability": float(probability),
            }
            for row_id, label, probability in zip(
                manifest["test"], labels, probabilities
            )
        ],
    )
    return report_saved_test(final_dir, n_resamples)


def report_saved_test(final_dir="results/final_test", n_resamples=2000):
    final_dir = Path(final_dir)
    selection = json.loads((final_dir / "selection.json").read_text())
    rows = json.loads((final_dir / "predictions.json").read_text())
    labels = [row["y"] for row in rows]
    probabilities = [row["probability"] for row in rows]
    threshold = selection["threshold"]
    report = {
        "selection": selection,
        "metrics": metrics(labels, probabilities, threshold),
        "intervals": bootstrap_intervals(
            labels, probabilities, threshold, n_resamples
        ),
        "n_resamples": n_resamples,
        "limitations": [
            "Full dataset examined in prior EDA",
            "No verified patient or temporal identifiers",
            "Three unverified ARFF repairs",
            "Intervals condition on selected pipeline and class counts; they "
            "exclude selection and training uncertainty",
        ],
    }
    write_json(final_dir / "report.json", report)
    return report
