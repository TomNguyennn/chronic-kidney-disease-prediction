"""Explicit experiment stages; no training on import or implicit test access."""

import copy
import csv
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import torch

from .data import DEFAULT_DATA, prepare_splits
from .evaluate import choose_threshold, metrics
from .models import Dummy, LogisticRegression, NeuralNetwork, RandomForest
from .schema import load_data
from .tabular import TabularPreprocessor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    code_hash = hashlib.sha256(
        b"".join(path.name.encode() + path.read_bytes() for path in source_files)
    ).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    versions = {}
    for package_name in ("torch", "numpy", "pandas", "matplotlib", "pytest"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return {
        "code_hash": code_hash,
        "code_commit": commit,
        "environment": versions,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_factory(config, seed):
    parameters = copy.deepcopy(config.get("model_params", {}))
    model_name = config["model"]
    if model_name == "dummy":
        return Dummy(**parameters)
    if model_name == "lr":
        return LogisticRegression(**parameters)
    if model_name == "rf":
        return RandomForest(seed=seed, **parameters)
    if model_name == "nn":
        return NeuralNetwork(seed=seed, **parameters)
    raise ValueError(f"Unknown model: {model_name}")


class FittedPipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocessor.transform(X))


def fit_pipeline(X, y, config, seed):
    preprocessing = copy.deepcopy(config.get("preprocessing", {}))
    default_scaler = "none" if config["model"] in ("rf", "dummy") else "standard"
    preprocessing.setdefault("scaler", default_scaler)
    history = None
    if config["model"] == "nn":
        generator = torch.Generator().manual_seed(42)
        inner_ids = []
        holdout_ids = []
        for label in (0, 1):
            label_ids = torch.where(y == label)[0]
            if len(label_ids) < 3:
                raise ValueError("Insufficient rows for internal early stopping")
            label_ids = label_ids[
                torch.randperm(len(label_ids), generator=generator)
            ].tolist()
            holdout_size = max(1, int(0.2 * len(label_ids)))
            holdout_ids += label_ids[:holdout_size]
            inner_ids += label_ids[holdout_size:]

        pilot_preprocessor = TabularPreprocessor(**preprocessing).fit(
            X[inner_ids], y[inner_ids]
        )
        pilot_model = model_factory(config, seed).fit(
            pilot_preprocessor.transform(X[inner_ids]),
            y[inner_ids],
            validation=(
                pilot_preprocessor.transform(X[holdout_ids]),
                y[holdout_ids],
            ),
        )
        history = {
            "train": pilot_model.history,
            "validation": pilot_model.validation_history,
            "best_epoch": pilot_model.best_epoch,
            "inner_ids": inner_ids,
            "holdout_ids": holdout_ids,
        }
        config = copy.deepcopy(config)
        config.setdefault("model_params", {})["epochs"] = pilot_model.best_epoch

    preprocessor = TabularPreprocessor(**preprocessing).fit(X, y)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        model = model_factory(config, seed).fit(preprocessor.transform(X), y)
    if (
        isinstance(model, LogisticRegression)
        and not isinstance(model, NeuralNetwork)
        and not model.converged
    ):
        raise RuntimeError(
            f"Logistic regression did not converge: gradient={model.gradient_norm}"
        )
    fitted = FittedPipeline(preprocessor, model)
    fitted.history = history
    fitted.warning_messages = [
        str(captured.message) for captured in captured_warnings
    ]
    return fitted


def validate_config(config):
    if not isinstance(config.get("experiment_id"), str):
        raise ValueError("Missing experiment ID")
    allowed_keys = {
        "experiment_id",
        "model",
        "model_params",
        "preprocessing",
        "seeds",
        "notes",
        "parent",
    }
    unknown_keys = set(config) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    seeds = config.get("seeds", [42, 43, 44] if config["model"] == "nn" else [42])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    if config["model"] == "nn" and seeds != [42, 43, 44]:
        raise ValueError("NN experiments require fixed seeds [42,43,44]")
    model_factory(config, seeds[0])
    TabularPreprocessor(**config.get("preprocessing", {}))
    return seeds


def run_experiment(
    config,
    data_path=DEFAULT_DATA,
    split_dir="data/splits",
    results_dir="results",
):
    """Run CV and validation without passing test rows to the data loader."""
    seeds = validate_config(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = prepare_splits(data_path, split_dir)
    provenance_info = provenance()
    config = copy.deepcopy(config)
    identity = json.dumps(
        {
            "config": config,
            "split_id": manifest["split_id"],
            "code_hash": provenance_info["code_hash"],
        },
        sort_keys=True,
    )
    run_id = (
        config["experiment_id"]
        + "-"
        + hashlib.sha256(identity.encode()).hexdigest()[:12]
    )
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    output_directory = results_root / run_id
    if output_directory.exists():
        raise FileExistsError(
            f"Run already exists: {output_directory}; use saved artifacts"
        )
    output_directory.mkdir()
    shutil.copytree(
        Path(__file__).parent,
        output_directory / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    write_json(output_directory / "config.json", config)
    write_json(
        output_directory / "provenance.json",
        {
            **provenance_info,
            "split_id": manifest["split_id"],
            "dataset_hash": manifest["dataset_hash"],
        },
    )
    start_time = time.monotonic()
    try:
        train_features, train_labels, _ = load_data(data_path, manifest["train"])
        validation_features, validation_labels, _ = load_data(
            data_path, manifest["validation"]
        )
        train_positions = {
            row_id: position for position, row_id in enumerate(manifest["train"])
        }
        fold_results = []
        predictions = []
        pipelines = []
        run_rows = []
        histories = []
        validation_seed_predictions = []
        for seed in seeds:
            seed_scores = []
            for fold, row_ids in enumerate(manifest["cv_folds"]):
                held_positions = [train_positions[row_id] for row_id in row_ids]
                held_position_set = set(held_positions)
                fitting_positions = [
                    position
                    for position in range(len(train_labels))
                    if position not in held_position_set
                ]
                pipeline = fit_pipeline(
                    train_features[fitting_positions],
                    train_labels[fitting_positions],
                    config,
                    seed,
                )
                probabilities = pipeline.predict_proba(
                    train_features[held_positions]
                )
                scores = metrics(train_labels[held_positions], probabilities)
                fold_results.append({"seed": seed, "fold": fold, **scores})
                seed_scores.append(scores["f2"])
                predictions.extend(
                    {
                        "seed": seed,
                        "fold": fold,
                        "row_id": row_id,
                        "y": int(train_labels[position]),
                        "probability": float(probability),
                    }
                    for row_id, position, probability in zip(
                        row_ids, held_positions, probabilities
                    )
                )

            pipeline = fit_pipeline(train_features, train_labels, config, seed)
            pipelines.append(pipeline)
            probabilities = pipeline.predict_proba(validation_features)
            threshold = choose_threshold(validation_labels, probabilities)
            validation_seed_predictions.extend(
                {
                    "seed": seed,
                    "row_id": row_id,
                    "y": int(label),
                    "probability": float(probability),
                }
                for row_id, label, probability in zip(
                    manifest["validation"], validation_labels, probabilities
                )
            )
            validation_result = metrics(
                validation_labels,
                probabilities,
                threshold if threshold is not None else 0.5,
            )
            score_tensor = torch.tensor(seed_scores, dtype=torch.float64)
            run_rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "run_id": run_id,
                    "split_id": manifest["split_id"],
                    "dataset_hash": manifest["dataset_hash"],
                    "feature_set": config.get("preprocessing", {}).get(
                        "feature_set", "all"
                    ),
                    "imputer": config.get("preprocessing", {}).get(
                        "imputer", "median"
                    ),
                    "encoder": "schema",
                    "scaler": pipeline.preprocessor.scaler,
                    "model": config["model"],
                    "model_params": json.dumps(config.get("model_params", {})),
                    "loss": (
                        config.get("model_params", {}).get("loss", "bce")
                        if config["model"] == "nn"
                        else None
                    ),
                    "seed": seed,
                    "cv_f2_mean": float(score_tensor.mean()),
                    "cv_f2_std": float(score_tensor.std(unbiased=True)),
                    "val_threshold": threshold,
                    "eligible": threshold is not None,
                    "code_commit": provenance_info["code_commit"],
                    "notes": config.get("notes", ""),
                    **{
                        "val_" + key: value
                        for key, value in validation_result.items()
                    },
                }
            )
            histories.append(
                {
                    "seed": seed,
                    "history": pipeline.history,
                    "features": pipeline.preprocessor.feature_names_,
                }
            )

        mean_probabilities = torch.stack(
            [pipeline.predict_proba(validation_features) for pipeline in pipelines]
        ).mean(0)
        threshold = choose_threshold(validation_labels, mean_probabilities)
        bundle = {
            "pipelines": pipelines,
            "config": config,
            "manifest": manifest,
            "provenance": provenance_info,
            "threshold": threshold,
        }
        torch.save(bundle, output_directory / "pipeline.pt")
        write_json(
            output_directory / "validation_predictions.json",
            [
                {
                    "row_id": row_id,
                    "y": int(label),
                    "probability": float(probability),
                }
                for row_id, label, probability in zip(
                    manifest["validation"], validation_labels, mean_probabilities
                )
            ],
        )
        write_json(output_directory / "cv_predictions.json", predictions)
        write_json(output_directory / "cv_metrics.json", fold_results)
        write_json(output_directory / "histories.json", histories)
        write_json(output_directory / "run_rows.json", run_rows)
        write_json(
            output_directory / "validation_seed_predictions.json",
            validation_seed_predictions,
        )
        write_json(
            output_directory / "warnings.json",
            [
                {
                    "seed": seed,
                    "warnings": pipeline.warning_messages,
                    "objective": getattr(pipeline.model, "objective", None),
                    "gradient_norm": getattr(
                        pipeline.model, "gradient_norm", None
                    ),
                    "converged": getattr(pipeline.model, "converged", None),
                }
                for seed, pipeline in zip(seeds, pipelines)
            ],
        )
        summary = {
            "run_id": run_id,
            "threshold": threshold,
            "eligible": threshold is not None,
            "seconds": time.monotonic() - start_time,
            "validation": (
                metrics(validation_labels, mean_probabilities, threshold)
                if threshold is not None
                else None
            ),
            "cv_f2_mean": sum(result["f2"] for result in fold_results)
            / len(fold_results),
            "status": "complete",
        }
        write_json(output_directory / "summary.json", summary)
        rebuild_runs(results_root)
        return output_directory
    except Exception as error:
        write_json(
            output_directory / "summary.json",
            {
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "seconds": time.monotonic() - start_time,
            },
        )
        raise


def rebuild_runs(root="results"):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/run_rows.json")):
        rows.extend(json.loads(path.read_text()))
    if rows:
        with (root / "runs.csv").open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def search_configs(base):
    """Return the predeclared E09 search space as reproducible run configs."""
    model_name = base["model"]
    if model_name == "lr":
        space = {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}
    elif model_name == "rf":
        space = {
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5, 10],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced"],
        }
    elif model_name == "nn":
        space = {
            "hidden": [[16], [32, 16]],
            "batch_size": [16, 32],
            "weight_decay": [0.0, 1e-4],
        }
    else:
        raise ValueError("Dummy models do not have a search")

    combinations = list(itertools.product(*space.values()))
    if model_name == "rf":
        selected_indices = torch.randperm(
            len(combinations), generator=torch.Generator().manual_seed(42)
        )[:24].tolist()
        combinations = [combinations[index] for index in selected_indices]
    result = []
    for index, values in enumerate(combinations):
        config = copy.deepcopy(base)
        config["parent"] = base["experiment_id"]
        config["experiment_id"] = f"E09-{model_name}-{index:02}"
        config.setdefault("model_params", {}).update(dict(zip(space, values)))
        result.append(config)
    return result


def loss_configs(base):
    if base["model"] != "nn":
        raise ValueError("Loss studies require NN")
    result = []
    for loss in ["bce", "weighted_bce", "mse", "focal"]:
        config = copy.deepcopy(base)
        config["parent"] = base["experiment_id"]
        config["experiment_id"] = "E10-" + loss
        config.setdefault("model_params", {})["loss"] = loss
        result.append(config)
    return result
