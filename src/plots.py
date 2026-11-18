"""Plots from saved development predictions; no test loading or model fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import DEFAULT_DATA, prepare_splits
from .evaluate import metrics
from .schema import NAMES, load_data


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_predictions(y, p, threshold, out_dir):
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = torch.as_tensor(y)
    probabilities = torch.as_tensor(p, dtype=torch.float64)
    thresholds = sorted({0.0, 1.0, *probabilities.tolist()})
    scores = [metrics(labels, probabilities, value) for value in thresholds]

    figure, axis = plt.subplots()
    for metric_name in ["f2", "precision", "recall", "specificity"]:
        axis.plot(
            thresholds,
            [result[metric_name] for result in scores],
            label=metric_name,
        )
    if threshold is not None:
        axis.axvline(
            threshold, color="black", linestyle="--", label="selected"
        )
    axis.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1.05))
    axis.legend()
    _save(figure, output_directory / "thresholds.png")

    figure, axis = plt.subplots()
    mean_probabilities = []
    positive_fractions = []
    bin_edges = torch.linspace(0, 1, 6)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1 else (probabilities <= upper)
        )
        if in_bin.any():
            mean_probabilities.append(float(probabilities[in_bin].mean()))
            positive_fractions.append(float(labels[in_bin].double().mean()))
    axis.plot(
        mean_probabilities,
        positive_fractions,
        "o-",
        label="5 equal-width bins",
    )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    _save(figure, output_directory / "calibration.png")

    if threshold is not None:
        result = metrics(labels, probabilities, threshold)
        confusion_matrix = torch.tensor(
            [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
        )
        row_percentages = (
            confusion_matrix.double()
            / confusion_matrix.sum(1, keepdim=True).clamp_min(1)
            * 100
        )
        figure, axes = plt.subplots(1, 2, figsize=(8, 3))
        for axis, values, title in zip(
            axes,
            [confusion_matrix, row_percentages],
            ["Counts", "Row percentages"],
        ):
            axis.imshow(values.numpy(), cmap="Blues")
            axis.set(
                title=title,
                xlabel="Predicted",
                ylabel="Actual",
                xticks=[0, 1],
                yticks=[0, 1],
            )
            for row_index in range(2):
                for column_index in range(2):
                    axis.text(
                        column_index,
                        row_index,
                        f"{values[row_index, column_index]:.1f}",
                        ha="center",
                    )
        _save(figure, output_directory / "confusion.png")


def compare_runs(run_dirs, out_dir="results/figures"):
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    precision_recall_figure, precision_recall_axis = plt.subplots()
    roc_figure, roc_axis = plt.subplots()
    cv_figure, cv_axis = plt.subplots()
    reference = None
    distributions = []
    labels = []
    fold_table = {}

    for run_directory in map(Path, run_dirs):
        prediction_rows = json.loads(
            (run_directory / "validation_predictions.json").read_text()
        )
        identity = [(row["row_id"], row["y"]) for row in prediction_rows]
        provenance = json.loads((run_directory / "provenance.json").read_text())
        current = (identity, provenance["split_id"])
        if reference is not None and current != reference:
            raise ValueError("Cannot compare different validation partitions")
        reference = current

        run_labels = [row["y"] for row in prediction_rows]
        run_probabilities = [row["probability"] for row in prediction_rows]
        summary = json.loads((run_directory / "summary.json").read_text())
        label = summary["run_id"]
        scores = [
            metrics(run_labels, run_probabilities, threshold)
            for threshold in [float("inf")]
            + sorted(set(run_probabilities), reverse=True)
        ]
        precision_recall_axis.plot(
            [result["recall"] for result in scores],
            [1.0] + [result["precision"] for result in scores[1:]],
            label=label,
        )
        roc_axis.plot(
            [1 - result["specificity"] for result in scores],
            [result["recall"] for result in scores],
            label=label,
        )

        fold_scores = json.loads((run_directory / "cv_metrics.json").read_text())
        fold_means = [
            sum(row["f2"] for row in fold_scores if row["fold"] == fold)
            / sum(row["fold"] == fold for row in fold_scores)
            for fold in range(5)
        ]
        distributions.append(fold_means)
        labels.append(label)
        fold_table[label] = fold_means
        plot_predictions(
            run_labels,
            run_probabilities,
            summary["threshold"],
            output_directory / label,
        )

        histories = json.loads((run_directory / "histories.json").read_text())
        for entry in histories:
            history = entry["history"]
            if history:
                figure, axis = plt.subplots()
                axis.plot(history["train"], label="inner training")
                axis.plot(history["validation"], label="inner holdout")
                axis.set(
                    xlabel="Epoch (zero-based)",
                    ylabel="Loss",
                    title=f"{label}, seed {entry['seed']}",
                )
                axis.legend()
                _save(
                    figure,
                    output_directory / label / f"learning_{entry['seed']}.png",
                )

    for axis in (precision_recall_axis, roc_axis):
        axis.legend(fontsize=6)
    precision_recall_axis.set(xlabel="Recall", ylabel="Precision")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate")
    roc_axis.plot([0, 1], [0, 1], "--", color="gray")
    cv_axis.boxplot(distributions, tick_labels=labels)
    cv_axis.tick_params(axis="x", labelrotation=90)
    cv_axis.set(ylabel="F2 at 0.5 (seed-averaged per fold)")
    _save(precision_recall_figure, output_directory / "precision_recall.png")
    _save(roc_figure, output_directory / "roc.png")
    _save(cv_figure, output_directory / "cv_distribution.png")

    paired_deltas = {}
    if labels:
        reference_scores = fold_table[labels[0]]
        for label in labels[1:]:
            paired_deltas[label] = [
                current - reference
                for current, reference in zip(fold_table[label], reference_scores)
            ]
    (output_directory / "paired_fold_deltas.json").write_text(
        json.dumps(
            {
                "reference": labels[0] if labels else None,
                "deltas": paired_deltas,
            },
            indent=2,
        )
    )
    return fold_table


def permutation_importance(
    run_dir, data_path=DEFAULT_DATA, split_dir="data/splits", repeats=10
):
    run_directory = Path(run_dir)
    bundle = torch.load(
        run_directory / "pipeline.pt", weights_only=False, map_location="cpu"
    )
    manifest = prepare_splits(data_path, split_dir)
    if bundle["manifest"]["split_id"] != manifest["split_id"]:
        raise ValueError("Split mismatch")
    features, labels, _ = load_data(data_path, manifest["validation"])
    threshold = bundle["threshold"]
    if threshold is None:
        raise ValueError("No feasible threshold")

    def score(permuted_features):
        probabilities = torch.stack(
            [
                pipeline.predict_proba(permuted_features)
                for pipeline in bundle["pipelines"]
            ]
        ).mean(0)
        return metrics(labels, probabilities, threshold)["f2"]

    baseline = score(features)
    generator = torch.Generator().manual_seed(42)
    result = []
    for feature_index, name in enumerate(NAMES):
        deltas = []
        for _ in range(repeats):
            permuted_features = features.clone()
            order = torch.randperm(len(permuted_features), generator=generator)
            permuted_features[:, feature_index] = permuted_features[
                order, feature_index
            ]
            deltas.append(baseline - score(permuted_features))
        delta_tensor = torch.tensor(deltas)
        result.append(
            {
                "feature": name,
                "mean": float(delta_tensor.mean()),
                "std": float(delta_tensor.std(unbiased=False)),
            }
        )

    (run_directory / "permutation_importance.json").write_text(
        json.dumps(result, indent=2)
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.barh(
        [row["feature"] for row in result],
        [row["mean"] for row in result],
        xerr=[row["std"] for row in result],
    )
    axis.set(xlabel="Validation F2 decrease (fixed threshold)")
    _save(figure, run_directory / "permutation_importance.png")
    return result
