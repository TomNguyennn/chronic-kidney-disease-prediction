"""Fixed binary metrics and constrained threshold selection."""

import torch


def metrics(y, p, threshold=0.5):
    y = torch.as_tensor(y)
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("Labels must be 0/1")
    y = y.bool()
    p = torch.as_tensor(p, dtype=torch.float64)
    if (
        y.shape != p.shape
        or y.ndim != 1
        or not len(y)
        or not torch.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid predictions")

    predictions = p >= threshold
    true_positives = int((predictions & y).sum())
    false_positives = int((predictions & ~y).sum())
    true_negatives = int((~predictions & ~y).sum())
    false_negatives = int((~predictions & y).sum())
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f2_denominator = (
        5 * true_positives + 4 * false_negatives + false_positives
    )
    result = {
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "specificity": (
            true_negatives / (true_negatives + false_positives)
            if true_negatives + false_positives
            else None
        ),
        "f2": 5 * true_positives / f2_denominator if f2_denominator else 0.0,
        "brier": float((p - y.double()).square().mean()),
    }

    if y.any() and (~y).any():
        positive_probabilities = p[y]
        negative_probabilities = p[~y]
        result["roc_auc"] = float(
            (
                (positive_probabilities[:, None] > negative_probabilities).double()
                + 0.5
                * (
                    positive_probabilities[:, None] == negative_probabilities
                )
            ).mean()
        )
        average_precision = 0.0
        previous_recall = 0.0
        for candidate_threshold in p.unique().sort(descending=True).values:
            selected = p >= candidate_threshold
            positives_found = float(y[selected].sum())
            recall_at_threshold = positives_found / len(positive_probabilities)
            average_precision += (
                (recall_at_threshold - previous_recall)
                * positives_found
                / int(selected.sum())
            )
            previous_recall = recall_at_threshold
        result["ap"] = average_precision
    else:
        result.update(ap=None, roc_auc=None)
    return result


def choose_threshold(y, p, precision_floor=0.8):
    p = torch.as_tensor(p, dtype=torch.float64)
    if not 0 <= precision_floor <= 1:
        raise ValueError("Invalid precision floor")
    metrics(y, p)
    candidates = []
    for threshold in p.unique():
        result = metrics(y, p, float(threshold))
        has_positive_predictions = result["tp"] + result["fp"]
        if has_positive_predictions and result["precision"] >= precision_floor:
            candidates.append(
                (result["f2"], result["precision"], float(threshold))
            )
    return max(candidates)[2] if candidates else None


def bootstrap_intervals(y, p, threshold, n_resamples=2000, seed=42):
    """Conditional stratified percentile intervals; no model refitting/selection."""
    y = torch.as_tensor(y, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    metrics(y, p, threshold)
    class_groups = [torch.where(y == label)[0] for label in (0, 1)]
    if any(len(group) == 0 for group in class_groups) or n_resamples < 2:
        raise ValueError("Both classes and at least two resamples required")

    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(n_resamples):
        sampled_ids = torch.cat(
            [
                group[
                    torch.randint(
                        len(group),
                        (len(group),),
                        generator=generator,
                    )
                ]
                for group in class_groups
            ]
        )
        scores.append(metrics(y[sampled_ids], p[sampled_ids], threshold))

    interval = torch.tensor([0.025, 0.975], dtype=torch.float64)
    return {
        key: torch.tensor(
            [score[key] for score in scores], dtype=torch.float64
        ).quantile(interval).tolist()
        for key in [
            "f2",
            "precision",
            "recall",
            "specificity",
            "ap",
            "roc_auc",
            "brier",
        ]
    }
