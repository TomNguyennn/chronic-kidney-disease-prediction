"""Schema-aware tensor pipeline. Every learned operation lives in fit."""

import math
import torch
from .schema import NAMES, NUMERIC, ORDERED, CATEGORIES, HISTORY
from .models import RandomForest


class NumericImputer:
    def __init__(self, strategy="median"):
        if strategy not in ("median", "knn", "iterative"):
            raise ValueError(strategy)
        self.strategy = strategy

    def fit(self, X):
        if X.isnan().all(0).any():
            raise ValueError("Entirely missing numeric feature")
        self.width = X.shape[1]
        self.median = torch.nanquantile(X, 0.5, dim=0)
        self.steps = []
        if self.strategy == "knn":
            self.mean = torch.nanmean(X, 0)
            self.scale = torch.nanmean((X - self.mean).square(), 0).sqrt()
            self.scale = torch.where(self.scale == 0, 1.0, self.scale)
            self.donors = ((X - self.mean) / self.scale).clone()
        elif self.strategy == "iterative":
            filled = torch.where(X.isnan(), self.median, X).clone()
            for _ in range(10):
                for feature_index in range(self.width):
                    other_features = [
                        index for index in range(self.width) if index != feature_index
                    ]
                    observed = ~X[:, feature_index].isnan()
                    predictors = filled[:, other_features]
                    center = predictors[observed].mean(0)
                    scale = predictors[observed].std(0, unbiased=False)
                    scale = torch.where(scale == 0, 1.0, scale)
                    design = torch.cat(
                        [
                            torch.ones(len(X), 1, dtype=X.dtype),
                            (predictors - center) / scale,
                        ],
                        1,
                    )
                    penalty = torch.eye(design.shape[1], dtype=X.dtype)
                    penalty[0, 0] = 0
                    observed_design = design[observed]
                    observed_values = X[observed, feature_index]
                    coefficients = torch.linalg.solve(
                        observed_design.T @ observed_design + penalty,
                        observed_design.T @ observed_values,
                    )
                    filled[~observed, feature_index] = (design @ coefficients)[
                        ~observed
                    ]
                    self.steps.append(
                        (
                            feature_index,
                            other_features,
                            center,
                            scale,
                            coefficients,
                        )
                    )
        return self

    def transform(self, X):
        if not hasattr(self, "width") or X.shape[1] != self.width:
            raise ValueError("Unfitted or wrong width")
        filled = torch.where(X.isnan(), self.median, X).clone()
        if self.strategy == "knn":
            scaled = (X - self.mean) / self.scale
            for row_index in range(len(X)):
                common = ~scaled[row_index].isnan() & ~self.donors.isnan()
                shared_feature_count = common.sum(1)
                distances = (
                    torch.where(
                        common, (self.donors - scaled[row_index]).square(), 0
                    ).sum(1)
                    * self.width
                    / shared_feature_count.clamp_min(1)
                )
                distances[shared_feature_count == 0] = float("inf")
                for feature_index in torch.where(X[row_index].isnan())[0].tolist():
                    eligible = ~self.donors[:, feature_index].isnan() & torch.isfinite(
                        distances
                    )
                    donor_ids = torch.where(eligible)[0]
                    if len(donor_ids):
                        donor_ids = donor_ids[
                            torch.argsort(distances[donor_ids], stable=True)[:5]
                        ]
                        filled[row_index, feature_index] = (
                            self.donors[donor_ids, feature_index].mean()
                            * self.scale[feature_index]
                            + self.mean[feature_index]
                        )
                    else:
                        filled[row_index, feature_index] = self.mean[feature_index]
        elif self.strategy == "iterative":
            for feature_index, other_features, center, scale, coefficients in self.steps:
                design = torch.cat(
                    [
                        torch.ones(len(X), 1, dtype=X.dtype),
                        (filled[:, other_features] - center) / scale,
                    ],
                    1,
                )
                missing = X[:, feature_index].isnan()
                filled[missing, feature_index] = (design @ coefficients)[missing]
        return filled


class TabularPreprocessor:
    def __init__(
        self,
        imputer="median",
        indicators=False,
        scaler="standard",
        missing_threshold=1.0,
        unknown=False,
        ordered_onehot=False,
        logs=False,
        feature_set="all",
        selection=None,
    ):
        if scaler not in ("standard", "robust", "minmax", "none"):
            raise ValueError(scaler)
        if feature_set not in ("all", "history"):
            raise ValueError(feature_set)
        if selection not in (None, "redundant", "model"):
            raise ValueError(selection)
        if not 0 <= missing_threshold <= 1:
            raise ValueError("Invalid missing threshold")
        self.imputer, self.indicators, self.scaler = imputer, indicators, scaler
        self.missing_threshold, self.unknown, self.ordered_onehot = (
            missing_threshold,
            unknown,
            ordered_onehot,
        )
        self.logs, self.feature_set, self.selection = logs, feature_set, selection

    def _check(self, X):
        X = torch.as_tensor(X, dtype=torch.float64)
        if (
            X.ndim != 2
            or X.shape[1] != len(NAMES)
            or not len(X)
            or torch.isinf(X).any()
        ):
            raise ValueError("Invalid schema tensor")
        for feature_index, name in enumerate(NAMES):
            observed = X[:, feature_index][~X[:, feature_index].isnan()]
            allowed = ORDERED.get(
                name, list(range(len(CATEGORIES[name]))) if name in CATEGORIES else None
            )
            if (
                allowed is not None
                and not torch.isin(observed, observed.new_tensor(allowed)).all()
            ):
                raise ValueError(f"Invalid {name}")
            if name in NUMERIC and (observed < 0).any():
                raise ValueError(f"Negative {name}")
        return X

    def fit(self, X, y=None):
        X = self._check(X)
        self.columns = [
            feature_index
            for feature_index, name in enumerate(NAMES)
            if (self.feature_set == "all" or name in HISTORY)
            and float(X[:, feature_index].isnan().double().mean())
            <= self.missing_threshold
        ]
        if not self.columns:
            raise ValueError("No retained features")
        for feature_index in self.columns:
            if X[:, feature_index].isnan().all():
                raise ValueError(f"Entirely missing feature: {NAMES[feature_index]}")
        self.numeric = [
            feature_index
            for feature_index in self.columns
            if NAMES[feature_index] in NUMERIC
        ]
        self.categorical = [
            feature_index
            for feature_index in self.columns
            if feature_index not in self.numeric
        ]
        self.log_columns = (
            [
                feature_index
                for feature_index in self.numeric
                if NAMES[feature_index] in ("sc", "bu", "bgr")
            ]
            if self.logs
            else []
        )
        numeric = self._numeric(X)
        self.numeric_imputer = (
            NumericImputer(self.imputer).fit(numeric) if numeric.shape[1] else None
        )
        self.modes = {}
        for feature_index in self.categorical:
            observed = X[:, feature_index][~X[:, feature_index].isnan()]
            self.modes[feature_index] = torch.mode(observed).values
        matrix, names, scaled = self._encode(X)
        self.names = names
        self.scaled = scaled
        self.center = torch.zeros(matrix.shape[1], dtype=X.dtype)
        self.scale = torch.ones_like(self.center)
        if scaled:
            scaled_features = matrix[:, scaled]
            if self.scaler == "standard":
                center = scaled_features.mean(0)
                scale = scaled_features.std(0, unbiased=False)
            elif self.scaler == "robust":
                center = scaled_features.quantile(0.5, dim=0)
                scale = scaled_features.quantile(
                    0.75, dim=0
                ) - scaled_features.quantile(0.25, dim=0)
            elif self.scaler == "minmax":
                center = scaled_features.min(0).values
                scale = scaled_features.max(0).values - center
            else:
                center = torch.zeros(scaled_features.shape[1], dtype=X.dtype)
                scale = torch.ones(
                    scaled_features.shape[1], dtype=X.dtype
                )
            self.center[scaled] = center
            self.scale[scaled] = torch.where(scale == 0, 1.0, scale)
        self.selected = list(range(len(names)))
        if self.selection == "redundant":
            # Pairwise observed Spearman ranks; ties receive average ranks.
            def rank(v):
                _, inverse_indices, counts = v.unique(
                    sorted=True, return_inverse=True, return_counts=True
                )
                ranks = counts.cumsum(0) - counts + (counts - 1) / 2
                return ranks[inverse_indices].double()

            retained = []
            for feature_index in self.numeric:
                redundant = False
                for retained_index in retained:
                    observed_together = (
                        ~X[:, feature_index].isnan()
                        & ~X[:, retained_index].isnan()
                    )
                    if observed_together.sum() < 3:
                        continue
                    feature_ranks = rank(X[observed_together, feature_index])
                    retained_ranks = rank(X[observed_together, retained_index])
                    feature_ranks -= feature_ranks.mean()
                    retained_ranks -= retained_ranks.mean()
                    denominator = feature_ranks.norm() * retained_ranks.norm()
                    if denominator > 0:
                        correlation = feature_ranks @ retained_ranks / denominator
                        if abs(float(correlation)) > 0.95:
                            redundant = True
                            break
                if not redundant:
                    retained.append(feature_index)
            removed = {
                NAMES[feature_index]
                for feature_index in self.numeric
                if feature_index not in retained
            }
            self.selected = [
                index for index, name in enumerate(names) if name not in removed
            ]
        elif self.selection == "model":
            if y is None:
                raise ValueError("Selection requires training labels")
            forest = RandomForest(seed=42).fit((matrix - self.center) / self.scale, y)
            self.selected = sorted(
                torch.argsort(forest.importances, descending=True, stable=True)[
                    : max(1, math.ceil(len(names) / 2))
                ].tolist()
            )
        self.feature_names_ = [names[index] for index in self.selected]
        return self

    def _numeric(self, X):
        numeric = X[:, self.numeric]
        if self.log_columns:
            numeric = torch.cat(
                [numeric, torch.log1p(X[:, self.log_columns])], 1
            )
        return numeric

    def _encode(self, X):
        numeric = self._numeric(X)
        if self.numeric_imputer is not None:
            numeric = self.numeric_imputer.transform(numeric)
        parts = [numeric]
        names = [NAMES[index] for index in self.numeric] + [
            f"log1p_{NAMES[index]}" for index in self.log_columns
        ]
        scaled = list(range(len(names)))
        for feature_index in self.categorical:
            name = NAMES[feature_index]
            column = X[:, feature_index]
            missing = column.isnan()
            filled = torch.where(missing, self.modes[feature_index], column)
            if name in ORDERED and not self.ordered_onehot:
                # Unknown gets a separate indicator, never an artificial ordinal value.
                parts.append(filled[:, None])
                scaled.append(len(names))
                names.append(name)
                if self.unknown:
                    parts.append(missing.double()[:, None])
                    names.append(name + "_unknown")
            elif name in ORDERED or self.unknown:
                domain = (
                    ORDERED[name]
                    if name in ORDERED
                    else list(range(len(CATEGORIES[name])))
                )
                for value in domain:
                    encoded = (filled == value) & (
                        ~missing if self.unknown else torch.ones_like(missing)
                    )
                    parts.append(encoded.double()[:, None])
                    names.append(f"{name}={value}")
                if self.unknown:
                    parts.append(missing.double()[:, None])
                    names.append(name + "_unknown")
            else:
                # All remaining source categories are binary; one 0/1 column is lossless.
                parts.append(filled[:, None])
                names.append(name)
        if self.indicators:
            parts.append(X[:, self.columns].isnan().double())
            names += [NAMES[index] + "_missing" for index in self.columns]
        return torch.cat(parts, 1), names, scaled

    def transform(self, X):
        X = self._check(X)
        if not hasattr(self, "selected"):
            raise ValueError("Preprocessor is not fitted")
        encoded, _, _ = self._encode(X)
        transformed = ((encoded - self.center) / self.scale)[:, self.selected]
        if not torch.isfinite(transformed).all():
            raise ValueError("Nonfinite transformed features")
        return transformed
