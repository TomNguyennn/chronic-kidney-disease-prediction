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
                for j in range(self.width):
                    others = [k for k in range(self.width) if k != j]
                    observed = ~X[:, j].isnan()
                    predictors = filled[:, others]
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
                    a, b = design[observed], X[observed, j]
                    coef = torch.linalg.solve(a.T @ a + penalty, a.T @ b)
                    filled[~observed, j] = (design @ coef)[~observed]
                    self.steps.append((j, others, center, scale, coef))
        return self

    def transform(self, X):
        if not hasattr(self, "width") or X.shape[1] != self.width:
            raise ValueError("Unfitted or wrong width")
        filled = torch.where(X.isnan(), self.median, X).clone()
        if self.strategy == "knn":
            scaled = (X - self.mean) / self.scale
            for i in range(len(X)):
                common = ~scaled[i].isnan() & ~self.donors.isnan()
                count = common.sum(1)
                distance = (
                    torch.where(common, (self.donors - scaled[i]).square(), 0).sum(1)
                    * self.width
                    / count.clamp_min(1)
                )
                distance[count == 0] = float("inf")
                for j in torch.where(X[i].isnan())[0].tolist():
                    eligible = ~self.donors[:, j].isnan() & torch.isfinite(distance)
                    ids = torch.where(eligible)[0]
                    if len(ids):
                        ids = ids[torch.argsort(distance[ids], stable=True)[:5]]
                        filled[i, j] = (
                            self.donors[ids, j].mean() * self.scale[j] + self.mean[j]
                        )
                    else:
                        filled[i, j] = self.mean[j]
        elif self.strategy == "iterative":
            for j, others, center, scale, coef in self.steps:
                design = torch.cat(
                    [
                        torch.ones(len(X), 1, dtype=X.dtype),
                        (filled[:, others] - center) / scale,
                    ],
                    1,
                )
                missing = X[:, j].isnan()
                filled[missing, j] = (design @ coef)[missing]
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
        for j, name in enumerate(NAMES):
            observed = X[:, j][~X[:, j].isnan()]
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
            j
            for j, n in enumerate(NAMES)
            if (self.feature_set == "all" or n in HISTORY)
            and float(X[:, j].isnan().double().mean()) <= self.missing_threshold
        ]
        if not self.columns:
            raise ValueError("No retained features")
        for j in self.columns:
            if X[:, j].isnan().all():
                raise ValueError(f"Entirely missing feature: {NAMES[j]}")
        self.numeric = [j for j in self.columns if NAMES[j] in NUMERIC]
        self.categorical = [j for j in self.columns if j not in self.numeric]
        self.log_columns = (
            [j for j in self.numeric if NAMES[j] in ("sc", "bu", "bgr")]
            if self.logs
            else []
        )
        numeric = self._numeric(X)
        self.numeric_imputer = (
            NumericImputer(self.imputer).fit(numeric) if numeric.shape[1] else None
        )
        self.modes = {}
        for j in self.categorical:
            v, c = X[:, j][~X[:, j].isnan()].unique(sorted=True, return_counts=True)
            self.modes[j] = v[c.argmax()]
        matrix, names, scaled = self._encode(X)
        self.names = names
        self.scaled = scaled
        self.center = torch.zeros(matrix.shape[1], dtype=X.dtype)
        self.scale = torch.ones_like(self.center)
        if scaled:
            z = matrix[:, scaled]
            if self.scaler == "standard":
                center, scale = z.mean(0), z.std(0, unbiased=False)
            elif self.scaler == "robust":
                center, scale = z.quantile(0.5, dim=0), z.quantile(
                    0.75, dim=0
                ) - z.quantile(0.25, dim=0)
            elif self.scaler == "minmax":
                center, scale = z.min(0).values, z.max(0).values - z.min(0).values
            else:
                center, scale = torch.zeros(z.shape[1], dtype=X.dtype), torch.ones(
                    z.shape[1], dtype=X.dtype
                )
            self.center[scaled] = center
            self.scale[scaled] = torch.where(scale == 0, 1.0, scale)
        self.selected = list(range(len(names)))
        if self.selection == "redundant":
            # Pairwise observed Spearman ranks; ties receive average ranks.
            def rank(v):
                _, inv, counts = v.unique(
                    sorted=True, return_inverse=True, return_counts=True
                )
                ranks = counts.cumsum(0) - counts + (counts - 1) / 2
                return ranks[inv].double()

            retained = []
            for j in self.numeric:
                redundant = False
                for k in retained:
                    mask = ~X[:, j].isnan() & ~X[:, k].isnan()
                    if mask.sum() < 3:
                        continue
                    a, b = rank(X[mask, j]), rank(X[mask, k])
                    a -= a.mean()
                    b -= b.mean()
                    denom = a.norm() * b.norm()
                    if denom > 0 and abs(float(a @ b / denom)) > 0.95:
                        redundant = True
                        break
                if not redundant:
                    retained.append(j)
            removed = {NAMES[j] for j in self.numeric if j not in retained}
            self.selected = [i for i, n in enumerate(names) if n not in removed]
        elif self.selection == "model":
            if y is None:
                raise ValueError("Selection requires training labels")
            forest = RandomForest(seed=42).fit((matrix - self.center) / self.scale, y)
            self.selected = sorted(
                torch.argsort(forest.importances, descending=True, stable=True)[
                    : max(1, math.ceil(len(names) / 2))
                ].tolist()
            )
        self.feature_names_ = [names[i] for i in self.selected]
        return self

    def _numeric(self, X):
        z = X[:, self.numeric]
        if self.log_columns:
            z = torch.cat([z, torch.log1p(X[:, self.log_columns])], 1)
        return z

    def _encode(self, X):
        z = self._numeric(X)
        if self.numeric_imputer is not None:
            z = self.numeric_imputer.transform(z)
        parts = [z]
        names = [NAMES[j] for j in self.numeric] + [
            f"log1p_{NAMES[j]}" for j in self.log_columns
        ]
        scaled = list(range(len(names)))
        for j in self.categorical:
            name = NAMES[j]
            col = X[:, j]
            missing = col.isnan()
            filled = torch.where(missing, self.modes[j], col)
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
            names += [NAMES[j] + "_missing" for j in self.columns]
        return torch.cat(parts, 1), names, scaled

    def transform(self, X):
        X = self._check(X)
        if not hasattr(self, "selected"):
            raise ValueError("Preprocessor is not fitted")
        z, _, _ = self._encode(X)
        z = ((z - self.center) / self.scale)[:, self.selected]
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite transformed features")
        return z
