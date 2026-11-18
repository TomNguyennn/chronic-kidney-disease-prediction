"""Training-fitted Torch median/mode preprocessing."""

import torch


class CKDPreprocessor:
    def __init__(self, indicators=False, scaler="standard", missing_threshold=1.0):
        if scaler not in ("standard", "robust", "minmax", "none"):
            raise ValueError(scaler)
        if not 0 <= missing_threshold <= 1:
            raise ValueError("Invalid missing threshold")
        self.indicators, self.scaler, self.missing_threshold = (
            indicators,
            scaler,
            missing_threshold,
        )

    def _validate(self, num, cat):
        num, cat = torch.as_tensor(num, dtype=torch.float64), torch.as_tensor(
            cat, dtype=torch.float64
        )
        if num.ndim != 2 or cat.ndim != 2 or len(num) != len(cat) or not len(num):
            raise ValueError("Invalid shapes")
        if torch.isinf(num).any() or torch.isinf(cat).any():
            raise ValueError("Infinite input")
        return num, cat

    def fit(self, num, cat):
        num, cat = self._validate(num, cat)
        self.widths = (num.shape[1], cat.shape[1])
        self.num_keep = num.isnan().double().mean(0) <= self.missing_threshold
        self.cat_keep = cat.isnan().double().mean(0) <= self.missing_threshold
        num, cat = num[:, self.num_keep], cat[:, self.cat_keep]
        if num.shape[1] + cat.shape[1] == 0:
            raise ValueError("No retained features")
        if num.isnan().all(0).any() or cat.isnan().all(0).any():
            raise ValueError("Entirely missing feature")
        self.medians = (
            torch.nanquantile(num, 0.5, dim=0)
            if num.shape[1]
            else torch.empty(0, dtype=num.dtype)
        )
        modes = []
        for column in cat.T:
            modes.append(torch.mode(column[~column.isnan()]).values)
        self.modes = torch.stack(modes) if modes else torch.empty(0, dtype=num.dtype)
        filled = torch.where(num.isnan(), self.medians, num)
        self.center = torch.zeros(num.shape[1], dtype=num.dtype)
        self.scale = torch.ones_like(self.center)
        if num.shape[1]:
            if self.scaler == "standard":
                self.center, self.scale = filled.mean(0), filled.std(0, unbiased=False)
            elif self.scaler == "robust":
                self.center = torch.quantile(filled, 0.5, dim=0)
                self.scale = torch.quantile(filled, 0.75, dim=0) - torch.quantile(
                    filled, 0.25, dim=0
                )
            elif self.scaler == "minmax":
                self.center = filled.min(0).values
                self.scale = filled.max(0).values - self.center
        self.scale = torch.where(
            self.scale == 0, torch.ones_like(self.scale), self.scale
        )
        return self

    def transform(self, num, cat):
        num, cat = self._validate(num, cat)
        if not hasattr(self, "widths") or (num.shape[1], cat.shape[1]) != self.widths:
            raise ValueError("Unfitted or incompatible features")
        num, cat = num[:, self.num_keep], cat[:, self.cat_keep]
        missing = torch.cat([num.isnan(), cat.isnan()], 1).double()
        parts = [
            (torch.where(num.isnan(), self.medians, num) - self.center) / self.scale,
            torch.where(cat.isnan(), self.modes, cat),
        ]
        if self.indicators:
            parts.append(missing)
        return torch.cat(parts, 1)
