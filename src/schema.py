"""Fixed source semantics; parsing never learns from another row."""

from pathlib import Path
import json
import pandas as pd
import torch

NUMERIC = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
ORDERED = {
    "sg": [1.005, 1.010, 1.015, 1.020, 1.025],
    "al": list(range(6)),
    "su": list(range(6)),
}
CATEGORIES = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["notpresent", "present"],
    "ba": ["notpresent", "present"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "cad": ["no", "yes"],
    "appet": ["good", "poor"],
    "pe": ["no", "yes"],
    "ane": ["no", "yes"],
}
NAMES = NUMERIC + list(ORDERED) + list(CATEGORIES)
HISTORY = ["age", "bp", "htn", "dm", "cad", "appet", "pe"]


def load_data(path, row_ids=None):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(raw.columns) != set(NAMES + ["class"]):
        raise ValueError("CSV must match the documented CKD schema")
    if row_ids is not None:
        raw = raw.iloc[row_ids]
    values, audit = [], []
    for name in NAMES:
        text = raw[name].str.strip()
        missing = text.isin(["", "?", "nan", "NaN"])
        if name in CATEGORIES:
            col = text.map({v: i for i, v in enumerate(CATEGORIES[name])}).astype(float)
        else:
            col = pd.to_numeric(text.mask(missing), errors="coerce").astype(float)
            if name in ORDERED:
                col = col.where(col.isin(ORDERED[name]))
            else:
                col = col.where((col >= 0) & (col < float("inf")))
        invalid = ~missing & col.isna()
        audit.append(
            dict(
                feature=name,
                condition="parse/domain/nonnegative check",
                rationale="Local source schema; negative measurements are invalid",
                action="mark missing",
                count_affected=int(invalid.sum()),
                applied=True,
                suspicious_tokens=sorted(text[invalid].unique().tolist()),
                missing_count=int(col.isna().sum()),
                missing_rate=float(col.isna().mean()),
            )
        )
        values.append(torch.tensor(col.to_numpy(), dtype=torch.float64))
    target = raw["class"].str.strip().map({"ckd": 1, "notckd": 0})
    if target.isna().any():
        raise ValueError("Missing or unknown target")
    return (
        torch.stack(values, 1),
        torch.tensor(target.to_numpy(), dtype=torch.float64),
        audit,
    )
