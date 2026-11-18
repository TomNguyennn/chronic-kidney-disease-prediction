"""Fixed source semantics; parsing never learns from another row."""

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
    raw_data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(raw_data.columns) != set(NAMES + ["class"]):
        raise ValueError("CSV must match the documented CKD schema")
    if row_ids is not None:
        raw_data = raw_data.iloc[row_ids]
    feature_values = []
    audit = []
    for name in NAMES:
        text_values = raw_data[name].str.strip()
        missing = text_values.isin(["", "?", "nan", "NaN"])
        if name in CATEGORIES:
            category_codes = {
                value: index for index, value in enumerate(CATEGORIES[name])
            }
            column = text_values.map(category_codes).astype(float)
        else:
            column = pd.to_numeric(
                text_values.mask(missing), errors="coerce"
            ).astype(float)
            if name in ORDERED:
                column = column.where(column.isin(ORDERED[name]))
            else:
                column = column.where((column >= 0) & (column < float("inf")))
        invalid = ~missing & column.isna()
        audit.append(
            dict(
                feature=name,
                condition="parse/domain/nonnegative check",
                rationale="Local source schema; negative measurements are invalid",
                action="mark missing",
                count_affected=int(invalid.sum()),
                applied=True,
                suspicious_tokens=sorted(text_values[invalid].unique().tolist()),
                missing_count=int(column.isna().sum()),
                missing_rate=float(column.isna().mean()),
            )
        )
        feature_values.append(torch.tensor(column.to_numpy(), dtype=torch.float64))
    labels = raw_data["class"].str.strip().map({"ckd": 1, "notckd": 0})
    if labels.isna().any():
        raise ValueError("Missing or unknown target")
    return (
        torch.stack(feature_values, 1),
        torch.tensor(labels.to_numpy(), dtype=torch.float64),
        audit,
    )
