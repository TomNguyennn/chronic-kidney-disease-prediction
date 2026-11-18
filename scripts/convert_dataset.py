"""Convert the source CKD ARFF file into the canonical CSV dataset."""

import csv
import shlex
from io import StringIO
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path(
    "Chronic_Kidney_Disease/chronic_kidney_disease_full.arff"
)
DEFAULT_DESTINATION = Path(
    "Chronic_Kidney_Disease/chronic_kidney_disease_full.csv"
)


def convert_dataset(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION):
    source = Path(source)
    destination = Path(destination)
    text = source.read_text()
    header, marker, rows = text.partition("@data")
    if not marker:
        raise ValueError("Missing @data section")

    attributes = [
        shlex.split(line)
        for line in header.splitlines()
        if line.strip().lower().startswith("@attribute ")
    ]
    if not attributes or any(len(attribute) < 3 for attribute in attributes):
        raise ValueError("Invalid @attribute declaration")

    column_names = [attribute[1] for attribute in attributes]
    numeric_columns = [
        attribute[1]
        for attribute in attributes
        if attribute[2].lower() == "numeric"
    ]
    clean_rows = []
    for line_number, values in enumerate(
        csv.reader(StringIO(rows)), header.count("\n") + 1
    ):
        values = [value.strip() for value in values]
        if not values or not values[0] or values[0].startswith("%"):
            continue

        # Three source rows contain one extra empty field.
        if len(values) == len(column_names) + 1 and values.count("") == 1:
            values.remove("")
        if len(values) != len(column_names):
            raise ValueError(
                f"Line {line_number}: expected {len(column_names)} fields, "
                f"got {len(values)}"
            )
        clean_rows.append(values)

    frame = pd.DataFrame(clean_rows, columns=column_names).replace("?", pd.NA)
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


if __name__ == "__main__":
    convert_dataset()
