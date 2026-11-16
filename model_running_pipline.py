from scipy.io import arff
import pandas as pd 
from pathlib import Path
from io import StringIO

text = Path(
    "Chronic_Kidney_Disease/chronic_kidney_disease_full.arff"
).read_text()

header, marker, rows = text.partition("@data")
assert marker, "Missing @data section"

field_count = sum(line.strip().lower().startswith("@attribute ") for line in header.splitlines())
clean_rows = []
# shortcut: this dataset has no quoted commas; use a CSV parser for other datasets.
for line_number, line in enumerate(rows.splitlines(), header.count("\n") + 1):
    if not line.strip() or line.lstrip().startswith("%"):
        continue
    values = [value.strip() for value in line.split(",")]
    # Three source rows contain one extra empty field.
    if len(values) == field_count + 1 and values.count("") == 1:
        values.remove("")
    if len(values) != field_count:
        raise ValueError(f"Line {line_number}: expected {field_count} fields, got {len(values)}")
    clean_rows.append(",".join(value or "?" for value in values))

data, metadata = arff.loadarff(
    StringIO(header + "@data\n" + "\n".join(clean_rows))
)

df = pd.DataFrame(data)

for column in df.select_dtypes(include="object"):
    df[column] = df[column].str.decode("utf-8")

df = df.replace("?", pd.NA)
df.to_csv("Chronic_Kidney_Disease/chronic_kidney_disease_full.csv", index=False)

