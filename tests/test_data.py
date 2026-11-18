import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.convert_dataset import convert_dataset
from src.data import prepare_splits


class SplitTests(unittest.TestCase):
    def test_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "input.csv"
            with data_path.open("w") as data_file:
                writer = csv.writer(data_file)
                writer.writerow(["class"])
                writer.writerows([["ckd"]] * 50 + [["notckd"]] * 50)
            split_directory = Path(temporary_directory) / "splits"
            manifest = prepare_splits(data_path, split_directory)
            self.assertEqual(
                manifest, prepare_splits(data_path, split_directory)
            )
            self.assertEqual(
                [
                    len(manifest[part])
                    for part in ["train", "validation", "test"]
                ],
                [60, 20, 20],
            )
            self.assertEqual(
                sorted(sum(manifest["cv_folds"], [])), manifest["train"]
            )
            data_path.write_text(data_path.read_text() + "ckd\n")
            with self.assertRaises(ValueError):
                prepare_splits(data_path, split_directory)

    def test_arff_converter_matches_canonical_csv(self):
        source = Path("Chronic_Kidney_Disease/chronic_kidney_disease_full.arff")
        expected_path = Path("Chronic_Kidney_Disease/chronic_kidney_disease_full.csv")
        with tempfile.TemporaryDirectory() as temporary_directory:
            converted_path = convert_dataset(source, Path(temporary_directory) / "data.csv")
            actual = pd.read_csv(converted_path)
        expected = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(actual, expected)
