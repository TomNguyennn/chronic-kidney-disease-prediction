import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data import prepare_splits
from src.evaluate import bootstrap_intervals, metrics
from src.final_test import evaluate_final, freeze_finalist, report_saved_test
from src.models import LogisticRegression, NeuralNetwork, RandomForest
from src.plots import compare_runs, permutation_importance
from src.schema import CATEGORIES, NAMES, NUMERIC, ORDERED, load_data
from src.tabular import NumericImputer, TabularPreprocessor
from src.train import fit_pipeline, loss_configs, run_experiment, search_configs


def fixture(n=30):
    generator = torch.Generator().manual_seed(7)
    labels = (torch.arange(n) % 2).double()
    columns = []
    for name in NAMES:
        if name in NUMERIC:
            column = (
                1
                + torch.rand(n, generator=generator, dtype=torch.float64)
                + labels * 3
            )
        elif name in ORDERED:
            values = torch.tensor(ORDERED[name], dtype=torch.float64)
            column = values[
                torch.randint(len(values), (n,), generator=generator)
            ]
        else:
            column = labels.clone()
        columns.append(column)
    return torch.stack(columns, 1), labels


def csv_fixture(path, n=50):
    features, labels = fixture(n)
    with Path(path).open("w", newline="") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(NAMES + ["class"])
        for row, label in zip(features, labels):
            values = [
                CATEGORIES[name][int(value)]
                if name in CATEGORIES
                else float(value)
                for name, value in zip(NAMES, row)
            ]
            writer.writerow(values + ["ckd" if label else "notckd"])


class ImputerTests(unittest.TestCase):
    def test_knn_mean_and_fallback(self):
        train = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64
        )
        imputer = NumericImputer("knn").fit(train)
        query = torch.tensor(
            [[2.0, float("nan")], [float("nan"), float("nan")]],
            dtype=torch.float64,
        )
        result = imputer.transform(query)
        expected = torch.tensor(
            [[2.0, 20.0], [2.0, 20.0]], dtype=torch.float64
        )
        self.assertTrue(torch.equal(result, expected))
        repeated = NumericImputer("knn").fit(train)
        self.assertTrue(torch.equal(imputer.donors, repeated.donors))

    def test_ridge_closed_form(self):
        features = torch.tensor(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float64
        )
        imputer = NumericImputer("iterative").fit(features)
        feature_index, others, center, scale, coefficients = imputer.steps[0]
        design = torch.cat(
            [
                torch.ones(3, 1, dtype=features.dtype),
                (features[:, others] - center) / scale,
            ],
            1,
        )
        penalty = torch.diag(torch.tensor([0.0, 1.0], dtype=features.dtype))
        expected = torch.linalg.solve(
            design.T @ design + penalty,
            design.T @ features[:, feature_index],
        )
        self.assertTrue(torch.allclose(coefficients, expected))
        query = features.clone()
        query[0, 0] = float("nan")
        self.assertTrue(torch.isfinite(imputer.transform(query)).all())
        self.assertEqual(len(imputer.steps), 20)

    def test_imputer_errors(self):
        with self.assertRaises(ValueError):
            NumericImputer("bad")
        with self.assertRaises(ValueError):
            NumericImputer().fit(torch.full((3, 1), float("nan")))
        with self.assertRaises(ValueError):
            NumericImputer().transform(torch.ones(2, 1))


class TabularTests(unittest.TestCase):
    def test_all_variants_and_roundtrip(self):
        features, labels = fixture()
        features[0, 0] = float("nan")
        features[1, NAMES.index("sg")] = float("nan")
        original = features.clone()
        variants = [
            {},
            {"imputer": "knn"},
            {"imputer": "iterative"},
            {"indicators": True},
            {"unknown": True},
            {"ordered_onehot": True},
            {"logs": True},
            {"feature_set": "history"},
            {"selection": "redundant"},
            {"selection": "model"},
            {"scaler": "robust"},
            {"scaler": "minmax"},
            {"scaler": "none"},
            {"missing_threshold": 0.3},
        ]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                preprocessor = TabularPreprocessor(**kwargs).fit(features, labels)
                result = preprocessor.transform(features)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(result.shape[1], len(preprocessor.feature_names_))
                buffer = io.BytesIO()
                torch.save(preprocessor, buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=False)
                self.assertTrue(torch.equal(result, restored.transform(features)))
        self.assertTrue(torch.allclose(features, original, equal_nan=True))

    def test_no_leakage_and_unknown(self):
        features, labels = fixture()
        preprocessor = TabularPreprocessor(
            unknown=True, indicators=True
        ).fit(features, labels)
        center = preprocessor.center.clone()
        columns = preprocessor.columns[:]
        changed = features.clone()
        changed[:, 0] *= 100
        changed[:, NAMES.index("htn")] = float("nan")
        result = preprocessor.transform(changed)
        self.assertTrue(torch.equal(center, preprocessor.center))
        self.assertEqual(columns, preprocessor.columns)
        self.assertTrue(
            (result[:, preprocessor.feature_names_.index("htn_unknown")] == 1).all()
        )
        self.assertTrue(
            (result[:, preprocessor.feature_names_.index("htn=0")] == 0).all()
        )
        self.assertTrue(
            (result[:, preprocessor.feature_names_.index("htn=1")] == 0).all()
        )

    def test_invalid_and_empty(self):
        features, labels = fixture()
        with self.assertRaises(ValueError):
            TabularPreprocessor().transform(features)
        for value in [-1.0, float("inf")]:
            invalid = features.clone()
            invalid[0, 0] = value
            with self.assertRaises(ValueError):
                TabularPreprocessor().fit(invalid, labels)
        invalid = features.clone()
        invalid[:, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "age"):
            TabularPreprocessor().fit(invalid, labels)
        preprocessor = TabularPreprocessor(missing_threshold=0.5).fit(
            invalid, labels
        )
        self.assertNotIn("age", preprocessor.feature_names_)
        invalid = features.clone()
        invalid[0, NAMES.index("sg")] = 1.03
        with self.assertRaises(ValueError):
            TabularPreprocessor().fit(invalid, labels)

    def test_ordered_not_truncated(self):
        features, labels = fixture()
        features[:, NAMES.index("sg")] = 1.005
        features[1, NAMES.index("sg")] = 1.025
        preprocessor = TabularPreprocessor(scaler="none").fit(features, labels)
        transformed = preprocessor.transform(features)
        index = preprocessor.feature_names_.index("sg")
        self.assertAlmostEqual(float(transformed[0, index]), 1.005)
        self.assertAlmostEqual(float(transformed[1, index]), 1.025)


class ModelCorrectnessTests(unittest.TestCase):
    def test_forest_rejects_rounded_empty_split(self):
        low = torch.nextafter(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
        features = torch.tensor(
            [[low], [1.0], [low], [1.0], [low], [1.0], [low], [1.0]],
            dtype=torch.float64,
        )
        labels = torch.tensor([0.0, 1.0] * 4, dtype=torch.float64)
        forest = RandomForest(n_estimators=10, max_features=1.0, seed=9).fit(
            features, labels
        )
        self.assertTrue(torch.isfinite(forest.predict_proba(features)).all())

    def test_forest_stump_bruteforce_bootstrap(self):
        features = torch.arange(8, dtype=torch.float64)[:, None]
        labels = torch.tensor(
            [0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.float64
        )
        for weighting in [None, "balanced"]:
            forest = RandomForest(
                n_estimators=1,
                max_depth=1,
                max_features=1.0,
                seed=9,
                class_weight=weighting,
            ).fit(features, labels)
            row_ids = torch.randint(
                8, (8,), generator=torch.Generator().manual_seed(9)
            )
            weights = torch.ones(8, dtype=torch.float64)
            if weighting:
                weights = 8 / (2 * torch.bincount(labels.long())[labels.long()])
            node_weights = weights[row_ids]
            targets = labels[row_ids]
            parent_rate = (node_weights * targets).sum() / node_weights.sum()
            candidates = []
            unique_values = features[row_ids].unique(sorted=True)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            for candidate_threshold in thresholds:
                left = features[row_ids, 0] <= candidate_threshold
                residual = 0.0
                for side in [left, ~left]:
                    probability = (
                        node_weights[side] * targets[side]
                    ).sum() / node_weights[side].sum()
                    residual += float(
                        node_weights[side].sum()
                        / node_weights.sum()
                        * 2
                        * probability
                        * (1 - probability)
                    )
                candidates.append(
                    (
                        float(2 * parent_rate * (1 - parent_rate)) - residual,
                        -float(candidate_threshold),
                    )
                )
            expected_threshold = -max(candidates)[1]
            self.assertAlmostEqual(forest.trees[0][1], expected_threshold)
            left = features[row_ids, 0] <= expected_threshold
            expected_probability = (
                node_weights[left] * targets[left]
            ).sum() / node_weights[left].sum()
            self.assertAlmostEqual(
                forest.trees[0][2], float(expected_probability)
            )

    def test_forest_stops_and_mean(self):
        features, labels = fixture(12)
        features = features[:, :2]
        forest = RandomForest(n_estimators=3, min_samples_leaf=20).fit(
            features, labels
        )
        self.assertTrue(all(isinstance(tree, float) for tree in forest.trees))
        expected = torch.full(
            (12,), sum(forest.trees) / 3, dtype=features.dtype
        )
        self.assertTrue(torch.allclose(forest.predict_proba(features), expected))
        pure = RandomForest(n_estimators=2).fit(features, torch.ones(12))
        self.assertTrue((pure.predict_proba(features) == 1).all())

    def test_logistic_regularization_and_gradient(self):
        features = torch.tensor(
            [[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64
        )
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        strong = LogisticRegression(C=0.01).fit(features, labels)
        weak = LogisticRegression(C=10).fit(features, labels)
        self.assertLess(
            float(strong.net.weight.detach().norm()),
            float(weak.net.weight.detach().norm()),
        )
        self.assertTrue(weak.converged)
        self.assertFalse(
            LogisticRegression(max_iter=1).fit(features, labels).converged
        )
        weight = weak.net.weight.detach().clone().requires_grad_()
        bias = weak.net.bias.detach().clone().requires_grad_()

        def objective(candidate_weight, candidate_bias):
            logits = (
                features @ candidate_weight.T + candidate_bias
            ).flatten()
            data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.double()
            )
            regularization = candidate_weight.square().sum() / (
                2 * 10 * len(labels)
            )
            return data_loss + regularization

        self.assertTrue(torch.autograd.gradcheck(objective, (weight, bias)))

    def test_nn_inner_partition_and_refit(self):
        features, labels = fixture(30)
        config = {
            "experiment_id": "test",
            "model": "nn",
            "model_params": {"epochs": 3},
        }
        fitted = fit_pipeline(features, labels, config, 42)
        history = fitted.history
        self.assertFalse(
            set(history["inner_ids"]) & set(history["holdout_ids"])
        )
        self.assertEqual(
            sorted(history["inner_ids"] + history["holdout_ids"]),
            list(range(30)),
        )
        self.assertEqual(len(fitted.model.history), history["best_epoch"])
        expected_median = torch.nanquantile(features[:, :11], 0.5, dim=0)
        self.assertTrue(
            torch.allclose(
                fitted.preprocessor.numeric_imputer.median, expected_median
            )
        )
        self.assertEqual(
            len(history["train"]), len(history["validation"])
        )


class WorkflowTests(unittest.TestCase):
    def test_complete_synthetic_workflow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.csv"
            csv_fixture(data_path)
            split_directory = root / "split"
            results_directory = root / "results"
            manifest = prepare_splits(data_path, split_directory)
            model_configs = [
                ("dummy", {}),
                ("lr", {}),
                ("rf", {"n_estimators": 3, "max_depth": 3}),
                ("nn", {"epochs": 2}),
            ]
            for model_name, parameters in model_configs:
                config = {
                    "experiment_id": "smoke-" + model_name,
                    "model": model_name,
                    "model_params": parameters,
                }
                with patch(
                    "src.train.load_data", wraps=load_data
                ) as loader:
                    directory = run_experiment(
                        config,
                        data_path,
                        split_directory,
                        results_directory,
                    )
                    for call in loader.call_args_list:
                        self.assertFalse(
                            set(call.args[1]) & set(manifest["test"])
                        )
                summary = json.loads(
                    (directory / "summary.json").read_text()
                )
                self.assertEqual(summary["status"], "complete")
                if model_name == "lr":
                    winner = directory
            with self.assertRaises(FileExistsError):
                run_experiment(
                    config,
                    data_path,
                    split_directory,
                    results_directory,
                )
            plot_directory = root / "plots"
            compare_runs([winner], plot_directory)
            self.assertTrue((plot_directory / "precision_recall.png").exists())
            permutation_importance(
                winner, data_path, split_directory, repeats=2
            )
            final_directory = root / "final"
            freeze_finalist(winner, final_directory)
            with self.assertRaises(FileExistsError):
                freeze_finalist(winner, final_directory)
            report = evaluate_final(
                data_path,
                split_directory,
                final_directory,
                n_resamples=10,
            )
            confusion_total = sum(
                report["metrics"][key] for key in ["tp", "fp", "tn", "fn"]
            )
            self.assertEqual(confusion_total, 10)
            with self.assertRaises(FileExistsError):
                evaluate_final(
                    data_path,
                    split_directory,
                    final_directory,
                    n_resamples=10,
                )
            self.assertEqual(
                report,
                report_saved_test(final_directory, n_resamples=10),
            )
            self.assertTrue((results_directory / "runs.csv").exists())

    def test_search_and_loss_controls(self):
        for model_name, count in [("lr", 8), ("rf", 24), ("nn", 8)]:
            configs = search_configs(
                {
                    "experiment_id": "base",
                    "model": model_name,
                    "preprocessing": {"indicators": True},
                }
            )
            self.assertEqual(len(configs), count)
            self.assertTrue(
                all(
                    config["preprocessing"] == {"indicators": True}
                    for config in configs
                )
            )
        configs = loss_configs(
            {
                "experiment_id": "base",
                "model": "nn",
                "model_params": {"hidden": [32, 16]},
                "seeds": [42, 43, 44],
            }
        )
        self.assertEqual(len(configs), 4)
        for config in configs:
            self.assertEqual(config["model_params"]["hidden"], [32, 16])
            self.assertEqual(config["seeds"], [42, 43, 44])

    def test_parser_and_metadata_tamper(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "input.csv"
            csv_fixture(data_path)
            features, _, _ = load_data(data_path)
            self.assertEqual(features.shape, (50, 24))
            split_directory = Path(temporary_directory) / "splits"
            prepare_splits(data_path, split_directory)
            metadata_path = split_directory / "split_metadata.json"
            manifest = json.loads(metadata_path.read_text())
            manifest["cv_folds"][0].append(manifest["test"][0])
            metadata_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                prepare_splits(data_path, split_directory)
            data_path.write_text(
                data_path.read_text().replace("normal", "invalid", 1)
            )
            _, _, audit = load_data(data_path)
            self.assertEqual(
                sum(row["count_affected"] for row in audit), 1
            )

    def test_metrics_validation_and_intervals(self):
        with self.assertRaises(ValueError):
            metrics([2, 0], [0.3, 0.2])
        intervals = bootstrap_intervals(
            [0, 0, 1, 1],
            [0.1, 0.3, 0.6, 0.9],
            0.5,
            n_resamples=10,
        )
        self.assertEqual(intervals["f2"], [1.0, 1.0])

    def test_notebook_compiles_and_gates_default_off(self):
        notebook = json.loads(
            Path("model_exeperimental_notebook.ipynb").read_text()
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                compile(text, f"cell-{index}", "exec")
                # Executed notebooks are valid working documents.
                self.assertIn("outputs", cell)
                for line in text.splitlines():
                    if line.startswith(
                        ("RUN_", "MAKE_", "FREEZE_", "REPORT_")
                    ):
                        self.assertTrue(line.endswith("= False"))

class AdditionalGuards(unittest.TestCase):
    def test_every_nn_loss_fits(self):
        features, labels = fixture(12)
        for loss in ["bce", "weighted_bce", "mse", "focal"]:
            model = NeuralNetwork(epochs=2, loss=loss).fit(
                features,
                labels,
                validation=(features, labels),
                patience=1,
            )
            self.assertTrue(
                torch.isfinite(model.predict_proba(features)).all()
            )
            self.assertTrue(all(torch.isfinite(torch.tensor(model.history))))

    def test_failed_run_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.csv"
            csv_fixture(data_path)
            config = {
                "experiment_id": "fail",
                "model": "lr",
                "model_params": {"max_iter": 1},
            }
            with self.assertRaises(RuntimeError):
                run_experiment(
                    config,
                    data_path,
                    root / "split",
                    root / "results",
                )
            summaries = list((root / "results").glob("*/summary.json"))
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["status"], "failed")

    def test_finalist_tampering_blocks_test_access(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.csv"
            csv_fixture(data_path)
            run = run_experiment(
                {"experiment_id": "guard", "model": "lr"},
                data_path,
                root / "split",
                root / "results",
            )
            final_directory = root / "final"
            freeze_finalist(run, final_directory)
            with (final_directory / "pipeline.pt").open("ab") as pipeline_file:
                pipeline_file.write(b"changed")
            with patch("src.final_test.load_data") as loader:
                with self.assertRaisesRegex(ValueError, "pipeline changed"):
                    evaluate_final(
                        data_path,
                        root / "split",
                        final_directory,
                        10,
                    )
                loader.assert_not_called()
            self.assertFalse((final_directory / "test_started.json").exists())
