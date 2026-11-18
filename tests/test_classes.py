import io
import unittest

import torch

from src.models import (
    Dummy,
    LogisticRegression,
    NeuralNetwork,
    RandomForest,
    binary_loss,
)
from src.preprocessing import CKDPreprocessor


class ClassTests(unittest.TestCase):
    def test_preprocessing(self):
        numeric = torch.tensor([[1.0], [2.0], [3.0], [4.0], [float("nan")]])
        categorical = torch.tensor([[0.0], [1.0], [0.0], [1.0], [float("nan")]])
        preprocessor = CKDPreprocessor(indicators=True).fit(numeric, categorical)
        self.assertEqual(preprocessor.medians.item(), 2.5)
        self.assertEqual(preprocessor.modes.item(), 0.0)
        median_before_transform = preprocessor.medians.clone()
        transformed = preprocessor.transform(numeric, categorical)
        self.assertEqual(transformed.shape, (5, 4))
        self.assertTrue(torch.isfinite(transformed).all())
        preprocessor.transform(numeric * 100, categorical)
        self.assertTrue(
            torch.equal(median_before_transform, preprocessor.medians)
        )
        self.assertTrue(torch.isnan(numeric[-1]).all())

    def test_preprocessor_errors(self):
        numeric = torch.ones(3, 1)
        categorical = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            CKDPreprocessor().transform(numeric, categorical)
        with self.assertRaises(ValueError):
            CKDPreprocessor().fit(numeric * float("nan"), categorical)
        for scaler in ["standard", "robust", "minmax", "none"]:
            preprocessor = CKDPreprocessor(scaler=scaler).fit(
                numeric, categorical
            )
            self.assertTrue(
                torch.isfinite(
                    preprocessor.transform(numeric, categorical)
                ).all()
            )

    def test_models(self):
        features = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        factories = [
            lambda: Dummy(),
            lambda: Dummy("always_ckd"),
            lambda: LogisticRegression(),
            lambda: RandomForest(n_estimators=5, max_depth=3),
            lambda: NeuralNetwork(epochs=3),
        ]
        for factory in factories:
            with self.subTest(model=factory().__class__.__name__):
                model = factory().fit(features, labels)
                probabilities = model.predict_proba(features)
                self.assertTrue(
                    ((probabilities >= 0) & (probabilities <= 1)).all()
                )
                repeated = factory().fit(features, labels).predict_proba(features)
                self.assertTrue(torch.equal(probabilities, repeated))
                stream = io.BytesIO()
                torch.save(model, stream)
                stream.seek(0)
                restored = torch.load(stream, weights_only=False)
                self.assertTrue(
                    torch.equal(
                        probabilities, restored.predict_proba(features)
                    )
                )
                with self.assertRaises(ValueError):
                    model.predict_proba(torch.ones(2, 3))
        self.assertTrue(LogisticRegression().fit(features, labels).converged)
        probabilities = RandomForest(n_estimators=20).fit(
            features, labels
        ).predict_proba(features)
        self.assertTrue(torch.equal(probabilities >= 0.5, labels.bool()))

    def test_losses(self):
        for loss_name in ["bce", "weighted_bce", "mse", "focal"]:
            logits = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
            loss = binary_loss(
                logits, torch.tensor([1.0, 0.0, 0.0]), loss_name, 0.5
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
