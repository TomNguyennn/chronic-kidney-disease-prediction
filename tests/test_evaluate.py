import unittest

from src.evaluate import choose_threshold, metrics


class EvaluationTests(unittest.TestCase):
    def test_baseline(self):
        result = metrics([1] * 250 + [0] * 150, [1] * 400)
        self.assertAlmostEqual(result["f2"], 1250 / 1400)
        self.assertEqual(result["specificity"], 0)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["ap"], 0.625)

    def test_threshold(self):
        self.assertIsNone(choose_threshold([0, 0], [0.2, 0.8]))
        self.assertAlmostEqual(choose_threshold([0, 1], [0.2, 0.8]), 0.8)
