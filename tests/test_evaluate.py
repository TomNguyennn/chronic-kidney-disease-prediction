import unittest
from src.evaluate import metrics,choose_threshold

class EvaluationTests(unittest.TestCase):
    def test_baseline(self):
        m=metrics([1]*250+[0]*150,[1]*400)
        self.assertAlmostEqual(m['f2'],1250/1400)
        self.assertEqual(m['specificity'],0)
        self.assertEqual(m['roc_auc'],.5)
        self.assertEqual(m['ap'],.625)
    def test_threshold(self):
        self.assertIsNone(choose_threshold([0,0],[.2,.8]))
        self.assertAlmostEqual(choose_threshold([0,1],[.2,.8]),.8)
