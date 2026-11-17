import csv
import tempfile
import unittest
from pathlib import Path
from src.data import prepare_splits

class SplitTests(unittest.TestCase):
    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.csv'
            with p.open('w') as f:
                w=csv.writer(f); w.writerow(['class']); w.writerows([['ckd']]*50+[['notckd']]*50)
            directory=Path(tmp)/'splits'
            s=prepare_splits(p,directory)
            self.assertEqual(s,prepare_splits(p,directory))
            self.assertEqual([len(s[k]) for k in ['train','validation','test']],[60,20,20])
            self.assertEqual(sorted(sum(s['cv_folds'],[])),s['train'])
            p.write_text(p.read_text()+'ckd\n')
            with self.assertRaises(ValueError): prepare_splits(p,directory)
