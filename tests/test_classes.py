import io
import unittest
import torch
from src.models import Dummy,LogisticRegression,RandomForest,NeuralNetwork,binary_loss
from src.preprocessing import CKDPreprocessor


class ClassTests(unittest.TestCase):
    def test_preprocessing(self):
        n=torch.tensor([[1.],[2.],[3.],[4.],[float('nan')]])
        c=torch.tensor([[0.],[1.],[0.],[1.],[float('nan')]])
        p=CKDPreprocessor(indicators=True).fit(n,c)
        self.assertEqual(p.medians.item(),2.5)
        self.assertEqual(p.modes.item(),0.)
        before=p.medians.clone()
        out=p.transform(n,c)
        self.assertEqual(out.shape,(5,4)); self.assertTrue(torch.isfinite(out).all())
        p.transform(n*100,c); self.assertTrue(torch.equal(before,p.medians))
        self.assertTrue(torch.isnan(n[-1]).all())

    def test_preprocessor_errors(self):
        n=torch.ones(3,1); c=torch.zeros(3,1)
        with self.assertRaises(ValueError): CKDPreprocessor().transform(n,c)
        with self.assertRaises(ValueError): CKDPreprocessor().fit(n*float('nan'),c)
        for scaler in ['standard','robust','minmax','none']:
            p=CKDPreprocessor(scaler=scaler).fit(n,c)
            self.assertTrue(torch.isfinite(p.transform(n,c)).all())

    def test_models(self):
        X=torch.tensor([[-3.],[-2.],[-1.],[1.],[2.],[3.]])
        y=torch.tensor([0.,0.,0.,1.,1.,1.])
        factories=[lambda:Dummy(),lambda:Dummy('always_ckd'),lambda:LogisticRegression(),lambda:RandomForest(n_estimators=5,max_depth=3),lambda:NeuralNetwork(epochs=3)]
        for factory in factories:
            with self.subTest(model=factory().__class__.__name__):
                model=factory().fit(X,y); p=model.predict_proba(X)
                self.assertTrue(((p>=0)&(p<=1)).all())
                self.assertTrue(torch.equal(p,factory().fit(X,y).predict_proba(X)))
                stream=io.BytesIO(); torch.save(model,stream); stream.seek(0)
                restored=torch.load(stream,weights_only=False)
                self.assertTrue(torch.equal(p,restored.predict_proba(X)))
                with self.assertRaises(ValueError): model.predict_proba(torch.ones(2,3))
        self.assertTrue(LogisticRegression().fit(X,y).converged)
        p=RandomForest(n_estimators=20).fit(X,y).predict_proba(X)
        self.assertTrue(torch.equal(p>=.5,y.bool()))

    def test_losses(self):
        for name in ['bce','weighted_bce','mse','focal']:
            z=torch.tensor([-1000.,0.,1000.],requires_grad=True)
            loss=binary_loss(z,torch.tensor([1.,0.,0.]),name,.5)
            loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(z.grad).all())

if __name__=='__main__': unittest.main()
