import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.schema import NAMES, NUMERIC, ORDERED, CATEGORIES, load_data
from src.tabular import NumericImputer, TabularPreprocessor
from src.models import RandomForest,LogisticRegression,NeuralNetwork
from src.train import run_experiment,fit_pipeline,search_configs,loss_configs
from src.data import prepare_splits
from src.final_test import freeze_finalist,evaluate_final,report_saved_test
from src.plots import compare_runs,permutation_importance
from src.evaluate import metrics,bootstrap_intervals


def fixture(n=30):
    g=torch.Generator().manual_seed(7)
    y=(torch.arange(n)%2).double()
    cols=[]
    for name in NAMES:
        if name in NUMERIC: col=1+torch.rand(n,generator=g,dtype=torch.float64)+y*3
        elif name in ORDERED:
            values=torch.tensor(ORDERED[name],dtype=torch.float64)
            col=values[torch.randint(len(values),(n,),generator=g)]
        else: col=y.clone()
        cols.append(col)
    return torch.stack(cols,1),y


def csv_fixture(path,n=50):
    X,y=fixture(n)
    with Path(path).open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(NAMES+['class'])
        for row,label in zip(X,y):
            writer.writerow([CATEGORIES[name][int(v)] if name in CATEGORIES else float(v) for name,v in zip(NAMES,row)]+['ckd' if label else 'notckd'])


class ImputerTests(unittest.TestCase):
    def test_knn_mean_and_fallback(self):
        train=torch.tensor([[1.,10.],[2.,20.],[3.,30.]],dtype=torch.float64)
        imp=NumericImputer('knn').fit(train)
        result=imp.transform(torch.tensor([[2.,float('nan')],[float('nan'),float('nan')]],dtype=torch.float64))
        self.assertTrue(torch.equal(result,torch.tensor([[2.,20.],[2.,20.]],dtype=torch.float64)))
        self.assertTrue(torch.equal(imp.donors,NumericImputer('knn').fit(train).donors))

    def test_ridge_closed_form(self):
        X=torch.tensor([[1.,2.],[2.,4.],[3.,6.]],dtype=torch.float64)
        imp=NumericImputer('iterative').fit(X)
        j,others,center,scale,coef=imp.steps[0]
        design=torch.cat([torch.ones(3,1,dtype=X.dtype),(X[:,others]-center)/scale],1)
        expected=torch.linalg.solve(design.T@design+torch.diag(torch.tensor([0.,1.],dtype=X.dtype)),design.T@X[:,0])
        self.assertTrue(torch.allclose(coef,expected))
        query=X.clone(); query[0,0]=float('nan')
        self.assertTrue(torch.isfinite(imp.transform(query)).all())
        self.assertEqual(len(imp.steps),20)

    def test_imputer_errors(self):
        with self.assertRaises(ValueError): NumericImputer('bad')
        with self.assertRaises(ValueError): NumericImputer().fit(torch.full((3,1),float('nan')))
        with self.assertRaises(ValueError): NumericImputer().transform(torch.ones(2,1))


class TabularTests(unittest.TestCase):
    def test_all_variants_and_roundtrip(self):
        X,y=fixture(); X[0,0]=float('nan'); X[1,NAMES.index('sg')]=float('nan')
        original=X.clone()
        variants=[{}, {'imputer':'knn'},{'imputer':'iterative'},{'indicators':True},{'unknown':True},
                  {'ordered_onehot':True},{'logs':True},{'feature_set':'history'}, {'selection':'redundant'},
                  {'selection':'model'}, {'scaler':'robust'},{'scaler':'minmax'},{'scaler':'none'},{'missing_threshold':.3}]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                pre=TabularPreprocessor(**kwargs).fit(X,y); result=pre.transform(X)
                self.assertTrue(torch.isfinite(result).all()); self.assertEqual(result.shape[1],len(pre.feature_names_))
                buf=io.BytesIO(); torch.save(pre,buf); buf.seek(0)
                restored=torch.load(buf,weights_only=False)
                self.assertTrue(torch.equal(result,restored.transform(X)))
        self.assertTrue(torch.allclose(X,original,equal_nan=True))

    def test_no_leakage_and_unknown(self):
        X,y=fixture(); pre=TabularPreprocessor(unknown=True,indicators=True).fit(X,y)
        center=pre.center.clone(); columns=pre.columns[:]
        changed=X.clone(); changed[:,0]*=100; changed[:,NAMES.index('htn')]=float('nan')
        result=pre.transform(changed)
        self.assertTrue(torch.equal(center,pre.center)); self.assertEqual(columns,pre.columns)
        self.assertTrue((result[:,pre.feature_names_.index('htn_unknown')]==1).all())
        self.assertTrue((result[:,pre.feature_names_.index('htn=0')]==0).all())
        self.assertTrue((result[:,pre.feature_names_.index('htn=1')]==0).all())

    def test_invalid_and_empty(self):
        X,y=fixture()
        with self.assertRaises(ValueError): TabularPreprocessor().transform(X)
        for value in [-1.,float('inf')]:
            bad=X.clone(); bad[0,0]=value
            with self.assertRaises(ValueError): TabularPreprocessor().fit(bad,y)
        bad=X.clone(); bad[:,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'age'): TabularPreprocessor().fit(bad,y)
        pre=TabularPreprocessor(missing_threshold=.5).fit(bad,y)
        self.assertNotIn('age',pre.feature_names_)
        bad=X.clone(); bad[0,NAMES.index('sg')]=1.03
        with self.assertRaises(ValueError): TabularPreprocessor().fit(bad,y)

    def test_ordered_not_truncated(self):
        X,y=fixture(); X[:,NAMES.index('sg')]=1.005; X[1,NAMES.index('sg')]=1.025
        pre=TabularPreprocessor(scaler='none').fit(X,y); out=pre.transform(X)
        index=pre.feature_names_.index('sg')
        self.assertAlmostEqual(float(out[0,index]),1.005); self.assertAlmostEqual(float(out[1,index]),1.025)


class ModelCorrectnessTests(unittest.TestCase):
    def test_forest_rejects_rounded_empty_split(self):
        low=torch.nextafter(torch.tensor(1.,dtype=torch.float64),torch.tensor(0.,dtype=torch.float64))
        X=torch.tensor([[low],[1.],[low],[1.],[low],[1.],[low],[1.]],dtype=torch.float64)
        y=torch.tensor([0.,1.]*4,dtype=torch.float64)
        forest=RandomForest(n_estimators=10,max_features=1.,seed=9).fit(X,y)
        self.assertTrue(torch.isfinite(forest.predict_proba(X)).all())

    def test_forest_stump_bruteforce_bootstrap(self):
        X=torch.arange(8,dtype=torch.float64)[:,None]; y=torch.tensor([0,1,0,0,1,0,0,1],dtype=torch.float64)
        for weighting in [None,'balanced']:
            forest=RandomForest(n_estimators=1,max_depth=1,max_features=1.,seed=9,class_weight=weighting).fit(X,y)
            ids=torch.randint(8,(8,),generator=torch.Generator().manual_seed(9))
            weights=torch.ones(8,dtype=torch.float64)
            if weighting: weights=8/(2*torch.bincount(y.long())[y.long()])
            w=weights[ids]; target=y[ids]; parent=(w*target).sum()/w.sum()
            candidates=[]
            for t in (X[ids].unique(sorted=True)[:-1]+X[ids].unique(sorted=True)[1:])/2:
                mask=X[ids,0]<=t; residual=0.
                for side in [mask,~mask]:
                    probability=(w[side]*target[side]).sum()/w[side].sum()
                    residual+=float(w[side].sum()/w.sum()*2*probability*(1-probability))
                candidates.append((float(2*parent*(1-parent))-residual,-float(t)))
            threshold=-max(candidates)[1]
            self.assertAlmostEqual(forest.trees[0][1],threshold)
            left=X[ids,0]<=threshold
            self.assertAlmostEqual(forest.trees[0][2],float((w[left]*target[left]).sum()/w[left].sum()))

    def test_forest_stops_and_mean(self):
        X,y=fixture(12); X=X[:,:2]
        forest=RandomForest(n_estimators=3,min_samples_leaf=20).fit(X,y)
        self.assertTrue(all(isinstance(t,float) for t in forest.trees))
        self.assertTrue(torch.allclose(forest.predict_proba(X),torch.full((12,),sum(forest.trees)/3,dtype=X.dtype)))
        pure=RandomForest(n_estimators=2).fit(X,torch.ones(12))
        self.assertTrue((pure.predict_proba(X)==1).all())

    def test_logistic_regularization_and_gradient(self):
        X=torch.tensor([[-2.],[-1.],[1.],[2.]],dtype=torch.float64); y=torch.tensor([0.,0.,1.,1.])
        strong=LogisticRegression(C=.01).fit(X,y); weak=LogisticRegression(C=10).fit(X,y)
        self.assertLess(float(strong.net.weight.detach().norm()),float(weak.net.weight.detach().norm()))
        self.assertTrue(weak.converged)
        self.assertFalse(LogisticRegression(max_iter=1).fit(X,y).converged)
        weight=weak.net.weight.detach().clone().requires_grad_(); bias=weak.net.bias.detach().clone().requires_grad_()
        def objective(w,b):
            return torch.nn.functional.binary_cross_entropy_with_logits((X@w.T+b).flatten(),y.double())+w.square().sum()/(2*10*len(y))
        self.assertTrue(torch.autograd.gradcheck(objective,(weight,bias)))

    def test_nn_inner_partition_and_refit(self):
        X,y=fixture(30)
        config=dict(experiment_id='test',model='nn',model_params={'epochs':3})
        fitted=fit_pipeline(X,y,config,42); h=fitted.history
        self.assertFalse(set(h['inner_ids'])&set(h['holdout_ids']))
        self.assertEqual(sorted(h['inner_ids']+h['holdout_ids']),list(range(30)))
        self.assertEqual(len(fitted.model.history),h['best_epoch'])
        self.assertTrue(torch.allclose(fitted.preprocessor.numeric_imputer.median,torch.nanquantile(X[:,:11],.5,dim=0)))
        self.assertEqual(len(h['train']),len(h['validation']))


class WorkflowTests(unittest.TestCase):
    def test_complete_synthetic_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data.csv'; csv_fixture(data)
            split=root/'split'; results=root/'results'; manifest=prepare_splits(data,split)
            for model,params in [('dummy',{}),('lr',{}),('rf',{'n_estimators':3,'max_depth':3}),('nn',{'epochs':2})]:
                config=dict(experiment_id='smoke-'+model,model=model,model_params=params)
                with patch('src.train.load_data',wraps=load_data) as loader:
                    directory=run_experiment(config,data,split,results)
                    for call in loader.call_args_list:
                        self.assertFalse(set(call.args[1])&set(manifest['test']))
                self.assertEqual(json.loads((directory/'summary.json').read_text())['status'],'complete')
                if model=='lr': winner=directory
            with self.assertRaises(FileExistsError): run_experiment(config,data,split,results)
            compare_runs([winner],root/'plots')
            self.assertTrue((root/'plots/precision_recall.png').exists())
            permutation_importance(winner,data,split,repeats=2)
            final=root/'final'; freeze_finalist(winner,final)
            with self.assertRaises(FileExistsError): freeze_finalist(winner,final)
            report=evaluate_final(data,split,final,n_resamples=10)
            self.assertEqual(sum(report['metrics'][k] for k in ['tp','fp','tn','fn']),10)
            with self.assertRaises(FileExistsError): evaluate_final(data,split,final,n_resamples=10)
            self.assertEqual(report,report_saved_test(final,n_resamples=10))
            self.assertTrue((results/'runs.csv').exists())

    def test_search_and_loss_controls(self):
        for model,count in [('lr',8),('rf',24),('nn',8)]:
            configs=search_configs(dict(experiment_id='base',model=model,preprocessing={'indicators':True}))
            self.assertEqual(len(configs),count)
            self.assertTrue(all(c['preprocessing']=={'indicators':True} for c in configs))
        configs=loss_configs(dict(experiment_id='base',model='nn',model_params={'hidden':[32,16]},seeds=[42,43,44]))
        self.assertEqual(len(configs),4)
        for c in configs: self.assertEqual(c['model_params']['hidden'],[32,16]); self.assertEqual(c['seeds'],[42,43,44])

    def test_parser_and_metadata_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.csv'; csv_fixture(p)
            X,y,audit=load_data(p); self.assertEqual(X.shape,(50,24))
            split=Path(tmp)/'splits'; prepare_splits(p,split)
            path=split/'split_metadata.json'; m=json.loads(path.read_text()); m['cv_folds'][0].append(m['test'][0]); path.write_text(json.dumps(m))
            with self.assertRaises(ValueError): prepare_splits(p,split)
            p.write_text(p.read_text().replace('normal','invalid',1))
            _,_,audit=load_data(p)
            self.assertEqual(sum(r['count_affected'] for r in audit),1)

    def test_metrics_validation_and_intervals(self):
        with self.assertRaises(ValueError): metrics([2,0],[.3,.2])
        a=bootstrap_intervals([0,0,1,1],[.1,.3,.6,.9],.5,n_resamples=10)
        self.assertEqual(a['f2'],[1.,1.])

    def test_notebook_compiles_and_gates_default_off(self):
        notebook=json.loads(Path('model_exeperimental_notebook.ipynb').read_text())
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                text=''.join(cell['source']); compile(text,f'cell-{i}','exec')
                self.assertIn('outputs', cell)  # Executed notebooks are valid working documents.
                for line in text.splitlines():
                    if line.startswith(('RUN_','MAKE_','FREEZE_','REPORT_')): self.assertTrue(line.endswith('= False'))

class AdditionalGuards(unittest.TestCase):
    def test_every_nn_loss_fits(self):
        X,y=fixture(12)
        for loss in ['bce','weighted_bce','mse','focal']:
            model=NeuralNetwork(epochs=2,loss=loss).fit(X,y,validation=(X,y),patience=1)
            self.assertTrue(torch.isfinite(model.predict_proba(X)).all())
            self.assertTrue(all(torch.isfinite(torch.tensor(model.history))))

    def test_failed_run_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'data.csv'; csv_fixture(path)
            config=dict(experiment_id='fail',model='lr',model_params={'max_iter':1})
            with self.assertRaises(RuntimeError): run_experiment(config,path,root/'split',root/'results')
            summaries=list((root/'results').glob('*/summary.json'))
            self.assertEqual(json.loads(summaries[0].read_text())['status'],'failed')

    def test_finalist_tampering_blocks_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'data.csv'; csv_fixture(path)
            run=run_experiment(dict(experiment_id='guard',model='lr'),path,root/'split',root/'results')
            final=root/'final'; freeze_finalist(run,final)
            with (final/'pipeline.pt').open('ab') as f: f.write(b'changed')
            with patch('src.final_test.load_data') as loader:
                with self.assertRaisesRegex(ValueError,'pipeline changed'): evaluate_final(path,root/'split',final,10)
                loader.assert_not_called()
            self.assertFalse((final/'test_started.json').exists())
