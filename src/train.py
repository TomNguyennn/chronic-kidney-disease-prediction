"""Explicit experiment stages; no training on import or implicit test access."""
import copy
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import shutil
import time
import traceback
import warnings
import torch
from .data import prepare_splits, DEFAULT_DATA
from .schema import load_data
from .tabular import TabularPreprocessor
from .models import Dummy, LogisticRegression, RandomForest, NeuralNetwork
from .evaluate import metrics, choose_threshold


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def provenance():
    files=sorted(Path(__file__).parent.glob('*.py'))
    code_hash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    versions={}
    for name in ('torch','numpy','pandas','matplotlib','pytest'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return dict(code_hash=code_hash,code_commit=commit,environment=versions)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def model_factory(config,seed):
    params=copy.deepcopy(config.get('model_params',{})); name=config['model']
    if name=='dummy': return Dummy(**params)
    if name=='lr': return LogisticRegression(**params)
    if name=='rf': return RandomForest(seed=seed,**params)
    if name=='nn': return NeuralNetwork(seed=seed,**params)
    raise ValueError(f'Unknown model: {name}')


class FittedPipeline:
    def __init__(self,preprocessor,model): self.preprocessor,self.model=preprocessor,model
    def predict_proba(self,X): return self.model.predict_proba(self.preprocessor.transform(X))


def fit_pipeline(X,y,config,seed):
    preprocessing=copy.deepcopy(config.get('preprocessing',{}))
    preprocessing.setdefault('scaler','none' if config['model'] in ('rf','dummy') else 'standard')
    history=None
    if config['model']=='nn':
        g=torch.Generator().manual_seed(42)
        inner,holdout=[],[]
        for label in (0,1):
            ids=torch.where(y==label)[0]
            if len(ids)<3: raise ValueError('Insufficient rows for internal early stopping')
            ids=ids[torch.randperm(len(ids),generator=g)].tolist(); n=max(1,int(.2*len(ids)))
            holdout+=ids[:n]; inner+=ids[n:]
        pilot_pre=TabularPreprocessor(**preprocessing).fit(X[inner],y[inner])
        pilot=model_factory(config,seed).fit(pilot_pre.transform(X[inner]),y[inner],validation=(pilot_pre.transform(X[holdout]),y[holdout]))
        history=dict(train=pilot.history,validation=pilot.validation_history,best_epoch=pilot.best_epoch,inner_ids=inner,holdout_ids=holdout)
        config=copy.deepcopy(config); config.setdefault('model_params',{})['epochs']=pilot.best_epoch
    pre=TabularPreprocessor(**preprocessing).fit(X,y)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model=model_factory(config,seed).fit(pre.transform(X),y)
    if isinstance(model,LogisticRegression) and not isinstance(model,NeuralNetwork) and not model.converged:
        raise RuntimeError(f'Logistic regression did not converge: gradient={model.gradient_norm}')
    fitted=FittedPipeline(pre,model); fitted.history=history; fitted.warning_messages=[str(w.message) for w in captured]
    return fitted


def validate_config(config):
    if not isinstance(config.get('experiment_id'),str): raise ValueError('Missing experiment ID')
    allowed={'experiment_id','model','model_params','preprocessing','seeds','notes','parent'}
    if set(config)-allowed: raise ValueError(f'Unknown config keys: {set(config)-allowed}')
    seeds=config.get('seeds',[42,43,44] if config['model']=='nn' else [42])
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Invalid seeds')
    if config['model']=='nn' and seeds!=[42,43,44]: raise ValueError('NN experiments require fixed seeds [42,43,44]')
    model_factory(config,seeds[0]); TabularPreprocessor(**config.get('preprocessing',{}))
    return seeds


def run_experiment(config,data_path=DEFAULT_DATA,split_dir='data/splits',results_dir='results'):
    """CV and validation only; test rows are never passed to the loader."""
    seeds=validate_config(config)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    manifest=prepare_splits(data_path,split_dir)
    info=provenance(); config=copy.deepcopy(config)
    identity=json.dumps(dict(config=config,split_id=manifest['split_id'],code_hash=info['code_hash']),sort_keys=True)
    run_id=config['experiment_id']+'-'+hashlib.sha256(identity.encode()).hexdigest()[:12]
    root=Path(results_dir); root.mkdir(parents=True,exist_ok=True)
    out=root/run_id
    if out.exists(): raise FileExistsError(f'Run already exists: {out}; use saved artifacts')
    out.mkdir()
    shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(out/'config.json',config)
    write_json(out/'provenance.json',dict(info,split_id=manifest['split_id'],dataset_hash=manifest['dataset_hash']))
    start=time.monotonic()
    try:
        X,y,_=load_data(data_path,manifest['train'])
        vx,vy,_=load_data(data_path,manifest['validation'])
        positions={row:i for i,row in enumerate(manifest['train'])}
        fold_results=[]; predictions=[]; pipelines=[]; rows=[]; histories=[]; validation_seeds=[]
        for seed in seeds:
            seed_scores=[]
            for fold,ids in enumerate(manifest['cv_folds']):
                held=[positions[i] for i in ids]; fitting=[i for i in range(len(y)) if i not in set(held)]
                pipeline=fit_pipeline(X[fitting],y[fitting],config,seed)
                probabilities=pipeline.predict_proba(X[held]); scores=metrics(y[held],probabilities)
                fold_results.append(dict(seed=seed,fold=fold,**scores)); seed_scores.append(scores['f2'])
                predictions.extend(dict(seed=seed,fold=fold,row_id=rid,y=int(y[i]),probability=float(p)) for rid,i,p in zip(ids,held,probabilities))
            pipeline=fit_pipeline(X,y,config,seed); pipelines.append(pipeline)
            p=pipeline.predict_proba(vx); threshold=choose_threshold(vy,p)
            validation_seeds.extend(dict(seed=seed,row_id=i,y=int(label),probability=float(prob)) for i,label,prob in zip(manifest['validation'],vy,p))
            val=metrics(vy,p,threshold) if threshold is not None else metrics(vy,p,.5)
            sd=torch.tensor(seed_scores,dtype=torch.float64)
            rows.append(dict(experiment_id=config['experiment_id'],run_id=run_id,split_id=manifest['split_id'],dataset_hash=manifest['dataset_hash'],
                             feature_set=config.get('preprocessing',{}).get('feature_set','all'),imputer=config.get('preprocessing',{}).get('imputer','median'),
                             encoder='schema',scaler=pipeline.preprocessor.scaler,model=config['model'],model_params=json.dumps(config.get('model_params',{})),
                             loss=config.get('model_params',{}).get('loss','bce') if config['model']=='nn' else None,seed=seed,
                             cv_f2_mean=float(sd.mean()),cv_f2_std=float(sd.std(unbiased=True)),val_threshold=threshold,
                             eligible=threshold is not None,code_commit=info['code_commit'],notes=config.get('notes',''),**{'val_'+k:v for k,v in val.items()}))
            histories.append(dict(seed=seed,history=pipeline.history,features=pipeline.preprocessor.feature_names_))
        mean_p=torch.stack([p.predict_proba(vx) for p in pipelines]).mean(0)
        threshold=choose_threshold(vy,mean_p)
        bundle=dict(pipelines=pipelines,config=config,manifest=manifest,provenance=info,threshold=threshold)
        torch.save(bundle,out/'pipeline.pt')
        write_json(out/'validation_predictions.json',[dict(row_id=i,y=int(label),probability=float(p)) for i,label,p in zip(manifest['validation'],vy,mean_p)])
        write_json(out/'cv_predictions.json',predictions); write_json(out/'cv_metrics.json',fold_results); write_json(out/'histories.json',histories)
        write_json(out/'run_rows.json',rows)
        write_json(out/'validation_seed_predictions.json',validation_seeds)
        write_json(out/'warnings.json',[dict(seed=seed,warnings=p.warning_messages,objective=getattr(p.model,'objective',None),gradient_norm=getattr(p.model,'gradient_norm',None),converged=getattr(p.model,'converged',None)) for seed,p in zip(seeds,pipelines)])
        summary=dict(run_id=run_id,threshold=threshold,eligible=threshold is not None,seconds=time.monotonic()-start,
                     validation=metrics(vy,mean_p,threshold) if threshold is not None else None,
                     cv_f2_mean=sum(r['f2'] for r in fold_results)/len(fold_results),status='complete')
        write_json(out/'summary.json',summary)
        rebuild_runs(root)
        return out
    except Exception as error:
        write_json(out/'summary.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start))
        raise


def rebuild_runs(root='results'):
    root=Path(root); rows=[]
    for path in sorted(root.glob('*/run_rows.json')): rows.extend(json.loads(path.read_text()))
    if rows:
        with (root/'runs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def search_configs(base):
    """Predeclared E09 spaces; each trial remains a standalone reproducible run."""
    name=base['model']
    if name=='lr': space=dict(C=[.01,.1,1,10],class_weight=[None,'balanced'])
    elif name=='rf': space=dict(max_depth=[None,3,5,10],min_samples_leaf=[1,3,5,10],max_features=['sqrt',.5],class_weight=[None,'balanced'])
    elif name=='nn': space=dict(hidden=[[16],[32,16]],batch_size=[16,32],weight_decay=[0.,1e-4])
    else: raise ValueError('Dummy models do not have a search')
    combinations=list(itertools.product(*space.values()))
    if name=='rf': combinations=[combinations[i] for i in torch.randperm(len(combinations),generator=torch.Generator().manual_seed(42))[:24].tolist()]
    result=[]
    for i,values in enumerate(combinations):
        config=copy.deepcopy(base); config['parent']=base['experiment_id']; config['experiment_id']=f'E09-{name}-{i:02}'
        config.setdefault('model_params',{}).update(dict(zip(space,values))); result.append(config)
    return result


def loss_configs(base):
    if base['model']!='nn': raise ValueError('Loss studies require NN')
    result=[]
    for loss in ['bce','weighted_bce','mse','focal']:
        c=copy.deepcopy(base); c['parent']=base['experiment_id']; c['experiment_id']='E10-'+loss
        c.setdefault('model_params',{})['loss']=loss; result.append(c)
    return result
