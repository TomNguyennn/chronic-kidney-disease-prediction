"""Explicit one-shot test stage. Reporting can be repeated from cached predictions."""
import json
from pathlib import Path
import shutil
import torch
from .data import DEFAULT_DATA,prepare_splits
from .schema import load_data
from .train import digest,write_json,provenance
from .evaluate import metrics,bootstrap_intervals


def freeze_finalist(run_dir,final_dir='results/final_test'):
    run_dir,final_dir=Path(run_dir),Path(final_dir)
    summary=json.loads((run_dir/'summary.json').read_text())
    if summary.get('status')!='complete' or not summary.get('eligible'): raise ValueError('Finalist must have a feasible validation threshold')
    if final_dir.exists(): raise FileExistsError('Finalist already frozen; no replacement in this protocol')
    bundle=torch.load(run_dir/'pipeline.pt',weights_only=False,map_location='cpu')
    if bundle['threshold']!=summary['threshold']: raise ValueError('Inconsistent run artifacts')
    final_dir.mkdir(parents=True)
    shutil.copy2(run_dir/'pipeline.pt',final_dir/'pipeline.pt')
    write_json(final_dir/'selection.json',dict(run_id=summary['run_id'],pipeline_hash=digest(final_dir/'pipeline.pt'),
                                            threshold=bundle['threshold'],dataset_hash=bundle['manifest']['dataset_hash'],
                                            split_id=bundle['manifest']['split_id'],code_hash=bundle['provenance']['code_hash']))
    return final_dir


def evaluate_final(data_path=DEFAULT_DATA,split_dir='data/splits',final_dir='results/final_test',n_resamples=2000):
    final_dir=Path(final_dir); selection=json.loads((final_dir/'selection.json').read_text())
    manifest=prepare_splits(data_path,split_dir)
    if selection['dataset_hash']!=manifest['dataset_hash'] or selection['split_id']!=manifest['split_id']: raise ValueError('Dataset/split changed')
    if digest(final_dir/'pipeline.pt')!=selection['pipeline_hash']: raise ValueError('Frozen pipeline changed')
    if provenance()['code_hash']!=selection['code_hash']: raise ValueError('Code changed since fitting; final-test protocol requires the recorded code version')
    # Exclusive marker protects against concurrent or accidental repeat test access.
    with (final_dir/'test_started.json').open('x') as f: json.dump(selection,f)
    bundle=torch.load(final_dir/'pipeline.pt',weights_only=False,map_location='cpu')
    X,y,_=load_data(data_path,manifest['test'])
    p=torch.stack([pipeline.predict_proba(X) for pipeline in bundle['pipelines']]).mean(0)
    write_json(final_dir/'predictions.json',[dict(row_id=i,y=int(label),probability=float(prob)) for i,label,prob in zip(manifest['test'],y,p)])
    return report_saved_test(final_dir,n_resamples)


def report_saved_test(final_dir='results/final_test',n_resamples=2000):
    final_dir=Path(final_dir)
    selection=json.loads((final_dir/'selection.json').read_text())
    rows=json.loads((final_dir/'predictions.json').read_text())
    y=[r['y'] for r in rows]; p=[r['probability'] for r in rows]
    report=dict(selection=selection,metrics=metrics(y,p,selection['threshold']),
                intervals=bootstrap_intervals(y,p,selection['threshold'],n_resamples),n_resamples=n_resamples,
                limitations=['Full dataset examined in prior EDA','No verified patient or temporal identifiers',
                             'Three unverified ARFF repairs','Intervals condition on selected pipeline and class counts; they exclude selection and training uncertainty'])
    write_json(final_dir/'report.json',report)
    return report
