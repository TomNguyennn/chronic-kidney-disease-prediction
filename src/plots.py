"""Plots from saved development predictions; no test loading or model fitting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .evaluate import metrics
from .schema import NAMES,load_data
from .data import DEFAULT_DATA,prepare_splits


def _save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def plot_predictions(y,p,threshold,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    y=torch.as_tensor(y); p=torch.as_tensor(p,dtype=torch.float64)
    thresholds=sorted(set([0.,1.]+p.tolist()))
    scores=[metrics(y,p,t) for t in thresholds]
    fig,ax=plt.subplots()
    for name in ['f2','precision','recall','specificity']: ax.plot(thresholds,[m[name] for m in scores],label=name)
    if threshold is not None: ax.axvline(threshold,color='black',linestyle='--',label='selected')
    ax.set(xlabel='Threshold',ylabel='Score',ylim=(0,1.05)); ax.legend(); _save(fig,out/'thresholds.png')
    fig,ax=plt.subplots(); xs=[]; ys=[]
    for lo,hi in zip(torch.linspace(0,1,6)[:-1],torch.linspace(0,1,6)[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any(): xs.append(float(p[mask].mean())); ys.append(float(y[mask].double().mean()))
    ax.plot(xs,ys,'o-',label='5 equal-width bins'); ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1)); ax.legend(); _save(fig,out/'calibration.png')
    if threshold is not None:
        m=metrics(y,p,threshold); matrix=torch.tensor([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        fig,axes=plt.subplots(1,2,figsize=(8,3))
        for ax,values,title in zip(axes,[matrix,matrix.double()/matrix.sum(1,keepdim=True).clamp_min(1)*100],['Counts','Row percentages']):
            ax.imshow(values.numpy(),cmap='Blues'); ax.set(title=title,xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1])
            for i in range(2):
                for j in range(2): ax.text(j,i,f'{values[i,j]:.1f}',ha='center')
        _save(fig,out/'confusion.png')


def compare_runs(run_dirs,out_dir='results/figures'):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pr,pa=plt.subplots(); roc,ra=plt.subplots(); cv,ca=plt.subplots(); reference=None
    distributions=[]; labels=[]; fold_table={}
    for directory in map(Path,run_dirs):
        rows=json.loads((directory/'validation_predictions.json').read_text())
        identity=[(r['row_id'],r['y']) for r in rows]
        provenance=json.loads((directory/'provenance.json').read_text())
        current=(identity,provenance['split_id'])
        if reference is not None and current!=reference: raise ValueError('Cannot compare different validation partitions')
        reference=current
        y=[r['y'] for r in rows]; p=[r['probability'] for r in rows]
        summary=json.loads((directory/'summary.json').read_text()); label=summary['run_id']
        scores=[metrics(y,p,t) for t in [float('inf')]+sorted(set(p),reverse=True)]
        pa.plot([s['recall'] for s in scores],[1.]+[s['precision'] for s in scores[1:]],label=label)
        ra.plot([1-s['specificity'] for s in scores],[s['recall'] for s in scores],label=label)
        fold_scores=json.loads((directory/'cv_metrics.json').read_text())
        fold_means=[sum(r['f2'] for r in fold_scores if r['fold']==i)/sum(r['fold']==i for r in fold_scores) for i in range(5)]
        distributions.append(fold_means); labels.append(label); fold_table[label]=fold_means
        plot_predictions(y,p,summary['threshold'],out/label)
        histories=json.loads((directory/'histories.json').read_text())
        for entry in histories:
            h=entry['history']
            if h:
                fig,ax=plt.subplots(); ax.plot(h['train'],label='inner training'); ax.plot(h['validation'],label='inner holdout')
                ax.set(xlabel='Epoch (zero-based)',ylabel='Loss',title=f"{label}, seed {entry['seed']}"); ax.legend()
                _save(fig,out/label/f"learning_{entry['seed']}.png")
    for ax in (pa,ra): ax.legend(fontsize=6)
    pa.set(xlabel='Recall',ylabel='Precision'); ra.set(xlabel='False positive rate',ylabel='True positive rate')
    ra.plot([0,1],[0,1],'--',color='gray')
    ca.boxplot(distributions,tick_labels=labels); ca.tick_params(axis='x',labelrotation=90); ca.set(ylabel='F2 at 0.5 (seed-averaged per fold)')
    _save(pr,out/'precision_recall.png'); _save(roc,out/'roc.png'); _save(cv,out/'cv_distribution.png')
    paired={}
    if labels:
        for label in labels[1:]: paired[label]=[a-b for a,b in zip(fold_table[label],fold_table[labels[0]])]
    (out/'paired_fold_deltas.json').write_text(json.dumps(dict(reference=labels[0] if labels else None,deltas=paired),indent=2))
    return fold_table


def permutation_importance(run_dir,data_path=DEFAULT_DATA,split_dir='data/splits',repeats=10):
    run_dir=Path(run_dir); bundle=torch.load(run_dir/'pipeline.pt',weights_only=False,map_location='cpu')
    manifest=prepare_splits(data_path,split_dir)
    if bundle['manifest']['split_id']!=manifest['split_id']: raise ValueError('Split mismatch')
    X,y,_=load_data(data_path,manifest['validation']); threshold=bundle['threshold']
    if threshold is None: raise ValueError('No feasible threshold')
    def score(z): return metrics(y,torch.stack([m.predict_proba(z) for m in bundle['pipelines']]).mean(0),threshold)['f2']
    baseline=score(X); generator=torch.Generator().manual_seed(42); result=[]
    for j,name in enumerate(NAMES):
        deltas=[]
        for _ in range(repeats):
            z=X.clone(); z[:,j]=z[torch.randperm(len(z),generator=generator),j]; deltas.append(baseline-score(z))
        result.append(dict(feature=name,mean=float(torch.tensor(deltas).mean()),std=float(torch.tensor(deltas).std(unbiased=False))))
    (run_dir/'permutation_importance.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(8,7)); ax.barh([r['feature'] for r in result],[r['mean'] for r in result],xerr=[r['std'] for r in result]); ax.set(xlabel='Validation F2 decrease (fixed threshold)')
    _save(fig,run_dir/'permutation_importance.png')
    return result
