"""Fixed binary metrics and constrained threshold selection."""
import torch

def metrics(y,p,threshold=.5):
    y=torch.as_tensor(y)
    if not ((y==0)|(y==1)).all(): raise ValueError('Labels must be 0/1')
    y=y.bool(); p=torch.as_tensor(p,dtype=torch.float64)
    if y.shape!=p.shape or y.ndim!=1 or not len(y) or not torch.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid predictions')
    pred=p>=threshold
    tp=int((pred&y).sum()); fp=int((pred&~y).sum()); tn=int((~pred&~y).sum()); fn=int((~pred&y).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    result=dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=precision,recall=recall,specificity=tn/(tn+fp) if tn+fp else None,f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,brier=float((p-y.double()).square().mean()))
    if y.any() and (~y).any():
        pos,neg=p[y],p[~y]
        result['roc_auc']=float(((pos[:,None]>neg).double()+.5*(pos[:,None]==neg)).mean())
        ap=0.; previous=0.
        for t in p.unique().sort(descending=True).values:
            selected=p>=t; found=float(y[selected].sum()); recall_at=found/len(pos)
            ap+=(recall_at-previous)*found/int(selected.sum()); previous=recall_at
        result['ap']=ap
    else: result.update(ap=None,roc_auc=None)
    return result

def choose_threshold(y,p,precision_floor=.8):
    p=torch.as_tensor(p,dtype=torch.float64)
    if not 0<=precision_floor<=1: raise ValueError('Invalid precision floor')
    metrics(y,p)
    candidates=[]
    for t in p.unique():
        m=metrics(y,p,float(t))
        if m['tp']+m['fp'] and m['precision']>=precision_floor: candidates.append((m['f2'],m['precision'],float(t)))
    return max(candidates)[2] if candidates else None


def bootstrap_intervals(y,p,threshold,n_resamples=2000,seed=42):
    """Conditional stratified percentile intervals; no model refitting/selection."""
    y=torch.as_tensor(y,dtype=torch.float64); p=torch.as_tensor(p,dtype=torch.float64)
    metrics(y,p,threshold)
    groups=[torch.where(y==k)[0] for k in (0,1)]
    if any(len(g)==0 for g in groups) or n_resamples<2: raise ValueError('Both classes and at least two resamples required')
    generator=torch.Generator().manual_seed(seed)
    scores=[]
    for _ in range(n_resamples):
        ids=torch.cat([g[torch.randint(len(g),(len(g),),generator=generator)] for g in groups])
        scores.append(metrics(y[ids],p[ids],threshold))
    return {key:torch.tensor([s[key] for s in scores],dtype=torch.float64).quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist()
            for key in ['f2','precision','recall','specificity','ap','roc_auc','brier']}
