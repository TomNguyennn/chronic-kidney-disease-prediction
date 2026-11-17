"""Small CPU Torch classifiers; probabilities always refer to CKD=1."""
import torch
from torch import nn
from torch.nn import functional as F


def checked(X, y=None):
    X = torch.as_tensor(X, dtype=torch.float64)
    if X.ndim != 2 or not X.numel() or not torch.isfinite(X).all():
        raise ValueError('Expected a finite nonempty feature matrix')
    if y is not None:
        y = torch.as_tensor(y, dtype=torch.float64)
        if y.shape != (len(X),) or not ((y == 0) | (y == 1)).all():
            raise ValueError('Expected binary labels matching rows')
    return X, y


class Dummy:
    def __init__(self, strategy='most_frequent'):
        if strategy not in ('most_frequent', 'always_ckd'): raise ValueError(strategy)
        self.strategy = strategy

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]
        self.value = float(self.strategy == 'always_ckd' or y.mean() > .5)
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self, 'width') or X.shape[1] != self.width: raise ValueError('Unfitted or incompatible input')
        return torch.full((len(X),), self.value, dtype=X.dtype)


class LogisticRegression(Dummy):
    def __init__(self, C=1., class_weight=None, max_iter=1000):
        if C <= 0 or class_weight not in (None, 'balanced'): raise ValueError('Invalid parameters')
        self.C, self.class_weight, self.max_iter = C, class_weight, max_iter

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]
        self.net = nn.Linear(self.width, 1).double()
        nn.init.zeros_(self.net.weight); nn.init.zeros_(self.net.bias)
        weights = torch.ones_like(y)
        if self.class_weight:
            counts = torch.bincount(y.long(), minlength=2)
            if (counts == 0).any(): raise ValueError('Balanced fitting requires both classes')
            weights = len(y) / (2 * counts[y.long()])
        optimizer = torch.optim.LBFGS(self.net.parameters(), max_iter=self.max_iter, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = (F.binary_cross_entropy_with_logits(self.net(X).flatten(), y, reduction='none') * weights).sum()/weights.sum()
            loss = loss + self.net.weight.square().sum()/(2*self.C*weights.sum())
            loss.backward()
            return loss
        optimizer.step(closure)
        self.objective = float(closure().detach())
        self.gradient_norm = max(float(p.grad.abs().max()) for p in self.net.parameters())
        self.converged = self.gradient_norm < 1e-4
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self, 'net') or X.shape[1] != self.width: raise ValueError('Unfitted or incompatible input')
        with torch.no_grad(): return self.net(X).flatten().sigmoid()


class RandomForest(Dummy):
    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight=None, seed=42):
        if n_estimators < 1 or min_samples_leaf < 1 or (max_depth is not None and max_depth < 1): raise ValueError('Invalid tree size')
        if max_features != 'sqrt' and not (isinstance(max_features, (float,int)) and 0 < max_features <= 1): raise ValueError('Invalid max_features')
        if class_weight not in (None, 'balanced'): raise ValueError('Invalid weighting')
        self.n_estimators, self.max_depth, self.min_samples_leaf = n_estimators, max_depth, min_samples_leaf
        self.max_features, self.class_weight, self.seed = max_features, class_weight, seed

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]; self.trees = []; self.importances = torch.zeros(self.width)
        g = torch.Generator().manual_seed(self.seed)
        weights = torch.ones_like(y)
        if self.class_weight:
            counts = torch.bincount(y.long(), minlength=2)
            if (counts == 0).any(): raise ValueError('Both classes required')
            weights = len(y)/(2*counts[y.long()])
        k = max(1, int(self.width**.5 if self.max_features == 'sqrt' else self.width*self.max_features))
        def build(ids, depth):
            w, labels = weights[ids], y[ids]
            p = (w*labels).sum()/w.sum(); best = None; gain_best = 0.
            if p == 0 or p == 1 or (self.max_depth is not None and depth >= self.max_depth): return float(p)
            for j in sorted(torch.randperm(self.width, generator=g)[:k].tolist()):
                order = torch.argsort(X[ids,j], stable=True)
                values = X[ids[order],j]
                cumulative_w = w[order].cumsum(0)
                cumulative_y = (w[order]*labels[order]).cumsum(0)
                cuts = torch.arange(1,len(ids))
                valid = (values[:-1] < values[1:]) & (cuts >= self.min_samples_leaf) & (len(ids)-cuts >= self.min_samples_leaf)
                if not valid.any(): continue
                wl,wr = cumulative_w[:-1],w.sum()-cumulative_w[:-1]
                pl,pr = cumulative_y[:-1]/wl,(cumulative_y[-1]-cumulative_y[:-1])/wr
                gains = 2*p*(1-p)-(wl*2*pl*(1-pl)+wr*2*pr*(1-pr))/w.sum()
                gains[~valid] = -float('inf')
                index = int(gains.argmax()); gain = float(gains[index])
                threshold = float(values[index]/2+values[index+1]/2)
                left = X[ids,j] <= threshold
                if min(int(left.sum()), int((~left).sum())) < self.min_samples_leaf: continue
                if gain > gain_best + 1e-14:
                    best = (j,threshold,left); gain_best = gain
            if best is None: return float(p)
            j,t,left = best; self.importances[j] += gain_best*float(w.sum())
            return (j,t,build(ids[left],depth+1),build(ids[~left],depth+1))
        for _ in range(self.n_estimators): self.trees.append(build(torch.randint(len(X),(len(X),),generator=g),0))
        if self.importances.sum() > 0: self.importances /= self.importances.sum()
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self,'trees') or X.shape[1] != self.width: raise ValueError('Unfitted or incompatible input')
        def predict(tree, row):
            while isinstance(tree,tuple): tree = tree[2] if row[tree[0]] <= tree[1] else tree[3]
            return tree
        return torch.tensor([[predict(t,row) for t in self.trees] for row in X],dtype=X.dtype).mean(1)


def binary_loss(logits, y, loss='bce', pos_weight=1.):
    if loss == 'mse': return F.mse_loss(logits.sigmoid(),y)
    if loss == 'weighted_bce': return F.binary_cross_entropy_with_logits(logits,y,pos_weight=logits.new_tensor(pos_weight))
    bce = F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    if loss == 'focal': return ((1-torch.exp(-bce)).square()*bce).mean()
    if loss != 'bce': raise ValueError(loss)
    return bce.mean()


class NeuralNetwork(LogisticRegression):
    """Fixed-epoch estimator; the experiment runner must select epochs using training-only data."""
    def __init__(self, hidden=(16,), epochs=200, batch_size=16, weight_decay=0., loss='bce', seed=42):
        self.hidden,self.epochs,self.batch_size = hidden,epochs,batch_size
        self.weight_decay,self.loss,self.seed = weight_decay,loss,seed

    def fit(self,X,y,validation=None,patience=20):
        X,y = checked(X,y); self.width = X.shape[1]
        if self.epochs < 1 or self.batch_size < 1 or self.weight_decay < 0 or not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError('Invalid NN configuration')
        if self.loss not in ('bce','weighted_bce','mse','focal'): raise ValueError('Invalid loss')
        if validation is not None:
            vx,vy=checked(*validation)
            if vx.shape[1]!=self.width: raise ValueError('Validation width mismatch')
        if not 0 < y.sum() < len(y): raise ValueError('Both classes required')
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            layers=[]; width=self.width
            for h in self.hidden: layers.extend([nn.Linear(width,h),nn.ReLU()]); width=h
            layers.append(nn.Linear(width,1)); self.net=nn.Sequential(*layers).double()
            optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-3,weight_decay=self.weight_decay)
            self.history=[]; self.validation_history=[]
            self.best_epoch=1; best=float('inf'); stale=0
            for epoch in range(self.epochs):
                order=torch.randperm(len(X))
                for ids in order.split(self.batch_size):
                    optimizer.zero_grad(); loss=binary_loss(self.net(X[ids]).flatten(),y[ids],self.loss,float((len(y)-y.sum())/y.sum()))
                    loss.backward(); optimizer.step()
                with torch.no_grad(): self.history.append(float(binary_loss(self.net(X).flatten(),y,self.loss,float((len(y)-y.sum())/y.sum()))))
                if validation is not None:
                    with torch.no_grad(): value=float(binary_loss(self.net(vx).flatten(),vy,self.loss,float((len(y)-y.sum())/y.sum())))
                    self.validation_history.append(value)
                    if value < best - 1e-8: best=value; self.best_epoch=epoch+1; stale=0
                    else: stale+=1
                    if stale>=patience: break
                else: self.best_epoch=epoch+1
        return self
