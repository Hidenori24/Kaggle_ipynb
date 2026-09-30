"""LightGBM CV + reporting: AUC vs pseudo-labels and vs true labels."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.1, min_child_samples=10, verbose=-1)


def cv_report(X, tr, labels, seed=0):
    oof = np.full((len(tr), len(labels)), 0.5)
    kf = KFold(5, shuffle=True, random_state=seed)
    for j, c in enumerate(labels):
        y = tr[c].values.astype(int)
        if y.min() == y.max():
            continue
        for a, b in kf.split(X):
            oof[b, j] = lgb.LGBMClassifier(**PARAMS).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    def macro(mask):
        r = {c: roc_auc_score(tr.loc[mask, c], oof[mask, j]) for j, c in enumerate(labels)
             if tr.loc[mask, c].nunique() > 1}
        return pd.Series(r)
    allm = np.ones(len(tr), bool); tm = tr.has_true.values
    p, t = macro(allm), macro(tm)
    print(p.round(3)); print("macro AUC (vs pseudo-labels): %.4f" % p.mean())
    print("macro AUC (vs TRUE labels, n=%d): %.4f" % (tm.sum(), t.mean()))
    return oof


def fit_models(X, tr, labels):
    """Per label: a fitted LGBMClassifier, or the constant 0.5 when only one class is present."""
    models = {}
    for c in labels:
        y = tr[c].values.astype(int)
        models[c] = None if y.min() == y.max() else lgb.LGBMClassifier(**PARAMS).fit(X, y)
    return models


def predict_models(models, X, labels, ids):
    sub = pd.DataFrame({"StudyInstanceUID": ids})
    for c in labels:
        m = models.get(c)
        sub[c] = 0.5 if m is None else m.predict_proba(X)[:, 1]
    return sub


def fit_predict(Xtr, tr, Xte, labels, ids):
    return predict_models(fit_models(Xtr, tr, labels), Xte, labels, ids)
