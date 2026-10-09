"""Bridge model: 04_vol_model.py's volatility forecast + volhackv1.py's extra features.

Same pseudo-cases and the same time-ordered folds are used for every model so the
numbers are comparable. Writes submission_bridge.csv (never touches submission.csv
or dataset.csv).
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

EPS = 1e-6
ts0, r = load()
cum = Cum(r)
n = len(r)
L = lambda x: np.log(np.asarray(x) + EPS)

# ---------- precomputed series ----------
ok = ~np.isnan(r)
bins = (((np.arange(n) * 60 + ts0) % 86400) // 1800).astype(int)
prof = np.bincount(bins[ok], weights=np.abs(r[ok]), minlength=48) / np.bincount(bins[ok], minlength=48)
lprof = np.log(prof)
tod = lambda i: lprof[(((np.asarray(i) * 60 + ts0) % 86400) // 1800).astype(int)]


def ewma_fwd(x, lam):
    z = np.where(np.isnan(x), 0.0, x)
    o = (~np.isnan(x)).astype(float)
    return lfilter([1], [1, -lam], z) / np.maximum(lfilter([1], [1, -lam], o), 1e-12)


ewma_bwd = lambda x, lam: ewma_fwd(x[::-1], lam)[::-1]
ab, sq = np.abs(r), r ** 2
EA = {"97f": ewma_fwd(ab, .97), "99f": ewma_fwd(ab, .99),
      "97b": ewma_bwd(ab, .97), "99b": ewma_bwd(ab, .99)}
RMS97 = np.sqrt(ewma_fwd(sq, .97))              # volhackv1's ewma_vol
prod = np.full(n, np.nan)
prod[:-1] = r[:-1] * r[1:]
cprod = Cum(prod)


# ---------- features ----------
def har_features(i0):
    i0 = np.asarray(i0)
    F = {"in15": L(cum.mabs(i0 + 1, i0 + 15)), "rms_in": L(cum.vol(i0 + 1, i0 + 15)),
         "in5": L(cum.mabs(i0 + 11, i0 + 15))}
    for w in (15, 60, 240, 1440):
        F[f"b{w}"] = L(cum.mabs(i0 - w + 1, i0))
        F[f"a{w}"] = L(cum.mabs(i0 + RESUME, i0 + RESUME + w - 1))
    F["rms_b60"] = L(cum.vol(i0 - 59, i0))
    F["rms_a60"] = L(cum.vol(i0 + RESUME, i0 + RESUME + 59))
    F["tod_B"], F["tod_in"] = tod(i0 + 23), tod(i0 + 8)
    F["tod_b"], F["tod_a"] = tod(i0 - 30), tod(i0 + 130)
    # new: EWMA vol states on both sides of the gap
    F["e97f"], F["e99f"] = L(EA["97f"][i0]), L(EA["99f"][i0])
    F["e97b"] = L(EA["97b"][np.minimum(i0 + RESUME, n - 1)])
    F["e99b"] = L(EA["99b"][np.minimum(i0 + RESUME, n - 1)])
    return pd.DataFrame(F)


def shape_features(i0, sig):
    """Scale-free window-shape / momentum features, in units of the forecast sigma."""
    i0 = np.asarray(i0)
    W = r[i0[:, None] + np.arange(1, 16)[None, :]]
    cs = np.cumsum(W, axis=1)
    rng = cs.max(1) - cs.min(1)
    pos = np.where(rng > 0, (cs[:, -1] - cs.min(1)) / np.where(rng > 0, rng, 1), 0.5)
    ac1 = cprod.sum(i0 + 1, i0 + 14) / np.maximum(cum.sum2(i0 + 1, i0 + 15), 1e-18)
    return pd.DataFrame(dict(
        ac1=np.clip(ac1, -1, 1), range_pos=pos,
        early=cum.sum(i0 + 1, i0 + 5) / (sig * np.sqrt(5)),
        late=cum.sum(i0 + 11, i0 + 15) / (sig * np.sqrt(5)),
        mom60=cum.sum(i0 - 59, i0) / (sig * np.sqrt(60)),
        mom240=cum.sum(i0 - 239, i0) / (sig * np.sqrt(240)),
    )).clip(-30, 30)


def friend_features(i0):
    """Emulation of volhackv1.FEATURE_COLS (history up to and including minute 15)."""
    i0 = np.asarray(i0)
    e = i0 + 15
    W = r[i0[:, None] + np.arange(1, 16)[None, :]]
    cs = np.cumsum(W, axis=1)
    rng = cs.max(1) - cs.min(1)
    pos = np.where(rng > 0, (cs[:, -1] - cs.min(1)) / np.where(rng > 0, rng, 1), 0.5)
    ac1 = cprod.sum(i0 + 1, i0 + 14) / np.maximum(cum.sum2(i0 + 1, i0 + 15), 1e-18)
    hod = ((i0 * 60 + ts0) // 3600) % 24
    ang = 2 * np.pi * hod / 24
    return pd.DataFrame(dict(
        g=cum.sum(i0 + 1, e), rv=np.sqrt(15) * cum.vol(i0 + 1, e), ac1=np.clip(ac1, -1, 1),
        range_pos=pos, rv_1h=cum.vol(e - 59, e), rv_1d=cum.vol(e - 1439, e),
        ewma_vol=RMS97[e], mom_1h=cum.sum(e - 59, e), hod_sin=np.sin(ang), hod_cos=np.cos(ang)))


def sig_stage2(A, Lh, extra=None):
    z = A / (np.exp(Lh) * np.sqrt(VIS))
    d = pd.DataFrame(dict(z=np.clip(z, -60, 60), slz=np.sign(z) * np.log1p(np.abs(z))))
    if extra is not None:
        for c in extra.columns:
            d[c] = extra[c].values
    return d


# need sum of squares in Cum (add lazily so common.py stays as is)
def _sum2(self, lo, hi):
    return self._win(self.s2, lo, hi)


Cum.sum2 = _sum2

# ---------- pseudo-cases ----------
i0, y, _ = train_cases(r, cum, step=23)
A = cum.sum(i0 + 1, i0 + VIS)
Lb = L(cum.mabs(i0 + VIS + 1, i0 + HORIZON))
XH = har_features(i0)
XF = friend_features(i0)
print(f"{len(y)} pseudo-cases; P(y=1)={y.mean():.4f}")

cols_old = [c for c in XH.columns if not c.startswith(("e97", "e99"))]
cut = np.linspace(0, len(y), 6).astype(int)
res = {}
add = lambda name, v: res.setdefault(name, []).append(v)
oof = {}

for k in range(1, 5):
    tr = np.arange(0, cut[k]); te = np.arange(cut[k], cut[k + 1])
    add("p = 0.5", logloss(y[te], np.full(len(te), 0.5)))

    # (a) volhackv1-style logistic on its 10 raw features
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(XF.iloc[tr], y[tr])
    add("friend: logistic on 10 raw features", logloss(y[te], m.predict_proba(XF.iloc[te])[:, 1]))

    # (b) 04_vol_model.py
    rg = Ridge(alpha=1.0).fit(XH[cols_old].iloc[tr], Lb[tr])
    lh_tr, lh_te = rg.predict(XH[cols_old].iloc[tr]), rg.predict(XH[cols_old].iloc[te])
    ex = lambda XHs, lh: pd.DataFrame(dict(surp=XHs.in15.values - lh, Lh=lh))
    S_tr, S_te = sig_stage2(A[tr], lh_tr, ex(XH.iloc[tr], lh_tr)), sig_stage2(A[te], lh_te, ex(XH.iloc[te], lh_te))
    m = LogisticRegression(C=100, max_iter=5000).fit(S_tr, y[tr])
    add("mine (04): HAR + logit", logloss(y[te], m.predict_proba(S_te)[:, 1]))
    r2_old = 1 - np.mean((Lb[te] - lh_te) ** 2) / np.var(Lb[te])

    # (c) bridge stage 1: add EWMA states from both sides of the gap
    rg2 = Ridge(alpha=1.0).fit(XH.iloc[tr], Lb[tr])
    l2_tr, l2_te = rg2.predict(XH.iloc[tr]), rg2.predict(XH.iloc[te])
    r2_new = 1 - np.mean((Lb[te] - l2_te) ** 2) / np.var(Lb[te])
    add("stage-1 R2 old -> new", (round(r2_old, 4), round(r2_new, 4)))
    S1_tr, S1_te = sig_stage2(A[tr], l2_tr, ex(XH.iloc[tr], l2_tr)), sig_stage2(A[te], l2_te, ex(XH.iloc[te], l2_te))
    m = LogisticRegression(C=100, max_iter=5000).fit(S1_tr, y[tr])
    add("bridge-1: +EWMA vol states", logloss(y[te], m.predict_proba(S1_te)[:, 1]))

    # (d) bridge stage 2: add friend's shape / momentum features, scale-free
    sh_tr = shape_features(i0[tr], np.exp(l2_tr) * 1.0)
    sh_te = shape_features(i0[te], np.exp(l2_te) * 1.0)
    S2_tr = pd.concat([S1_tr.reset_index(drop=True), sh_tr.reset_index(drop=True)], axis=1)
    S2_te = pd.concat([S1_te.reset_index(drop=True), sh_te.reset_index(drop=True)], axis=1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000)).fit(S2_tr, y[tr])
    p_lr = lr.predict_proba(S2_te)[:, 1]
    add("bridge-2: +shape/momentum, logit", logloss(y[te], p_lr))
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                        l2_regularization=5.0, min_samples_leaf=200).fit(S2_tr, y[tr])
    p_gb = gb.predict_proba(S2_te)[:, 1]
    add("bridge-2: same features, GBM", logloss(y[te], p_gb))
    add("bridge-2: avg(logit, GBM)", logloss(y[te], 0.5 * (p_lr + p_gb)))

print(f"\n{'model':42s} per-fold                          mean")
for name, v in res.items():
    if name.startswith("stage-1"):
        print(f"{name:42s} {v}")
    else:
        print(f"{name:42s} {np.round(v, 4)}  {np.mean(v):.4f}")

# ---------- final fit and submission ----------
sub = pd.read_csv(SAMPLE)
t0 = case_starts(sub.ID.values, ts0)
XHt = har_features(t0)
At = cum.sum(t0 + 1, t0 + VIS)
rg2 = Ridge(alpha=1.0).fit(XH, Lb)
lh, lht = rg2.predict(XH), rg2.predict(XHt)
ex = lambda XHs, l: pd.DataFrame(dict(surp=XHs.in15.values - l, Lh=l))
S1, S1t = sig_stage2(A, lh, ex(XH, lh)), sig_stage2(At, lht, ex(XHt, lht))
S2 = pd.concat([S1.reset_index(drop=True), shape_features(i0, np.exp(lh)).reset_index(drop=True)], axis=1)
S2t = pd.concat([S1t.reset_index(drop=True), shape_features(t0, np.exp(lht)).reset_index(drop=True)], axis=1)
lr = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000)).fit(S2, y)
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                    l2_regularization=5.0, min_samples_leaf=200).fit(S2, y)
p = 0.5 * (lr.predict_proba(S2t)[:, 1] + gb.predict_proba(S2t)[:, 1])
p = np.clip(p, 0.02, 0.98)
out = pd.DataFrame({"ID": sub.ID.values, "p_up": p})
assert len(out) == len(sub) and out.ID.is_unique and np.isfinite(out.p_up).all()
out.to_csv("submission_bridge.csv", index=False)
print("\nlogit coefficients (standardised):", dict(zip(S2.columns, np.round(lr[-1].coef_[0], 3))))
print("wrote submission_bridge.csv:", out.p_up.describe().round(3).to_dict())
old = pd.read_csv("submission.csv")
print("corr with current submission.csv:", round(np.corrcoef(old.p_up, out.p_up)[0, 1], 4),
      " mean |diff|:", round(float(np.abs(old.p_up - out.p_up).mean()), 4))
