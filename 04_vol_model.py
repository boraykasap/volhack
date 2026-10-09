"""Volatility-driven model.

Stage 1 (HAR-style): predict the log-scale of the unseen 15 minutes,
    L = log mean|r_16..r_30|,
from log mean|r| / rms over windows before, inside and after the case plus the
time-of-day volatility profile.
Stage 2: z = A / (exp(L_hat) sqrt(15)) -> calibrated probability.

Time-ordered expanding-window CV; then fit on everything and write submission.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from common import *

EPS = 1e-6
ts0, r = load()
cum = Cum(r)
n = len(r)

# time-of-day profile of mean |r| in 30-minute bins (uses all supplied data)
bins = (((np.arange(n) * 60 + ts0) % 86400) // 1800).astype(int)
ok = ~np.isnan(r)
prof = np.bincount(bins[ok], weights=np.abs(r[ok]), minlength=48) / np.bincount(bins[ok], minlength=48)
lprof = np.log(prof)


def tod(i):
    return lprof[(((np.asarray(i) * 60 + ts0) % 86400) // 1800).astype(int)]


def vol_features(i0):
    i0 = np.asarray(i0)
    L = lambda x: np.log(x + EPS)
    F = {}
    F["in15"] = L(cum.mabs(i0 + 1, i0 + 15))
    F["rms_in"] = L(cum.vol(i0 + 1, i0 + 15))
    F["in5"] = L(cum.mabs(i0 + 11, i0 + 15))  # most recent 5 minutes before the gap
    for w in (15, 60, 240, 1440):
        F[f"b{w}"] = L(cum.mabs(i0 - w + 1, i0))
        F[f"a{w}"] = L(cum.mabs(i0 + RESUME, i0 + RESUME + w - 1))
    F["rms_b60"] = L(cum.vol(i0 - 59, i0))
    F["rms_a60"] = L(cum.vol(i0 + RESUME, i0 + RESUME + 59))
    F["tod_B"] = tod(i0 + 23)
    F["tod_in"] = tod(i0 + 8)
    F["tod_b"] = tod(i0 - 30)
    F["tod_a"] = tod(i0 + 130)
    return pd.DataFrame(F)


def stage2_X(A, Lhat, extra=None):
    z = A / (np.exp(Lhat) * np.sqrt(VIS))
    d = pd.DataFrame(dict(z=np.clip(z, -60, 60), slz=np.sign(z) * np.log1p(np.abs(z))))
    if extra is not None:
        for c in extra.columns:
            d[c] = extra[c].values
    return d


# ---------------- pseudo-cases ----------------
i0, y, B = train_cases(r, cum, step=23)
A = cum.sum(i0 + 1, i0 + VIS)
Lb = np.log(cum.mabs(i0 + VIS + 1, i0 + HORIZON) + EPS)  # stage-1 target
X = vol_features(i0)
print(f"{len(y)} pseudo-cases; P(y=1)={y.mean():.4f}")

# ---------------- time-ordered CV ----------------
cut = np.linspace(0, len(y), 6).astype(int)
res = {}


def add(name, k, v):
    res.setdefault(name, []).append(v)


for k in range(1, 5):
    tr = np.arange(0, cut[k])
    te = np.arange(cut[k], cut[k + 1])
    # stage 1
    ridge = Ridge(alpha=1.0).fit(X.iloc[tr], Lb[tr])
    Lh_tr, Lh_te = ridge.predict(X.iloc[tr]), ridge.predict(X.iloc[te])
    r2 = 1 - np.mean((Lb[te] - Lh_te) ** 2) / np.var(Lb[te])
    r2_in = 1 - np.mean((Lb[te] - X.in15.iloc[te]) ** 2) / np.var(Lb[te])
    add("stage1 R2 (HAR)", k, r2)
    # stage 2 variants
    S_tr, S_te = stage2_X(A[tr], Lh_tr), stage2_X(A[te], Lh_te)
    m = LogisticRegression(C=100, max_iter=2000).fit(S_tr[["z"]], y[tr])
    add("HAR + logit[z]", k, logloss(y[te], m.predict_proba(S_te[["z"]])[:, 1]))
    m = LogisticRegression(C=100, max_iter=2000).fit(S_tr, y[tr])
    add("HAR + logit[z, slog z]", k, logloss(y[te], m.predict_proba(S_te)[:, 1]))
    # control: no HAR, just in-window mean|r| as the scale
    Sc_tr, Sc_te = stage2_X(A[tr], X.in15.iloc[tr].values), stage2_X(A[te], X.in15.iloc[te].values)
    m = LogisticRegression(C=100, max_iter=2000).fit(Sc_tr, y[tr])
    add("control: in-window scale + logit", k, logloss(y[te], m.predict_proba(Sc_te)[:, 1]))
    # extras: how surprising the window is relative to forecast
    ex_tr = pd.DataFrame(dict(surp=X.in15.iloc[tr].values - Lh_tr, Lh=Lh_tr))
    ex_te = pd.DataFrame(dict(surp=X.in15.iloc[te].values - Lh_te, Lh=Lh_te))
    S2_tr, S2_te = stage2_X(A[tr], Lh_tr, ex_tr), stage2_X(A[te], Lh_te, ex_te)
    m = LogisticRegression(C=100, max_iter=5000).fit(S2_tr, y[tr])
    add("HAR + logit[z, slog z, surprise, Lhat]", k, logloss(y[te], m.predict_proba(S2_te)[:, 1]))
    g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                       l2_regularization=5.0, min_samples_leaf=200)
    g.fit(S2_tr, y[tr])
    add("HAR + GBM[z, slog z, surprise, Lhat]", k, logloss(y[te], g.predict_proba(S2_te)[:, 1]))
    add("p = 0.5", k, logloss(y[te], np.full(len(te), 0.5)))

print(f"\n{'model':45s} folds                         mean")
for name, v in res.items():
    print(f"{name:45s} {np.round(v, 4)}  {np.mean(v):.4f}")

# ---------------- fit on all, predict test ----------------
sub = pd.read_csv(SAMPLE)
t0 = case_starts(sub.ID.values, ts0)
Xt = vol_features(t0)
At = cum.sum(t0 + 1, t0 + VIS)

ridge = Ridge(alpha=1.0).fit(X, Lb)
print("\nstage-1 ridge coefficients:")
print(pd.Series(ridge.coef_, index=X.columns).round(3).to_string())
Lh, Lht = ridge.predict(X), ridge.predict(Xt)
ex = pd.DataFrame(dict(surp=X.in15.values - Lh, Lh=Lh))
ext = pd.DataFrame(dict(surp=Xt.in15.values - Lht, Lh=Lht))
S, St = stage2_X(A, Lh, ex), stage2_X(At, Lht, ext)
m = LogisticRegression(C=100, max_iter=5000).fit(S, y)
print("stage-2 coefficients:", dict(zip(S.columns, np.round(m.coef_[0], 3))), "intercept", round(m.intercept_[0], 3))

for lo in (0.02, 0.05):
    pc = np.clip(m.predict_proba(S)[:, 1], lo, 1 - lo)
    print(f"in-sample logloss with clip [{lo}, {1 - lo}]: {logloss(y, pc):.4f}")

p = np.clip(m.predict_proba(St)[:, 1], 0.02, 0.98)
out = pd.DataFrame({"ID": sub.ID.values, "p_up": p})
assert len(out) == len(sub) and out.ID.is_unique and np.isfinite(out.p_up).all()
out.to_csv("submission.csv", index=False)
print("\nwrote submission.csv:", out.p_up.describe().round(3).to_dict())
