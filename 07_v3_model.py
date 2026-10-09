"""v3: nonlinear volatility forecast (stage 1) + GBM classifier with jump/calendar features (stage 2).

Stage 1 is cross-fitted (out-of-fold) so stage 2 never trains on in-sample forecasts.
Time-ordered CV against the 05 bridge pipeline; writes submission_v3.csv only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from feat import *

STEP = 11
i0, y, _ = train_cases(r, cum, step=STEP)
A = cum.sum(i0 + 1, i0 + VIS)
Lb = L(cum.mabs(i0 + VIS + 1, i0 + HORIZON))
print(f"{len(y)} pseudo-cases")

# calendar profile: weekday x half-hour mean|r|
dow = ((ts0 + np.arange(n) * 60) // 86400 + 4) % 7
bins2 = (dow * 48 + bins).astype(int)
lprof2 = np.log(np.bincount(bins2[ok], weights=np.abs(r[ok]), minlength=336)
                / np.bincount(bins2[ok], minlength=336))
cal = lambda i: lprof2[bins2[np.clip(np.asarray(i), 0, n - 1)]]

# jumps
thr = np.nanquantile(np.abs(r), 0.999)
cj = Cum(np.where(ok, (np.abs(r) > thr).astype(float), np.nan))
cn = Cum(np.where(ok, (r < -thr).astype(float), np.nan))
lastj = np.maximum.accumulate(np.where(np.nan_to_num(np.abs(r)) > thr, np.arange(n), -10**9))


def stage1_X(i0):
    i0 = np.asarray(i0)
    X = har_features(i0)
    X["cal_B"], X["cal_in"], X["cal_b"], X["cal_a"] = cal(i0 + 23), cal(i0 + 8), cal(i0 - 30), cal(i0 + 130)
    X["a5"] = L(cum.mabs(i0 + RESUME, i0 + RESUME + 4))
    X["in_last2"] = L(cum.mabs(i0 + 14, i0 + 15))
    X["j_b1440"] = cj.sum(i0 - 1439, i0)
    X["j_a1440"] = cj.sum(i0 + RESUME, i0 + RESUME + 1439)
    X["j_in"] = cj.sum(i0 + 1, i0 + 15)
    return X


def stage2_X(i0, Lh, X1):
    i0 = np.asarray(i0)
    sig = np.exp(Lh)
    S = sig_stage2(A_of(i0), Lh, pd.DataFrame(dict(surp=X1.in15.values - Lh, Lh=Lh)))
    S = pd.concat([S.reset_index(drop=True), shape_features(i0, sig).reset_index(drop=True)], axis=1)
    Wc = r[i0[:, None] + np.arange(1, 16)[None, :]]
    big = Wc[np.arange(len(i0)), np.abs(Wc).argmax(1)]
    S["w_signed"] = np.clip(big / sig, -50, 50)
    S["w_negj"] = cn.sum(i0 + 1, i0 + 15)
    S["nj_b1440"] = cn.sum(i0 - 1439, i0)
    S["since_last"] = np.log1p(np.minimum(i0 - lastj[i0], 20000))
    S["cal_B"] = X1.cal_B.values
    return S


A_of = lambda i: cum.sum(np.asarray(i) + 1, np.asarray(i) + VIS)

reg = lambda: HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300,
                                            min_samples_leaf=300, l2_regularization=1.0)
clf = lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                             l2_regularization=5.0, min_samples_leaf=300)


def fit_stage1(X, t):
    rg, gb = Ridge(alpha=1.0).fit(X, t), reg().fit(X, t)
    return lambda Z: 0.5 * (rg.predict(Z) + gb.predict(Z))


def oof_stage1(X, t, k=4):
    """Out-of-fold stage-1 forecasts on the training set (contiguous time blocks)."""
    out = np.empty(len(t))
    edges = np.linspace(0, len(t), k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.r_[0:a, b:len(t)]
        out[a:b] = fit_stage1(X.iloc[m], t[m])(X.iloc[a:b])
    return out


X1 = stage1_X(i0)
XH_old = har_features(i0)
cut = np.linspace(0, len(y), 6).astype(int)
rows = []
for k in range(1, 5):
    tr = np.arange(0, cut[k]); te = np.arange(cut[k], cut[k + 1])
    # reference: 05 bridge pipeline (ridge stage 1, GBM stage 2 on shape features)
    rg = Ridge(alpha=1.0).fit(XH_old.iloc[tr], Lb[tr])
    lh = rg.predict(XH_old)
    S = sig_stage2(A, lh, pd.DataFrame(dict(surp=XH_old.in15.values - lh, Lh=lh)))
    S = pd.concat([S.reset_index(drop=True), shape_features(i0, np.exp(lh)).reset_index(drop=True)], axis=1)
    p_ref = clf().fit(S.iloc[tr], y[tr]).predict_proba(S.iloc[te])[:, 1]
    # v3
    f1 = fit_stage1(X1.iloc[tr], Lb[tr])
    lh_tr, lh_te = oof_stage1(X1.iloc[tr], Lb[tr]), f1(X1.iloc[te])
    r2 = 1 - np.mean((Lb[te] - lh_te) ** 2) / np.var(Lb[te])
    S_tr, S_te = stage2_X(i0[tr], lh_tr, X1.iloc[tr]), stage2_X(i0[te], lh_te, X1.iloc[te])
    p_v3 = clf().fit(S_tr, y[tr]).predict_proba(S_te)[:, 1]
    rows.append(dict(fold=k, ref_05=logloss(y[te], p_ref), v3=logloss(y[te], p_v3),
                     blend=logloss(y[te], 0.5 * (p_ref + p_v3)), stage1_R2_v3=r2))
    print(rows[-1])
cv = pd.DataFrame(rows)
print("\nmean:", cv.mean().round(4).to_dict())

# ---- final fit on all pseudo-cases, predict real cases ----
sub = pd.read_csv(SAMPLE)
t0 = case_starts(sub.ID.values, ts0)
X1t = stage1_X(t0)
f1 = fit_stage1(X1, Lb)
lh_all, lh_t = oof_stage1(X1, Lb), f1(X1t)
S, St = stage2_X(i0, lh_all, X1), stage2_X(t0, lh_t, X1t)
p = clf().fit(S, y).predict_proba(St)[:, 1]
p = np.clip(p, 0.02, 0.98)
out = pd.DataFrame({"ID": sub.ID.values, "p_up": p})
assert len(out) == len(sub) and out.ID.is_unique and np.isfinite(out.p_up).all() and (out.ID.values == sub.ID.values).all()
out.to_csv("submission_v3.csv", index=False)
print("wrote submission_v3.csv", out.p_up.describe().round(3).to_dict())
import os
for f in ("submission_bridge.csv", "submission.csv"):
    if os.path.exists(f):
        o = pd.read_csv(f)
        print(f"corr with {f}:", round(np.corrcoef(o.p_up, out.p_up)[0, 1], 4))
