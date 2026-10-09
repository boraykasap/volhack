"""v4: v3 + seasonal-analogue volatility (same window on nearby days), minute-level |r|
at the gap edges, weekly vol, multiple z ratios, bagged GBMs.

Same pseudo-cases (step 11) and folds as 07_v3_model.py, so CV numbers are comparable
(v3 mean was 0.4821). Writes submission_v4.csv only.
"""
import os
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

dow = ((ts0 + np.arange(n) * 60) // 86400 + 4) % 7
bins2 = (dow * 48 + bins).astype(int)
lprof2 = np.log(np.bincount(bins2[ok], weights=np.abs(r[ok]), minlength=336)
                / np.bincount(bins2[ok], minlength=336))
cal = lambda i: lprof2[bins2[np.clip(np.asarray(i), 0, n - 1)]]

thr = np.nanquantile(np.abs(r), 0.999)
cj = Cum(np.where(ok, (np.abs(r) > thr).astype(float), np.nan))
cn = Cum(np.where(ok, (r < -thr).astype(float), np.nan))
lastj = np.maximum.accumulate(np.where(np.nan_to_num(np.abs(r)) > thr, np.arange(n), -10**9))
absr = np.abs(r)


def analog(i0, shift):
    """log mean|r| over minutes 16..30 of the same clock window shifted by `shift` minutes; nan if unseen."""
    lo, hi = i0 + VIS + 1 + shift, i0 + HORIZON + shift
    c = cum.count(lo, hi)
    v = np.log(cum.mabs(lo, hi) + EPS)
    valid = (c >= 8) & (lo >= 0) & (hi < n)
    return np.where(valid, v, np.nan)


def stage1_X(i0):
    i0 = np.asarray(i0)
    X = har_features(i0)
    X["cal_B"], X["cal_in"], X["cal_b"], X["cal_a"] = cal(i0 + 23), cal(i0 + 8), cal(i0 - 30), cal(i0 + 130)
    X["a5"] = L(cum.mabs(i0 + RESUME, i0 + RESUME + 4))
    X["in_last2"] = L(cum.mabs(i0 + 14, i0 + 15))
    X["j_b1440"] = cj.sum(i0 - 1439, i0)
    X["j_a1440"] = cj.sum(i0 + RESUME, i0 + RESUME + 1439)
    X["j_in"] = cj.sum(i0 + 1, i0 + 15)
    X["b10080"] = L(cum.mabs(i0 - 10079, i0))
    X["a10080"] = L(cum.mabs(i0 + RESUME, i0 + RESUME + 10079))
    for k in range(5):  # minute-level edges of the gap
        X[f"m_in{k}"] = L(absr[i0 + 15 - k])
        X[f"m_a{k}"] = L(np.nan_to_num(absr[np.minimum(i0 + RESUME + k, n - 1)], nan=0.0))
    an = {}
    for d, s in [("ym1", -1440), ("yp1", 1440), ("wm1", -10080), ("wp1", 10080), ("ym2", -2880), ("yp2", 2880)]:
        an[d] = analog(i0, s)
        # analogue relative to the local level at that day's same clock window, filled for ridge
        X[f"an_{d}"] = np.where(np.isnan(an[d]), X["cal_B"] + (X["b240"] - X["cal_b"]), an[d])
        X[f"has_{d}"] = (~np.isnan(an[d])).astype(float)
    stack = np.column_stack(list(an.values()))
    cnt = (~np.isnan(stack)).sum(1)
    X["an_mean"] = np.where(cnt > 0, np.nansum(stack, 1) / np.maximum(cnt, 1), X["cal_B"] + X["b240"] - X["cal_b"])
    return X


def stage2_X(i0, Lh, X1):
    i0 = np.asarray(i0)
    sig = np.exp(Lh)
    Ai = cum.sum(i0 + 1, i0 + VIS)
    S = sig_stage2(Ai, Lh, pd.DataFrame(dict(surp=X1.in15.values - Lh, Lh=Lh)))
    S = pd.concat([S.reset_index(drop=True), shape_features(i0, sig).reset_index(drop=True)], axis=1)
    Wc = r[i0[:, None] + np.arange(1, 16)[None, :]]
    big = Wc[np.arange(len(i0)), np.abs(Wc).argmax(1)]
    S["w_signed"] = np.clip(big / sig, -50, 50)
    S["w_negj"] = cn.sum(i0 + 1, i0 + 15)
    S["nj_b1440"] = cn.sum(i0 - 1439, i0)
    S["since_last"] = np.log1p(np.minimum(i0 - lastj[i0], 20000))
    S["cal_B"] = X1.cal_B.values
    # alternative scale estimates -> extra z ratios
    for c in ("in15", "b240", "a240", "an_mean"):
        S[f"z_{c}"] = np.clip(Ai / (np.exp(X1[c].values) * np.sqrt(VIS)), -60, 60)
    return S


reg = lambda: HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300,
                                            min_samples_leaf=300, l2_regularization=1.0)


def clf_bag(Str, ytr, Ste):
    ps = []
    for seed in range(3):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                           l2_regularization=5.0, min_samples_leaf=300,
                                           max_features=0.8, random_state=seed).fit(Str, ytr)
        ps.append(m.predict_proba(Ste)[:, 1])
    return np.mean(ps, 0)


def fit_stage1(X, t):
    rg, gb = Ridge(alpha=1.0).fit(X, t), reg().fit(X, t)
    return lambda Z: 0.5 * (rg.predict(Z) + gb.predict(Z))


def oof_stage1(X, t, k=4):
    out = np.empty(len(t))
    edges = np.linspace(0, len(t), k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.r_[0:a, b:len(t)]
        out[a:b] = fit_stage1(X.iloc[m], t[m])(X.iloc[a:b])
    return out


X1 = stage1_X(i0)
cut = np.linspace(0, len(y), 6).astype(int)
rows = []
for k in range(1, 5):
    tr = np.arange(0, cut[k]); te = np.arange(cut[k], cut[k + 1])
    f1 = fit_stage1(X1.iloc[tr], Lb[tr])
    lh_tr, lh_te = oof_stage1(X1.iloc[tr], Lb[tr]), f1(X1.iloc[te])
    r2 = 1 - np.mean((Lb[te] - lh_te) ** 2) / np.var(Lb[te])
    p = clf_bag(stage2_X(i0[tr], lh_tr, X1.iloc[tr]), y[tr], stage2_X(i0[te], lh_te, X1.iloc[te]))
    rows.append(dict(fold=k, v4=round(logloss(y[te], p), 4), stage1_R2=round(r2, 4)))
    print(rows[-1], flush=True)
cv = pd.DataFrame(rows)
print("\nv4 mean:", cv.mean().round(4).to_dict(), "  (v3: logloss 0.4821, R2 0.5752)")

sub = pd.read_csv(SAMPLE)
t0 = case_starts(sub.ID.values, ts0)
X1t = stage1_X(t0)
lh_all, lh_t = oof_stage1(X1, Lb), fit_stage1(X1, Lb)(X1t)
p = np.clip(clf_bag(stage2_X(i0, lh_all, X1), y, stage2_X(t0, lh_t, X1t)), 0.02, 0.98)
out = pd.DataFrame({"ID": sub.ID.values, "p_up": p})
assert len(out) == len(sub) and out.ID.is_unique and np.isfinite(out.p_up).all() and (out.ID.values == sub.ID.values).all()
out.to_csv("submission_v4.csv", index=False)
print("wrote submission_v4.csv", out.p_up.describe().round(3).to_dict())
for f in ("submission_v3.csv", "submission.csv"):
    if os.path.exists(f):
        print(f"corr with {f}:", round(np.corrcoef(pd.read_csv(f).p_up, out.p_up)[0, 1], 4))
