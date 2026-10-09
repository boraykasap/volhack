"""Where is the remaining edge?  Time-ordered CV on pseudo-cases (never writes any submission).

E_base    : bridge-2 GBM as in 05 (more pseudo-cases)
E_oracle  : same pipeline but sigma = TRUE realised |r_16..r_30| scale (headroom for vol forecasting)
E_jump    : + jump features
E_dow     : + day-of-week x time-of-day volatility profile
E_mono    : base GBM with a monotone constraint in z
E_all     : jump + dow + monotone
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from feat import *

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 11
i0, y, _ = train_cases(r, cum, step=STEP)
A = cum.sum(i0 + 1, i0 + VIS)
Lb = L(cum.mabs(i0 + VIS + 1, i0 + HORIZON))
XH = har_features(i0)
print(f"{len(y)} pseudo-cases (step={STEP}); P(y=1)={y.mean():.4f}")

# ---- jump definition and diagnostics ----
thr = np.nanquantile(np.abs(r), 0.999)
jump = np.where(ok, (np.abs(r) > thr).astype(float), np.nan)
negj = np.where(ok, ((r < -thr)).astype(float), np.nan)
cj, cn = Cum(jump), Cum(negj)
print(f"\njump threshold |r|>{thr:.4f}: {np.nansum(jump):.0f} jumps, {np.nansum(negj) / np.nansum(jump):.1%} negative")
W = r[i0[:, None] + np.arange(1, 16)[None, :]]
has_j = (np.abs(W) > thr).any(1)
jneg = (W < -thr).any(1)
Bsum = cum.sum(i0 + 16, i0 + 30)
sig_loc = cum.mabs(i0 - 239, i0)
print(f"cases with a jump in the visible window: {has_j.mean():.2%}")
for name, m in [("no jump in window", ~has_j), ("window has a negative jump", has_j & jneg),
                ("window has only positive jump(s)", has_j & ~jneg)]:
    print(f"  {name:34s} n={m.sum():6d}  P(y=1)={y[m].mean():.3f}  "
          f"median(B/sig)={np.median(Bsum[m] / (15 * sig_loc[m] + 1e-12)):+.3f}  P(B>0)={(Bsum[m] > 0).mean():.3f}")

lastj = np.maximum.accumulate(np.where(np.nan_to_num(jump) > 0, np.arange(n), -10**9))


def jump_features(i0, sig):
    i0 = np.asarray(i0)
    Wc = r[i0[:, None] + np.arange(1, 16)[None, :]]
    am = np.abs(Wc).argmax(1)
    big = Wc[np.arange(len(i0)), am]
    since = np.minimum(i0 - lastj[i0], 20000)
    return pd.DataFrame(dict(
        w_max=np.abs(big) / sig, w_signed=big / sig, w_nj=cj.sum(i0 + 1, i0 + 15),
        j_b240=cj.sum(i0 - 239, i0), j_b1440=cj.sum(i0 - 1439, i0), nj_b1440=cn.sum(i0 - 1439, i0),
        j_a1440=cj.sum(i0 + RESUME, i0 + RESUME + 1439), nj_a1440=cn.sum(i0 + RESUME, i0 + RESUME + 1439),
        since_last=np.log1p(since),
    )).clip(-50, 1e5)


# ---- day-of-week x time-of-day profile ----
dow = ((ts0 + np.arange(n) * 60) // 86400 + 4) % 7
bins2 = (dow * 48 + bins).astype(int)
prof2 = np.bincount(bins2[ok], weights=np.abs(r[ok]), minlength=336) / np.bincount(bins2[ok], minlength=336)
lprof2 = np.log(prof2)
print("\nmean|r| by weekday (0=Mon..6=Sun):",
      np.round(pd.Series(np.abs(r)).groupby(dow).mean().values * 1e4, 3))
tod2 = lambda i: lprof2[bins2[np.minimum(np.asarray(i), n - 1)]]


def dow_features(i0):
    i0 = np.asarray(i0)
    return pd.DataFrame(dict(tod2_B=tod2(i0 + 23), tod2_in=tod2(i0 + 8), tod2_a=tod2(i0 + 130),
                             tod2_b=tod2(i0 - 30)))


XD = dow_features(i0)
cut = np.linspace(0, len(y), 6).astype(int)


def gbm(mono_idx=None, ncols=0):
    kw = {}
    if mono_idx is not None:
        cst = np.zeros(ncols, dtype=int)
        cst[mono_idx] = 1
        kw["monotonic_cst"] = cst
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                          l2_regularization=5.0, min_samples_leaf=300, **kw)


def run(name, use_dow=False, use_jump=False, mono=False, oracle=False):
    ll, r2s = [], []
    for k in range(1, 5):
        tr = np.arange(0, cut[k]); te = np.arange(cut[k], cut[k + 1])
        X1 = pd.concat([XH, XD], axis=1) if use_dow else XH
        rg = Ridge(alpha=1.0).fit(X1.iloc[tr], Lb[tr])
        lh = rg.predict(X1)
        r2s.append(1 - np.mean((Lb[te] - lh[te]) ** 2) / np.var(Lb[te]))
        if oracle:
            lh = Lb.copy()
        S = sig_stage2(A, lh, pd.DataFrame(dict(surp=XH.in15.values - lh, Lh=lh)))
        S = pd.concat([S.reset_index(drop=True), shape_features(i0, np.exp(lh)).reset_index(drop=True)], axis=1)
        if use_jump:
            S = pd.concat([S, jump_features(i0, np.exp(lh)).reset_index(drop=True)], axis=1)
        if use_dow:
            S = pd.concat([S, XD.reset_index(drop=True)], axis=1)
        m = gbm(0 if mono else None, S.shape[1]).fit(S.iloc[tr], y[tr])
        p = m.predict_proba(S.iloc[te])[:, 1]
        ll.append(logloss(y[te], p))
    print(f"{name:34s} folds {np.round(ll, 4)}  mean {np.mean(ll):.4f}   stage1 R2 {np.mean(r2s):.4f}")
    return np.mean(ll)


print("\n--- GBM CV (log loss) ---")
run("E_base")
run("E_oracle (true B-window scale)", oracle=True)
run("E_jump", use_jump=True)
run("E_dow", use_dow=True)
run("E_mono", mono=True)
run("E_all (jump+dow+mono)", use_jump=True, use_dow=True, mono=True)
