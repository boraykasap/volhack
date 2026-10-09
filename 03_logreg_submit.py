"""Logistic regression / GBM on z-score + context features, time-split CV, write submission."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from common import *

ts0, r = load()
cum = Cum(r)
i0, y, B = train_cases(r, cum, step=13)


def design(X):
    s = np.maximum(np.sqrt((X.vol_bef ** 2 + X.vol_aft ** 2) / 2), 1e-12)
    root = np.sqrt(VIS)
    D = pd.DataFrame(dict(
        z=X.A / (s * root),
        z_in=X.A / (np.maximum(X.vol_in, 1e-12) * root),
        z_last3=X.last3 / (s * np.sqrt(3)),
        z_first3=X.first3 / (s * np.sqrt(3)),
        z_half2=X.half2 / (s * np.sqrt(8)),
        z_bef=X.ret_bef / (s * np.sqrt(60)),
        z_aft=X.ret_aft / (s * np.sqrt(60)),
        vol_ratio=np.log(np.maximum(X.vol_in, 1e-12) / s),
        tod_sin=np.sin(2 * np.pi * X.minute_of_day / 1440),
        tod_cos=np.cos(2 * np.pi * X.minute_of_day / 1440),
    ))
    return D.clip(-8, 8)


X = design(features(r, cum, i0))
# Expanding-window split in time (no shuffling)
cut = np.linspace(0, len(y), 6).astype(int)
models = {
    "logreg z only": (lambda: LogisticRegression(C=10), ["z"]),
    "logreg all": (lambda: LogisticRegression(C=1, max_iter=1000), list(X.columns)),
    "gbm all": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                       max_iter=200, l2_regularization=1.0),
                list(X.columns)),
}
print(f"{len(y)} pseudo-cases")
for name, (mk, cols) in models.items():
    scores = []
    for k in range(2, 5):
        tr, te = slice(0, cut[k]), slice(cut[k], cut[k + 1])
        m = mk().fit(X[cols].iloc[tr], y[tr])
        scores.append(logloss(y[te], m.predict_proba(X[cols].iloc[te])[:, 1]))
    print(f"{name:15s} CV logloss per fold {np.round(scores, 4)}  mean {np.mean(scores):.4f}")

# Fit on everything with the simple model and write the submission
sub = pd.read_csv(SAMPLE)
i0_test = case_starts(sub.ID.values, ts0)
Xt = design(features(r, cum, i0_test))
m = LogisticRegression(C=10).fit(X[["z"]], y)
sub["p_up"] = np.clip(m.predict_proba(Xt[["z"]])[:, 1], 0.02, 0.98)
sub[["ID", "p_up"]].to_csv("submission.csv", index=False)
print("\nwrote submission.csv", sub.p_up.describe().round(3).to_dict())
