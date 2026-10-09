"""
EPFL VolHack: estimate p_up = P(P_30 > P_0 | supplied data up to minute 15).

Data (Data tab)
---------------
dataset.csv: timestamp (unix s, 60s apart, none missing), ret = ln(P_t/P_{t-60})
or NaN when withheld (~35.3% NaN; first row NaN). Withholding = NaN, not missing rows.
No labels -> manufacture them.

Per case, ID = unix sec of target (minute 30):
  P_0 @ ID-1800   P_15 @ ID-900   P_30 @ ID
  g = sum ret[ID-1740 : ID-900]  (15 supplied) = ln(P_15/P_0)   KNOWN
  R = sum ret[ID-840  : ID    ]  (15 withheld) = ln(P_30/P_15)  UNKNOWN
  y = 1{ g + R > 0 }
Model:  p = P(R > -g | F_15) = F_z( (g + mu) / sigma ).

FRAMING (b): the label is anchored to the fixed 15+15 geometry, but FEATURES
draw on the full trailing history up to the window end (<= ID-900). HARD RULE:
features never touch any return stamped in (ID-900, ID]. That boundary is the
whole leakage discipline -- it is index `e = j+WINDOW` below, used as an
EXCLUSIVE upper bound everywhere in feature code.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss

MIN = 60
WINDOW, HORIZON = 15, 15
SPAN = WINDOW + HORIZON
EWMA_LAMBDA = 0.97        # minute-level RiskMetrics-style decay (TODO tune)
MAX_LOOKBACK = 1440       # 1 day of trailing minutes for history features

FEATURE_COLS = ["g", "rv", "ac1", "range_pos",
                "rv_1h", "rv_1d", "ewma_vol", "mom_1h", "hod_sin", "hod_cos"]


# ======================================================================
# 1. DATA LAYER
# ======================================================================
def load_returns(path: str):
    df = pd.read_csv(path)                       # columns: timestamp, ret
    ts = df["timestamp"].to_numpy(np.int64)
    r = df["ret"].to_numpy(float)
    order = np.argsort(ts)
    ts, r = ts[order], r[order]
    assert np.all(np.diff(ts) == MIN), "rows must be 60s apart, none missing"
    return ts, r


def nonnan_blocks(r: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of non-NaN ret as (start, end_exclusive)."""
    idx = np.where(~np.isnan(r))[0]
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _window_features(sup: np.ndarray) -> dict:
    """From the 15 supplied returns only."""
    rv = float(np.sqrt(np.sum(sup ** 2)))
    cum = np.cumsum(sup)
    rng = cum.max() - cum.min()
    return {"rv": rv, "ac1": _lag1_autocorr(sup),
            "range_pos": float((cum[-1] - cum.min()) / rng) if rng > 0 else 0.5}


def _trailing_features(r: np.ndarray, e: int, rv_fallback: float) -> dict:
    """
    History features from r[:e] ONLY (e = window end, EXCLUSIVE). NaN-safe:
    values inside withheld stretches (incl. the ~0.02% neighbour bleed in the
    hour before a window) are masked out. Vol horizons fall back to the 15-min
    estimate when history is absent.
    """
    def rms(h):
        seg = r[max(0, e - h):e]
        seg = seg[~np.isnan(seg)]
        return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else rv_fallback

    lo = max(0, e - MAX_LOOKBACK)
    hist = r[lo:e]; hist = hist[~np.isnan(hist)]
    if len(hist):
        w = EWMA_LAMBDA ** np.arange(len(hist) - 1, -1, -1)
        ewma_vol = float(np.sqrt(np.sum(w * hist ** 2) / np.sum(w)))
    else:
        ewma_vol = rv_fallback

    seg1h = r[max(0, e - 60):e]; seg1h = seg1h[~np.isnan(seg1h)]
    return {"rv_1h": rms(60), "rv_1d": rms(1440), "ewma_vol": ewma_vol,
            "mom_1h": float(seg1h.sum()) if len(seg1h) else 0.0}


def _time_features(interval_start_ts: int) -> dict:
    hod = (int(interval_start_ts) // 3600) % 24
    ang = 2 * np.pi * hod / 24
    return {"hod_sin": float(np.sin(ang)), "hod_cos": float(np.cos(ang))}


def _assemble(r, j, e, interval_start_ts) -> dict:
    sup = r[j:e]
    wf = _window_features(sup)
    tf = _trailing_features(r, e, rv_fallback=wf["rv"])   # e exclusive -> no leak
    return {"g": float(sup.sum()), **wf, **tf, **_time_features(interval_start_ts)}


def build_training_cases(ts, r, stride: int = 5) -> pd.DataFrame:
    rows = []
    for b0, b1 in nonnan_blocks(r):
        for j in range(b0, b1 - SPAN + 1, stride):
            e = j + WINDOW
            R = float(r[e:j + SPAN].sum())
            idv = int(ts[j + SPAN - 1])
            feat = _assemble(r, j, e, idv - 1800)
            rows.append({**feat, "y": int(feat["g"] + R > 0), "id": idv})
    return pd.DataFrame(rows)


def build_test_cases(ts, r, ids) -> pd.DataFrame:
    pos = {int(t): i for i, t in enumerate(ts)}
    rows = []
    for ID in ids:
        ID = int(ID)
        j = pos.get(ID - 1740)                 # first supplied return by OFFSET
        if j is None:
            raise KeyError(f"ID {ID}: stamp {ID-1740} missing")
        e = j + WINDOW
        if np.isnan(r[j:e]).any():
            raise ValueError(f"ID {ID}: supplied window has NaN")
        rows.append({**_assemble(r, j, e, ID - 1800), "id": ID})
    return pd.DataFrame(rows)


# ======================================================================
# 2. ROUTE A -- DISTRIBUTIONAL
# ======================================================================
@dataclass
class DistributionalModel:
    nu: float = 5.0
    mu_coef: float = 0.0        # TODO fit R ~ g; sign = momentum vs reversion
    vol_scale: float = 1.0
    vol_col: str = "ewma_vol"   # sigma source; "rv" = window-only, "ewma_vol" = history

    def predict_proba(self, df):
        g = df["g"].to_numpy()
        sigma = np.maximum(self.vol_scale * df[self.vol_col].to_numpy(), 1e-12)
        z = (g + self.mu_coef * g) / sigma
        scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1.0
        return stats.t.cdf(z / scale, df=self.nu)

    def fit(self, df):
        return self   # TODO estimate mu_coef, vol_scale, nu on TRAIN only


# ======================================================================
# 3. ROUTE B -- CLASSIFIERS (scaled logistic first; trees must be calibrated)
# ======================================================================
@dataclass
class LogisticModel:
    pipe = None
    def fit(self, df, y):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.pipe.fit(df[FEATURE_COLS].values, y)
        return self
    def predict_proba(self, df):
        return self.pipe.predict_proba(df[FEATURE_COLS].values)[:, 1]
    # TODO GBM + CalibratedClassifierCV only if it beats this OOS.


# ======================================================================
# 4. BASELINES
# ======================================================================
def baseline_base_rate(y_tr, n): return np.full(n, y_tr.mean())
def baseline_naive_g(df, eps=0.02): return np.where(df["g"].to_numpy() > 0, .5 + eps, .5 - eps)


# ======================================================================
# 5. WALK-FORWARD CV + PURGE/EMBARGO
# ======================================================================
def walk_forward_splits(n, n_folds=5, embargo=50):
    fold = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        tr_end = fold * k
        te_start = tr_end + embargo
        te_end = min(te_start + fold, n)
        if te_start >= n:
            break
        yield np.arange(0, tr_end), np.arange(te_start, te_end)


# ======================================================================
# 6. SCORING (binary log loss; p=0.5 -> 0.6931)
# ======================================================================
def score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}

def reliability_curve(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1); idx = np.digitize(p, e) - 1; out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((.5*(e[b]+e[b+1]), p[m].mean(), y[m].mean(), int(m.sum())))
    return pd.DataFrame(out, columns=["bin_mid", "pred_mean", "emp_rate", "count"])


# ======================================================================
# 7. DRIVERS
# ======================================================================
def run_cv(cases: pd.DataFrame):
    cases = cases.sort_values("id").reset_index(drop=True)
    y = cases["y"].to_numpy()
    res = {"logistic": [], "base_rate": [], "naive_g": []}
    for tr, te in walk_forward_splits(len(cases)):
        m = LogisticModel().fit(cases.iloc[tr], y[tr])
        res["logistic"].append(score(y[te], m.predict_proba(cases.iloc[te])))
        res["base_rate"].append(score(y[te], baseline_base_rate(y[tr], len(te))))
        res["naive_g"].append(score(y[te], baseline_naive_g(cases.iloc[te])))
    print(f"base rate P(up) = {y.mean():.4f}")
    for name, f in res.items():
        print(f"{name:10s} log_loss={np.mean([x['log_loss'] for x in f]):.4f} "
              f"brier={np.mean([x['brier'] for x in f]):.4f}")


def make_submission(train, test, out="submission.csv"):
    m = LogisticModel().fit(train, train["y"].to_numpy())
    p = np.clip(m.predict_proba(test), 1e-6, 1 - 1e-6)
    pd.DataFrame({"ID": test["id"].to_numpy(), "p_up": p}).to_csv(out, index=False)


if __name__ == "__main__":
    ts, r = load_returns("dataset.csv")
    train = build_training_cases(ts, r, stride=5)
    run_cv(train)
    ids = pd.read_csv("sample_submission.csv")["ID"]
    make_submission(train, build_test_cases(ts, r, ids))