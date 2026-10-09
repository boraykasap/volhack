"""Shared helpers: loading data, locating cases, building features and labels.

Notation (minutes relative to an interval start T0, i0 = grid index of T0):
    r_t = ret at timestamp T0 + 60 t
    supplied: r_1..r_15      withheld: r_16..r_121      supplied again: r_122..
    ID = T0 + 1800           A = r_1+...+r_15           B = r_16+...+r_30
    target: y = 1{A + B > 0}
"""
import numpy as np
import pandas as pd

DATA = "dataset.csv"
SAMPLE = "sample_submission.csv"
VIS = 15       # visible minutes
HORIZON = 30   # target minute
RESUME = 122   # first supplied return after the gap
CTX = 240      # minutes of context used for volatility estimates


def load(path=DATA):
    df = pd.read_csv(path)
    ts0 = int(df.timestamp.iloc[0])
    assert (np.diff(df.timestamp.values) == 60).all(), "grid is not 60s"
    return ts0, df.ret.values.astype(float)


def case_starts(ids, ts0):
    """Grid index i0 of minute 0 for each submission ID."""
    return (np.asarray(ids, dtype=np.int64) - 1800 - ts0) // 60


class Cum:
    """O(1) window sums/counts of r and r^2 that ignore nans."""

    def __init__(self, r):
        ok = ~np.isnan(r)
        z = np.where(ok, r, 0.0)
        self.s1 = np.concatenate([[0.0], np.cumsum(z)])
        self.s2 = np.concatenate([[0.0], np.cumsum(z * z)])
        self.sa = np.concatenate([[0.0], np.cumsum(np.abs(z))])
        self.n = np.concatenate([[0], np.cumsum(ok)])

    def _win(self, c, lo, hi):  # sum over indices lo..hi inclusive
        lo = np.clip(lo, 0, len(c) - 1)
        hi = np.clip(hi + 1, 0, len(c) - 1)
        return c[hi] - c[lo]

    def sum(self, lo, hi):
        return self._win(self.s1, lo, hi)

    def vol(self, lo, hi):
        n = self._win(self.n, lo, hi)
        return np.sqrt(self._win(self.s2, lo, hi) / np.maximum(n, 1))

    def mabs(self, lo, hi):
        """Mean |r| over observed returns in lo..hi (robust to jumps)."""
        n = self._win(self.n, lo, hi)
        return self._win(self.sa, lo, hi) / np.maximum(n, 1)

    def count(self, lo, hi):
        return self._win(self.n, lo, hi)


def features(r, cum, i0):
    """Feature matrix for cases starting at grid indices i0 (only uses supplied data)."""
    i0 = np.asarray(i0)
    A = cum.sum(i0 + 1, i0 + VIS)
    vol_in = cum.vol(i0 + 1, i0 + VIS)
    vol_bef = cum.vol(i0 - CTX + 1, i0)
    vol_aft = cum.vol(i0 + RESUME, i0 + RESUME + CTX - 1)
    vol_bef_s = cum.vol(i0 - 59, i0)
    vol_aft_s = cum.vol(i0 + RESUME, i0 + RESUME + 59)
    last3 = cum.sum(i0 + VIS - 2, i0 + VIS)
    first3 = cum.sum(i0 + 1, i0 + 3)
    half2 = cum.sum(i0 + 8, i0 + VIS)
    # drift context: signed move before / after the gap
    ret_bef = cum.sum(i0 - 59, i0)
    ret_aft = cum.sum(i0 + RESUME, i0 + RESUME + 59)
    return pd.DataFrame(dict(
        A=A, vol_in=vol_in, vol_bef=vol_bef, vol_aft=vol_aft,
        vol_bef_s=vol_bef_s, vol_aft_s=vol_aft_s,
        last3=last3, first3=first3, half2=half2,
        ret_bef=ret_bef, ret_aft=ret_aft,
        minute_of_day=((i0 * 60) % 86400) // 60,
    ))


def train_cases(r, cum, step=47, seed=0):
    """Pseudo-cases from fully observed stretches, with true labels.

    A start i0 is valid if r_1..r_30 and the post-gap context r_122..r_122+CTX-1
    are all supplied and the pre-window is available. The gap itself is not
    required to be missing: we only use features a real case would have.
    """
    n = len(r)
    rng = np.random.default_rng(seed)
    cand = np.arange(CTX + 1, n - RESUME - CTX - 1, step)
    cand = cand + rng.integers(0, step, size=len(cand)) % step
    cand = cand[cand < n - RESUME - CTX - 1]
    ok = (cum.count(cand + 1, cand + HORIZON) == HORIZON) \
        & (cum.count(cand - CTX + 1, cand) >= 30) \
        & (cum.count(cand + RESUME, cand + RESUME + CTX - 1) >= 30)
    i0 = cand[ok]
    S = cum.sum(i0 + 1, i0 + HORIZON)
    B = cum.sum(i0 + VIS + 1, i0 + HORIZON)
    y = (S > 0).astype(int)
    return i0, y, B


def logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
