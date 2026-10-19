"""Detectors that read only the final linear classifier (W, b), K rows. Higher scores mean more backdoored.

    fields     Fields et al. (2021): Dixon's Q on the classifier row means w_i, Q = (w(1) - w(2)) / (w(1) - w(K)), target
               argmax w_i. Native rule: Q above Dixon's critical value at alpha = 0.05 for K classes (Monte Carlo under
               normal rows; 0.466 at K = 10)
    dfb        DFBScanner (Yu et al. 2026; github.com/EboYu/DFBScanner, DFBScanner.py): the 62 final-layer indicators, each
               a per-class vector min-max normalised to [0, 1]. A configuration sums the selected indicators into a
               detection score, compares it with the clean profile by cosine similarity and flags when the similarity is
               below a threshold (bdbaselines.rules); here: the 62 x K indicator matrix
    rownorm    the ratio of the two largest classifier row norms ||W_c||; target argmax
"""
from __future__ import annotations

import functools

import numpy as np
from torch import nn

DFB_BASES = ['AverageWeight', 'AbsAverageWeight', 'AverageAbsWeight', 'WeightVariance', 'WeightSTD', 'WeightL2Norm',
             'WeightEnergy', 'Bias', 'WeightSumBias', 'WeightMeanBias', 'WeightSimilarity', 'WeightCertainty', 'WeightBiasZscore']
# DFBScanner.all_metrics (the official list, 62 indicators)
DFB_ALL = [f'{b}{s}' for b in DFB_BASES[:-1] for s in ('', 'ZScore', 'Nad', 'IQRU', 'IQRL')] + ['WeightBiasZscore', 'WeightBiasZscoreNad']


def classifier(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """(W, b) of the final linear layer in float64; W is K x d (one row per class)."""
    fc = model.fc if hasattr(model, 'fc') else [m for m in model.modules() if isinstance(m, nn.Linear)][-1]  # e.g. 'linear'
    return fc.weight.detach().double().cpu().numpy(), fc.bias.detach().double().cpu().numpy()


def _top2(v: np.ndarray) -> tuple[float, int]:
    order = np.argsort(v)[::-1]
    return float(v[order[0]] / max(v[order[1]], 1e-300)), int(order[0])


def rownorm(W: np.ndarray, b: np.ndarray | None = None) -> dict:
    r, t = _top2(np.linalg.norm(W, axis=1))
    return {'rownorm': r, 'rownorm_target': t}


# ----------------------------------------------------------------------------------------------- Fields (Dixon's Q)

def dixon_q(v: np.ndarray) -> float:
    s = np.sort(v)[::-1]
    return float((s[0] - s[1]) / max(s[0] - s[-1], 1e-300))


@functools.lru_cache(maxsize=None)
def dixon_critical(K: int, alpha: float = 0.05, n: int = 200_000) -> float:
    """Dixon's Q critical value for K normal observations at level alpha, by Monte Carlo with a fixed seed. Like the
    tabulated values (Q95 = 0.466 at K = 10; DFBScanner's Dixon rule uses 0.468) it is the 1 - alpha/2 quantile of the
    upper-end Q."""
    x = np.random.default_rng(12345).standard_normal((n, K))
    s = np.sort(x, 1)[:, ::-1]
    return float(np.quantile((s[:, 0] - s[:, 1]) / (s[:, 0] - s[:, -1]), 1 - alpha / 2))


def fields(W: np.ndarray, b: np.ndarray | None = None) -> dict:
    w = W.mean(1)
    q = dixon_q(w)
    crit = dixon_critical(len(w))
    return {'fields_q': q, 'fields_target': int(np.argmax(w)), 'fields_crit': crit, 'fields_flag': q > crit}


# ----------------------------------------------------------------------------------------------- DFBScanner

def _norm(f: np.ndarray) -> np.ndarray:
    r = np.max(f) - np.min(f)
    return np.zeros_like(f) if r == 0 else (f - np.min(f)) / r


def _transform(f: np.ndarray, sub: str) -> np.ndarray:
    if sub == '':
        return _norm(f)
    if sub == 'ZScore':
        return _norm((f - f.mean()) / (f.std() + 1e-8))
    if sub == 'Nad':
        return _norm(np.abs(f - f.mean()) / max(np.max(f) - np.min(f), 1e-300))
    q1, q3 = np.quantile(f, 0.25), np.quantile(f, 0.75)
    iqr = q3 - q1
    return _norm(f - (q3 + 1.5 * iqr)) if sub == 'IQRU' else _norm((q1 - 1.5 * iqr) - f)


def dfb_bases(W: np.ndarray, b: np.ndarray) -> dict:
    """DFBScanner's per-class base features (DFBScanner.compute_anomaly_indicator; weights there = W.T)."""
    mean = W.mean(1)
    energy = np.exp(np.mean(W ** 2, 1) - np.max(np.mean(W ** 2, 1)))
    unit = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-300)
    cos_dist = 1 - unit @ unit.T  # scipy.spatial.distance.cosine
    eu = W.shape[1] / (W.sum(1) + W.shape[1])
    z = np.abs((W - W.mean(1, keepdims=True)) / (W.std(1, ddof=1, keepdims=True) + 1e-8)).sum(1)
    z = _norm(np.abs((z - z.mean()) / z.std(ddof=1)))
    bn = _norm(b)
    return {'AverageWeight': mean, 'AbsAverageWeight': np.abs(mean), 'AverageAbsWeight': np.abs(W).mean(1),
            'WeightVariance': W.var(1), 'WeightSTD': W.std(1), 'WeightL2Norm': np.linalg.norm(W, axis=1),
            'WeightEnergy': energy / energy.sum(), 'Bias': b, 'WeightSumBias': W.sum(1) + b, 'WeightMeanBias': mean + b,
            'WeightSimilarity': 1 - _norm(cos_dist.mean(1)), 'WeightCertainty': 1 - _norm(eu),
            'WeightBiasZscore': z + _norm((bn - bn.mean()) / (bn.std() + 1e-8))}


def dfb_indicators(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 62 x K indicator matrix (rows in DFB_ALL order)."""
    bases = dfb_bases(W, b)
    rows = []
    for name in DFB_ALL:
        sub = next((s for s in ('Nad', 'ZScore', 'IQRL', 'IQRU') if name.endswith(s)), '')
        rows.append(_transform(bases[name[:len(name) - len(sub)]], sub))
    return np.vstack(rows)


def dfb(W: np.ndarray, b: np.ndarray) -> dict:
    """The indicator matrix, rounded to 5 decimals as stored for the paper; its configuration-A target."""
    ind = np.round(dfb_indicators(W, b), 5)
    return {'dfb_ind': ind.tolist(), 'dfb_target': int(np.argmax(ind.sum(0)))}
