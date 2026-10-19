"""Each method's own decision rule, as published (marks: * needs no reference models, ** needs labelled clean and poisoned
configuration models, (clean) needs clean reference models only).

    fields     *   Dixon's Q above its critical value at alpha = 0.05
    mmbd       *   p <= 0.05 under the fitted gamma null
    clp        *   any unit with z > 3 (score: the largest z)
    bnp        *   any unit with z > 3, from 500 clean images
    dfb        **  DFBScanner configuration A (all 62 indicators), as in its IndicatorSelection.py: the clean profile is
                   score_norm(mean of the clean configuration models' score_norm'd indicator sums); the cosine-similarity
                   threshold maximises F1 on the labelled configuration models (gp_minimize, 50 calls, 10 random starts,
                   random_state 42); a model is flagged when its similarity is below the threshold
    htell      **  the probe variant (u/g law, mean/l2 statistic) with the highest AUROC on the labelled configuration models;
                   the threshold maximises TPR subject to FPR below 5% on them
    freeeagle  **  score |M - 1|; threshold from the ROC of the labelled configuration models, as in its
                   analyze_result_csv.py: the point whose FPR is closest to 0.05 and the midpoint to its neighbour
    htell_lc1  (clean) LC-1 uniform probes, mean statistic, above the clean reference maximum
    rownorm    (clean) above the clean reference maximum
    prop           no threshold specified (AUROC only)

The dfb and freeeagle rules need scikit-optimize and scikit-learn (pip install 'bdbaselines[rules]').
"""
from __future__ import annotations

import numpy as np

FIXED = {'fields': ('fields_q', 'fields_flag'), 'mmbd': ('mmbd_score', 'mmbd_flag'), 'clp': ('clp_z', 'clp_flag'),
         'bnp': ('bnp_z', 'bnp_flag')}
CLEAN_MAX = {'rownorm': 'rownorm', 'htell_lc1': 'htell_lc1_u_mean'}
METHODS = ('fields', 'dfb', 'htell', 'htell_lc1', 'clp', 'prop', 'bnp', 'mmbd', 'freeeagle', 'rownorm')


def auroc(p, n) -> float | None:
    p, n = np.asarray(p, float), np.asarray(n, float)
    if not len(p) or not len(n):
        return None
    return float(((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def score_norm(score):  # DFBScanner IndicatorSelection.score_norm
    if np.max(score) - np.min(score) == 0:
        return score
    return (score - np.min(score)) / (np.max(score) - np.min(score))


def _dfb_vec(ind) -> np.ndarray:
    return score_norm(np.asarray(ind).sum(0))  # (62, K) -> per-class score


def _cos(s, profile) -> float:
    return float(np.dot(s, profile) / (np.linalg.norm(s) * np.linalg.norm(profile)))


def _labelled(rows, key):
    pos = [r for r in rows if r.get('label') == 'backdoored' and r.get(key) is not None]
    neg = [r for r in rows if r.get('label') == 'clean' and r.get(key) is not None]
    return pos, neg


def fit(method: str, cfg: list[dict]) -> dict:
    """The parameters of ``method``'s decision rule from configuration score rows (each with 'label')."""
    if method in FIXED or method == 'prop':
        return {}
    if method in CLEAN_MAX:
        _, neg = _labelled(cfg, CLEAN_MAX[method])
        return {'threshold': max(r[CLEAN_MAX[method]] for r in neg)}
    if method == 'htell':
        variants = ('u_mean', 'u_l2', 'g_mean', 'g_l2')
        aucs = {v: auroc(*([r[f'htell_{v}'] for r in s] for s in _labelled(cfg, f'htell_{v}'))) or 0.0 for v in variants}
        v = max(variants, key=lambda v: (aucs[v], v == 'u_mean'))
        neg = sorted((r[f'htell_{v}'] for r in _labelled(cfg, f'htell_{v}')[1]), reverse=True)
        allowed = max(k for k in range(len(neg)) if k / len(neg) < 0.05)  # clean flags permitted at FPR < 5%
        return {'variant': v, 'aucs': aucs, 'threshold': neg[allowed]}
    if method == 'dfb':
        from sklearn.metrics import f1_score
        from skopt import gp_minimize
        from skopt.space import Real

        pos, neg = _labelled(cfg, 'dfb_ind')
        clean, pois = [_dfb_vec(r['dfb_ind']) for r in neg], [_dfb_vec(r['dfb_ind']) for r in pos]
        profile = score_norm(np.mean(clean, 0))
        cfg_cos = np.array([_cos(score_norm(s), profile) for s in clean + pois])
        trues = np.r_[np.zeros(len(clean)), np.ones(len(pois))]
        obj = lambda x: -f1_score(trues, (cfg_cos < x[0]).astype(float))  # noqa: E731
        res = gp_minimize(obj, [Real(0, 1.0, name='c_threshold')], n_calls=50, n_random_starts=10, random_state=42, verbose=False)
        return {'profile': profile.tolist(), 'threshold': float(res.x[0]), 'config_f1': float(-res.fun)}
    if method == 'freeeagle':
        from sklearn.metrics import roc_curve

        pos, neg = _labelled(cfg, 'freeeagle')
        dp, dn = [r['freeeagle'] for r in pos], [r['freeeagle'] for r in neg]
        labels, metrics = [1] * len(dp) + [0] * len(dn), dp + dn
        fpr, tpr, thr = roc_curve(labels, metrics)
        thr = thr.copy()
        thr[0] = max(metrics) + 1  # sklearn 1.0.2, the version of the released code
        d = np.abs(fpr - 0.05).tolist()
        temp_index, max_index = 0, None
        try:
            while 1:
                temp_index = d.index(min(d), temp_index + 1)
        except ValueError:
            max_index = temp_index
        T = (thr[max_index] + thr[max_index + 1]) / 2. if fpr[max_index] > 0.05 else (thr[max_index] + thr[max_index - 1]) / 2.
        return {'threshold': float(T)}
    raise ValueError(f'unknown method {method!r}')


def decide(method: str, params: dict, row: dict) -> tuple[float | None, bool | None]:
    """(score oriented so that higher is more backdoored, flag) of one scored model; None when unavailable."""
    if method in FIXED:
        key, flag = FIXED[method]
        return row.get(key), row.get(flag)
    if method in CLEAN_MAX:
        s = row.get(CLEAN_MAX[method])
        return s, None if s is None else s > params['threshold']
    if method == 'htell':
        s = row.get(f"htell_{params['variant']}")
        return s, None if s is None else s > params['threshold']
    if method == 'dfb':
        if row.get('dfb_ind') is None:
            return None, None
        c = _cos(score_norm(_dfb_vec(row['dfb_ind'])), np.asarray(params['profile']))
        return -c, c < params['threshold']
    if method == 'freeeagle':
        s = row.get('freeeagle')
        return s, None if s is None else s > params['threshold']
    if method == 'prop':
        return row.get('prop'), None
    raise ValueError(f'unknown method {method!r}')
