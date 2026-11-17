import math

import numpy as np
import torch
from torch import nn

from bdbaselines import inputs


class Constant(nn.Module):
    """Logits that ignore the input: class ``c`` always wins (or all classes tie)."""

    def __init__(self, K=4, c=None):
        super().__init__()
        self.K, self.c = K, c

    def forward(self, x):
        out = torch.zeros(len(x), self.K)
        if self.c is not None:
            out[:, self.c] = 5.0
        return out


def _bn_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1),
                         nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).eval()


def test_scale_up_spc():
    x = torch.rand(6, 3, 8, 8)
    assert torch.all(inputs.scale_up(Constant(c=2), x, torch.full((6,), 2), 4) == 1)
    assert torch.all(inputs.scale_up(Constant(c=2), x, torch.zeros(6, dtype=torch.long), 4) == 0)


def test_ibd_psc_amplification_is_restored():
    m = _bn_net()
    before = [(b.weight.clone(), b.bias.clone()) for b in (m[1], m[4])]
    amp = inputs.Amplify(m)
    with amp(1):
        assert torch.allclose(m[4].weight, before[1][0] * 1.5) and torch.equal(m[1].weight, before[0][0])
    with amp(5):  # more layers than the model has: all of them
        assert torch.allclose(m[1].bias, before[0][1] * 1.5)
    assert all(torch.equal(b.weight, w) and torch.equal(b.bias, c) for b, (w, c) in zip((m[1], m[4]), before))
    x, y = torch.rand(8, 3, 8, 8), torch.zeros(8, dtype=torch.long)
    psc = inputs.ibd_psc(m, amp, 1, x, m(x).argmax(1), 4)
    assert psc.shape == (8,) and torch.all((psc >= 0) & (psc <= 1))
    assert inputs.ibd_start(Constant(c=0), inputs.Amplify(Constant()), x, y, 4) == -1  # no BN2d: no start layer


def test_strip_entropy_and_threshold():
    x, overlay = torch.rand(5, 3, 8, 8), torch.rand(20, 3, 8, 8)
    h = inputs.strip(Constant(K=4), x, overlay, torch.Generator().manual_seed(0), 300)
    assert torch.allclose(h, torch.full((5,), math.log2(4), dtype=torch.float64))  # uniform predictions: log2 K
    a = inputs.strip(Constant(c=1), x, overlay, torch.Generator().manual_seed(0), 300)
    assert torch.all(a < h)
    v = np.random.default_rng(0).normal(2, .5, 1000)
    assert np.isclose(inputs.strip_threshold(v), v.mean() - 2.3263478740408408 * v.std())


def _B(n=200, K=4, target=1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    pc, pt = y.copy(), np.where(y != target, target, y)
    return {'y': y, 'eval_pos': np.arange(n), 'num_classes': K, 'pred_clean': pc, 'pred_trig': pt,
            'spc_clean': np.where(rng.random(n) < .1, 1.0, .2), 'spc_trig': np.where(rng.random(n) < .7, 1.0, .2),
            'strip_val': rng.normal(2, .3, 100), 'named': [target]}


def test_rates_all_to_one_and_clean():
    meta = {'attack': 'patch', 'settings': {'target': 1, 'label_mode': 'dirty', 'trigger_params': {}}}
    B = _B()
    first = np.flatnonzero(B['pred_clean'] == 1)
    other = np.flatnonzero(B['pred_clean'] != 1)[:30]
    B['strip_clean_pos'] = np.sort(np.r_[first, other])
    B['strip_clean'] = np.where(np.arange(len(B['strip_clean_pos'])) % 4 == 0, 0.0, 2.0)
    B['strip_trig'] = np.r_[np.zeros(40), np.full(10, 2.0)]
    r = inputs.rates(B, meta)
    pos = B['y'] != 1
    assert np.isclose(r['scale_up']['tpr'], (B['spc_trig'][pos] >= .5).mean())
    assert np.isclose(r['scale_up']['rej'], (B['spc_clean'] >= .5).mean()) and 'ibd_psc' not in r
    assert np.isclose(r['strip']['tpr'], .8)
    low = B['strip_clean'] < inputs.strip_threshold(B['strip_val'])
    a = np.isin(B['pred_clean'][B['strip_clean_pos']], [1])
    n_b = int((B['pred_clean'] != 1).sum())
    assert np.isclose(r['strip']['rej'], (low[a].sum() + n_b * low[~a].mean()) / 200)  # the sample weighted to all inputs
    clean = inputs.rates({k: v for k, v in B.items() if k not in ('pred_trig', 'spc_trig', 'strip_trig')},
                         {'attack': 'clean', 'settings': {}})
    assert set(clean['scale_up']) == {'rej', 'lost'}


def test_rates_all_to_all():
    meta = {'attack': 'patch', 'settings': {'target': 0, 'label_mode': 'all2all', 'trigger_params': {}}}
    B = _B()
    B['pred_trig'] = (B['y'] + 1) % 4
    B['strip_clean_pos'] = np.arange(0, 200, 2)
    B['strip_clean'], B['strip_trig'] = np.full(100, 2.0), np.zeros(50)
    r = inputs.rates(B, meta)
    assert np.isclose(r['scale_up']['tpr'], (B['spc_trig'] >= .5).mean()) and r['strip'] == {'tpr': 1.0, 'rej': 0.0, 'lost': 0.0}
    assert 'fpr_t' not in r['scale_up']


def test_summarise():
    s = inputs.summarise({'tpr': .5, 'rej': .1, 'lost': .05}, .9, .8)
    assert np.isclose(s['ba'], .7) and np.isclose(s['asr_after'], .4) and np.isclose(s['accuracy_after'], .85)
