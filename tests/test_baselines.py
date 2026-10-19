import numpy as np
import pytest
import torch
from torch import nn

from bdzoo import build_model
from bdbaselines import bn, optimization, probes, rules, weights


def _model(arch, seed=0):
    torch.manual_seed(seed)
    kw = {'cnn_mlphead': dict(widths=(8, 16, 16), head_width=24), 'preact_resnet18_mlphead': dict(width=8, head_width=24),
          'preact_resnet18_bb': dict(width=8), 'cnn_gap': dict(widths=(8, 16, 16))}[arch]
    m = build_model(arch, num_classes=5, **kw)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.weight.uniform_(0.5, 1.5, generator=g)
                mod.running_var.uniform_(0.2, 3.0, generator=g)
                mod.num_batches_tracked.fill_(10)
    return m.eval()


def test_dixon_critical_value_matches_the_table():
    assert abs(weights.dixon_critical(10) - 0.466) < 0.005


def test_dfb_indicators_and_rownorm():
    W, b = np.random.default_rng(0).standard_normal((10, 32)), np.zeros(10)
    W[3] *= 3
    ind = weights.dfb_indicators(W, b)
    assert ind.shape == (62, 10) and ind.min() >= 0 and ind.max() <= 1
    assert weights.rownorm(W)['rownorm_target'] == 3 and weights.fields(W, b)['fields_crit'] > 0


@pytest.mark.parametrize('arch', ['cnn_mlphead', 'preact_resnet18_mlphead', 'preact_resnet18_bb'])
def test_probe_and_bn_detectors_run(arch):
    m = _model(arch)
    if hasattr(m, 'head_fc'):
        assert 0 < probes.htell(m, 'cpu', n_probes=256)['htell_u_mean'] <= 1
    assert 0 < probes.htell_lc1(m, 'cpu', n_probes=256, batch=128)['htell_lc1_u_mean'] <= 1
    assert 0 <= probes.prop(m, 'cpu', n=64, batch=32)['prop'] <= 1
    assert np.isfinite(bn.clp(m)['clp_z'])
    assert np.isfinite(bn.bnp(m, torch.rand(16, 3, 32, 32), 'cpu')['bnp_z'])
    assert torch.relu(torch.tensor([-1.0, 2.0])).tolist() == [0.0, 2.0]  # ProP restored torch.relu


@torch.no_grad()
@pytest.mark.parametrize('arch', ['cnn_mlphead', 'cnn_gap', 'preact_resnet18_mlphead', 'preact_resnet18_bb'])
def test_freeeagle_split_reproduces_the_model(arch):
    m = _model(arch)
    front, later, _ = optimization.split_model(m)
    x = torch.rand(4, 3, 32, 32)
    torch.testing.assert_close(later(front(x)), m(x))


def test_rules_on_synthetic_scores():
    rng = np.random.default_rng(0)
    cfg = [{'label': 'clean', 'rownorm': 1 + rng.random() * 0.1, 'htell_lc1_u_mean': 0.2 + 0.01 * rng.random(),
            **{f'htell_{v}': 0.2 + 0.01 * rng.random() for v in ('u_mean', 'u_l2', 'g_mean', 'g_l2')}} for _ in range(10)]
    cfg += [{'label': 'backdoored', 'rownorm': 2.0, 'htell_lc1_u_mean': 0.9,
             **{f'htell_{v}': 0.9 for v in ('u_mean', 'u_l2', 'g_mean', 'g_l2')}} for _ in range(5)]
    p = rules.fit('htell', cfg)
    assert p['variant'] == 'u_mean' and p['threshold'] == max(r['htell_u_mean'] for r in cfg if r['label'] == 'clean')
    assert rules.decide('rownorm', rules.fit('rownorm', cfg), {'rownorm': 3.0}) == (3.0, True)
    assert rules.decide('clp', {}, {'clp_z': 4.0, 'clp_flag': True}) == (4.0, True)
    assert rules.auroc([2, 3], [1, 2]) == 0.875
