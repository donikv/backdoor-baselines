import numpy as np
import pytest
import torch
from torch import nn

from bdzoo import build_model
from bdbaselines import bn, optimization, probes, rules, weights
from bdbaselines.score import label_of

SMALL = {'preact_resnet18_mlphead': dict(width=8, head_width=24), 'preact_resnet18': dict(width=8),
         'preact_resnet18_bb': dict(width=8), 'vgg16_mlphead': dict(head_width=24), 'vgg16_gap': {}}


def _model(arch, seed=0, **extra):
    torch.manual_seed(seed)
    m = build_model(arch, num_classes=5, **SMALL[arch], **extra)
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


@pytest.mark.parametrize('arch', ['preact_resnet18_mlphead', 'vgg16_mlphead', 'vgg16_gap', 'preact_resnet18_bb'])
def test_probe_and_bn_detectors_run(arch):
    m = _model(arch)
    if hasattr(m, 'head_fc'):
        assert 0 < probes.htell(m, 'cpu', n_probes=256)['htell_u_mean'] <= 1
    else:
        with pytest.raises(AttributeError):  # HTell at the head needs the MLP head
            probes.htell(m, 'cpu', n_probes=256)
    assert 0 < probes.htell_lc1(m, 'cpu', n_probes=256, batch=128)['htell_lc1_u_mean'] <= 1
    c = bn.clp(m)
    assert np.isfinite(c['clp_z']) and c['clp_layers'] == {'vgg16_mlphead': 13, 'vgg16_gap': 13}.get(arch, c['clp_layers'])
    assert ('clp_head_z' in c) == hasattr(m, 'head_bn')
    out = bn.bnp(m, torch.rand(16, 3, 32, 32), 'cpu')
    assert np.isfinite(out['bnp_z']) and out['bnp_layer'] == ('head_bn' if hasattr(m, 'head_bn') else out['bnp_layer'])


@torch.no_grad()
@pytest.mark.parametrize('arch,kw,size,clamp', [('preact_resnet18_mlphead', {}, 32, False), ('preact_resnet18', {}, 32, False),
                                                ('preact_resnet18_mlphead', {'stem': 'imagenet'}, 64, False),
                                                ('vgg16_gap', {}, 32, True), ('vgg16_mlphead', {}, 112, True),
                                                ('preact_resnet18_bb', {}, 32, False)])
def test_freeeagle_split_reproduces_the_model(arch, kw, size, clamp):
    m = _model(arch, **kw)
    front, later, c = optimization.split_model(m)
    x = torch.rand(2, 3, size, size)
    torch.testing.assert_close(later(front(x)), m(x))
    assert c == clamp
    if arch.startswith('vgg'):  # FreeEagle's VGG-16 default: the input of the fourth stage (256 channels at 1/8 resolution)
        assert front(x).shape[1:] == (256, size // 8, size // 8) and float(front(x).min()) >= 0


def test_labels_follow_the_usable_rule():
    meta = {'attack': 'tact', 'asr_bar': .5, 'settings': {'min_clean_accuracy': .9},
            'baseline': {'clean_accuracy': .86, 'asr_non_target': .8}}
    assert label_of(meta) == 'backdoored'  # 0.04 below the accuracy bar: usable
    assert label_of({**meta, 'baseline': {'clean_accuracy': .84, 'asr_non_target': .8}}) == 'failed'
    assert label_of({**meta, 'usable': False}) == 'failed' and label_of({'attack': 'clean'}) == 'clean'
    assert label_of({'attack': 'badnet', 'label': 'backdoored'}) == 'backdoored'  # BackdoorBench imports carry their label


def test_rules_on_synthetic_scores():
    rng = np.random.default_rng(0)
    cfg = [{'label': 'clean', 'rownorm': 1 + rng.random() * 0.1, 'htell_lc1_u_mean': 0.2 + 0.01 * rng.random(),
            **{f'htell_{v}': 0.2 + 0.01 * rng.random() for v in ('u_mean', 'u_l2', 'g_mean', 'g_l2')}} for _ in range(10)]
    cfg += [{'label': 'backdoored', 'rownorm': 2.0, 'htell_lc1_u_mean': 0.9,
             **{f'htell_{v}': 0.9 for v in ('u_mean', 'u_l2', 'g_mean', 'g_l2')}} for _ in range(5)]
    p = rules.fit('htell', cfg)
    assert p['variant'] == 'u_mean' and p['threshold'] == max(r['htell_u_mean'] for r in cfg if r['label'] == 'clean')
    # the variant with the best configuration AUROC, thresholded at its clean maximum (no FPR quota)
    many = [{'label': 'clean', 'htell_u_mean': .5, 'htell_u_l2': i / 10, 'htell_g_mean': .5, 'htell_g_l2': .5} for i in range(40)]
    many += [{'label': 'backdoored', 'htell_u_mean': .5, 'htell_u_l2': 9., 'htell_g_mean': .5, 'htell_g_l2': .5}]
    p = rules.fit('htell', many)
    assert p['variant'] == 'u_l2' and p['threshold'] == 3.9
    assert rules.decide('htell', p, {'htell_u_l2': 3.95}) == (3.95, True)
    assert rules.decide('htell', p, {'htell_u_l2': 3.9}) == (3.9, False)
    assert rules.decide('rownorm', rules.fit('rownorm', cfg), {'rownorm': 3.0}) == (3.0, True)
    assert rules.decide('clp', {}, {'clp_z': 4.0, 'clp_flag': True}) == (4.0, True)
    assert rules.auroc([2, 3], [1, 2]) == 0.875
