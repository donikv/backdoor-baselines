"""Data-free detectors that optimise inputs or features. Higher scores mean more backdoored.

    mmbd       MM-BD (Wang et al., IEEE S&P 2024), the official univ_bd.py (github.com/wanghangpsu/MM-BD, commit 903338d)
               reproduced line by line: for every class t, 30 uniform random images in [0, 1] take up to 300 SGD steps
               (lr 1e-2, momentum 0.2; the optimizer is re-created every step, as in the official loop) on the loss
               -f_t + max_{k != t} f_k, stopping when the relative change of the loss is below 1e-5; the class statistic is
               the largest margin f_t - max_{k != t} f_k over the 30 images. A gamma law fitted to the other classes'
               statistics gives p = 1 - F(r_max)^K; native rule p <= 0.05, target argmax. Score -log10 p. The random
               images are seeded (the official script draws them unseeded).
    freeeagle  FreeEagle (Fu et al., USENIX Security 2023): its own functions (backdoor_inspection_new.py of
               github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle, commit d045822, MIT license; fetch
               it with scripts/fetch_third_party.sh) run on the later part of the model with the official defaults (one
               dummy per class; Adam lr 1e-2, weight decay 0.005, 1000 steps; softmax posteriors). The split follows the
               paper's rule for small models ("around the middle layer") and the code's ResNet-18 default (the input of the
               second stage): CNN at the input of the second max-pool (a ReLU output: clamped at 0, as the official
               bound_on); PreAct-ResNets at the input of the second stage (a residual sum: not clamped). Metric
               M = (max v - Q3) / (Q3 - Q1) over the column means v of the posterior matrix; the released score is
               |M - 1| (bdbaselines.rules); target argmax v.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

FE_DIR = Path(os.environ.get('FREEEAGLE_DIR', 'third_party/FreeEagle'))


def mmbd(model, n_classes: int, size: int, device: str, seed: int = 0, n_images: int = 30, n_steps: int = 300) -> dict:
    import torch
    import torch.nn.functional as F
    from scipy.stats import gamma

    torch.manual_seed(seed)
    model = model.to(device).eval()
    res = []
    for t in range(n_classes):
        images = torch.rand([n_images, 3, size, size]).to(device)
        images.requires_grad = True
        last_loss = 1000
        labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
        onehot_label = F.one_hot(labels, num_classes=n_classes)
        for _ in range(n_steps):
            optimizer = torch.optim.SGD([images], lr=1e-2, momentum=0.2)
            optimizer.zero_grad()
            outputs = model(torch.clamp(images, min=0, max=1))
            loss = -1 * torch.sum((outputs * onehot_label)) \
                + torch.sum(torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
            loss.backward(retain_graph=True)
            optimizer.step()
            if abs(last_loss - loss.item()) / abs(last_loss) < 1e-5:
                break
            last_loss = loss.item()
        res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)
                             - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
    stats = np.array(res)
    ind_max = int(np.argmax(stats))
    r_eval = float(np.amax(stats))
    r_null = np.delete(stats, ind_max)
    shape, loc, scale = gamma.fit(r_null)
    pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null) + 1)
    # -log10 p without underflow: 1 - F^K = K (1 - F) (1 + O(1 - F)) when F -> 1
    if pv > 1e-12:
        score = -math.log10(pv)
    else:
        score = -(math.log10(len(r_null) + 1) + gamma.logsf(r_eval, a=shape, loc=loc, scale=scale) / math.log(10))
    return {'mmbd_stats': stats.tolist(), 'mmbd_stat': r_eval, 'mmbd_pv': float(pv), 'mmbd_score': float(score),
            'mmbd_target': ind_max, 'mmbd_flag': bool(pv <= 0.05), 'mmbd_seed': seed}


def split_model(model):
    """(front, later, clamp) with later(front(x)) = model(x): FreeEagle's classifier part and its input."""
    import torch
    from torch import nn

    name = type(model).__name__
    if name in ('CNNMLPHead', 'CNNGAP'):
        pools = [i for i, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
        cut = pools[1]  # the input of the second max-pool: after the second convolution block (a ReLU output)
        head = (lambda z: model.fc(model.head_relu(model.head_bn(model.head_fc(z))))) if hasattr(model, 'head_fc') else model.fc
        front = lambda x: model.features[:cut](model.norm(x))  # noqa: E731
        later = lambda h: head(model.pool(model.features[cut:](h)).flatten(1))  # noqa: E731
        return front, later, True
    if name in ('PreActResNet', 'PreActResNetMLPHead'):
        cut = next(i for i, b in enumerate(model.layers) if b.conv1.stride[0] == 2)  # the first block of the second stage
        head = (lambda z: model.fc(model.head_relu(model.head_bn(model.head_fc(z))))) if hasattr(model, 'head_fc') else model.fc
        front = lambda x: model.layers[:cut](model.stem(x))  # noqa: E731
        later = lambda h: head(torch.flatten(model.pool(model.relu(model.bn(model.layers[cut:](h)))), 1))  # noqa: E731
        return front, later, False
    if name == 'PreActResNetBB':
        front = lambda x: model.layer1(model.conv1(model.norm(x)))  # noqa: E731
        later = lambda h: model.linear(torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(h)))), 1))  # noqa: E731
        return front, later, False
    raise ValueError(f'no FreeEagle split for {name}')


def _freeeagle_module(fe_dir: Path = FE_DIR):
    import importlib.machinery
    import types

    import torch  # noqa: F401 - torch and torchvision before the stubs (torch._dynamo inspects tqdm's import spec)
    import torchvision  # noqa: F401
    code = Path(fe_dir) / 'MLBackdoorDetection'
    if not (code / 'backdoor_inspection_new.py').exists():
        raise FileNotFoundError(f'FreeEagle code not found under {fe_dir} (run scripts/fetch_third_party.sh or set FREEEAGLE_DIR)')
    for stub in ('seaborn', 'tqdm'):  # plotting and progress imports of the official module
        if stub not in sys.modules:
            m = types.ModuleType(stub)
            m.__spec__ = importlib.machinery.ModuleSpec(stub, None)
            sys.modules[stub] = m
    sys.modules['tqdm'].tqdm = getattr(sys.modules['tqdm'], 'tqdm', None)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import backdoor_inspection_new as fe
    return fe


class _Later:
    """The classifier part as a callable module for the official functions (they call .eval(), .cuda() and model(x))."""

    def __init__(self, later, module):
        self.later, self.module = later, module

    def eval(self):
        self.module.eval()
        return self

    def cuda(self):
        return self

    def __call__(self, x):
        return self.later(x)


def freeeagle(model, n_classes: int, size: int, device: str, seed: int = 0, fe_dir: Path = FE_DIR) -> dict:
    import torch

    fe = _freeeagle_module(fe_dir)
    model = model.to(device).eval()
    front, later, clamp = split_model(model)
    opt = argparse.Namespace(bound_on=clamp, num_dummy=1, metric='softmax_score', n_cls=n_classes, num_important_neurons=5)
    with torch.no_grad():
        template = torch.ones_like(front(torch.ones(1, 3, size, size, device=device)))
    part = _Later(later, model)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    dummies = [[fe.optimize_inner_embedding(opt, part, template, c)] for c in range(n_classes)]
    mat = np.zeros((n_classes, n_classes))
    for source in range(n_classes):
        row = fe.compute_metrics_one_source(opt, part, source, dummies)
        mat[source] = np.where(row >= 0., row, 0.)
    tendency = np.average(mat, axis=0) * n_classes / (n_classes - 1)
    metric = float(fe.compute_metrics_for_array(tendency))
    for p in model.parameters():
        p.requires_grad_(True)
    return {'freeeagle_metric': metric, 'freeeagle': abs(metric - 1.0), 'freeeagle_target': int(np.argmax(tendency)),
            'freeeagle_tendency': tendency.tolist(), 'freeeagle_clamp': clamp, 'freeeagle_seed': seed}
