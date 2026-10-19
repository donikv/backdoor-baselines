"""Detectors built on BatchNorm layers, adapted from neuron-level repair to model-level decisions (flag a model when any
evaluated unit would be pruned). Higher scores mean more backdoored.

    clp   CLP (Zheng et al., ECCV 2022; BackdoorBench defense/clp.py): for each Conv2d -> BatchNorm2d pair, the upper bound
          of the channel Lipschitz constant UCLC_k = spectral norm of channel k's kernel (C_in x kh kw) times
          |gamma_k / sqrt(v_k)|; also head_fc with head_bn folded in (UCLC_j = |A_j| ||w_j||). Score: the largest z-score
          (UCLC - layer mean) / layer std over layers; native rule z > 3. Data-free
    bnp   BNP (Zheng et al., NeurIPS 2022): per unit of a BN layer, KL( N(mu_c, v_c) || N(mu_stored, v_stored) ) with
          (mu_c, v_c) the inference-mode moments of the unit's input over 500 clean test images. Score: the largest z-score
          of the KL within the layer (head BN, else the last BN2d); native rule z > 3. Needs clean images
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def clp(model: nn.Module, u: float = 3.0) -> dict:
    zs, conv, layers = [], None, 0
    for _, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and conv is not None:
            scale = (m.weight.double() / m.running_var.double().clamp_min(1e-30).sqrt()).abs()
            k = conv.weight.double().reshape(conv.weight.shape[0], conv.weight.shape[1], -1)
            lips = torch.linalg.matrix_norm(k, ord=2) * scale
            zs.append(float(((lips - lips.mean()) / lips.std()).max()))
            layers += 1
        elif isinstance(m, nn.Conv2d):
            conv = m
    out = {'clp_conv_z': max(zs) if zs else None, 'clp_layers': layers}
    if hasattr(model, 'head_bn'):
        bn = model.head_bn
        A = (bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)).abs()
        lips = A * model.head_fc.weight.double().norm(dim=1)
        out['clp_head_z'] = float(((lips - lips.mean()) / lips.std()).max())
    out['clp_z'] = max(v for v in (out['clp_conv_z'], out.get('clp_head_z')) if v is not None)
    out['clp_flag'] = out['clp_z'] > u
    return out


@torch.no_grad()
def input_moments(model: nn.Module, x: torch.Tensor, names: list[str], device: str, batch: int = 500) -> dict:
    """Per-unit mean and population variance (float64) of each named BN layer's input over x, inference mode."""
    acc = {n: [0, 0.0, 0.0] for n in names}

    def hook(n):
        def f(_, inputs):
            a = inputs[0].double()
            dims = [0] + list(range(2, a.ndim))
            acc[n][0] += a.numel() // a.shape[1]
            acc[n][1] = acc[n][1] + a.sum(dims)
            acc[n][2] = acc[n][2] + (a * a).sum(dims)
        return f

    handles = [model.get_submodule(n).register_forward_pre_hook(hook(n)) for n in names]
    model.eval()
    try:
        for b in range(0, len(x), batch):
            model(x[b:b + batch].to(device))
    finally:
        for h in handles:
            h.remove()
    out = {}
    for n, (k, s1, s2) in acc.items():
        mu = s1 / k
        out[n] = (mu, (s2 / k - mu * mu).clamp_min(0))
    return out


def clean_images(dataset: str, data_dir: str, n: int = 500) -> torch.Tensor:
    """The defender's clean images: the first n of a seed-0 permutation of the dataset's test split."""
    from bdzoo.datasets import load

    x, _, _ = load(dataset, False, data_dir)
    return x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:500]][:n]


def bnp_layer(model: nn.Module) -> str:
    if hasattr(model, 'head_bn'):
        return 'head_bn'
    return [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)][-1]


def bnp(model: nn.Module, x: torch.Tensor, device: str, bn_name: str | None = None, u: float = 3.0) -> dict:
    bn_name = bn_name or bnp_layer(model)
    bn = model.get_submodule(bn_name)
    (mc, vc), = input_moments(model, x, [bn_name], device).values()
    mc, vc = mc.cpu().numpy(), vc.cpu().numpy() + bn.eps
    ms, vs = bn.running_mean.double().cpu().numpy(), bn.running_var.double().cpu().numpy() + bn.eps
    kl = 0.5 * np.log(vs / vc) + (vc + (mc - ms) ** 2) / (2 * vs) - 0.5
    z = (kl - kl.mean()) / max(kl.std(), 1e-300)
    return {'bnp_z': float(z.max()), 'bnp_max': float(kl.max()), 'bnp_unit': int(kl.argmax()), 'bnp_layer': bn_name,
            'bnp_flag': bool(z.max() > u)}
