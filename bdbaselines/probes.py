"""Detectors that feed random probes through part of the network. Higher scores mean more backdoored.

    htell      HTell (Yu et al. 2026): N random probes injected at the head input z, uniform U[0, 1] ('u') or Gaussian
               N(0, sigma^2) with sigma = max |z| over 32 U[0, 1] noise images / 3 ('g'); per class r_mean = mean softmax
               confidence or r_l2 = L2 norm of the softmax column; score max_c r, target argmax. The variant is chosen on
               labelled configuration models (bdbaselines.rules)
    htell_lc1  HTell with the probes at the input of the last feature block ("LC-1"): CNN, the first conv of the last
               stage; PreAct-ResNet, the last residual block. Same laws, statistics and target
    prop       ProP (Ren et al. 2024): Gaussian noise of variance sigma^2 = 1e4 added after every ReLU while forwarding
               random U[0, 1] inputs; class histogram P of the argmax; benign score beta = (1 - p_max) s_rest / s_all;
               score 1 - beta, target argmax P
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def head_logits(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return model.fc(model.head_relu(model.head_bn(model.head_fc(z))))


@torch.no_grad()
def head_inputs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    captured = []
    h = model.head_fc.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
    try:
        model(x)
    finally:
        h.remove()
    return captured[0]


@torch.no_grad()
def htell(model: nn.Module, device: str, n_probes: int = 10000, seed: int = 0, image_size: int = 32) -> dict:
    model = model.eval()
    gen = torch.Generator().manual_seed(seed)
    d = model.head_fc.in_features
    noise = torch.rand(32, 3, image_size, image_size, generator=gen).to(device)
    sigma = float(head_inputs(model, noise).abs().max()) / 3
    out = {'htell_sigma': sigma}
    for law, z in (('u', torch.rand(n_probes, d, generator=gen)), ('g', sigma * torch.randn(n_probes, d, generator=gen))):
        p = torch.softmax(head_logits(model, z.to(device)).double(), 1)
        for stat, r in (('mean', p.mean(0)), ('l2', p.norm(dim=0))):
            out[f'htell_{law}_{stat}'] = float(r.max())
            out[f'htell_{law}_{stat}_target'] = int(r.argmax())
    return out


def last_block(model: nn.Module) -> nn.Module:
    """The last feature block (HTell's LC-1 probe point)."""
    if hasattr(model, 'layers'):
        return model.layers[-1]
    if hasattr(model, 'layer4'):  # BackdoorBench's PreActResNet18
        return model.layer4[-1]
    mods = list(model.features)
    last_pool = max(i for i, m in enumerate(mods) if isinstance(m, nn.MaxPool2d))
    return mods[last_pool + 1]


@torch.no_grad()
def htell_lc1(model: nn.Module, device: str, n_probes: int = 10000, seed: int = 0, image_size: int = 32, batch: int = 1000) -> dict:
    model = model.eval()
    gen = torch.Generator().manual_seed(seed)
    block = last_block(model)
    seen = []
    h = block.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].detach()))
    try:
        model(torch.rand(32, 3, image_size, image_size, generator=gen).to(device))
    finally:
        h.remove()
    shape, sigma = seen[0].shape[1:], float(seen[0].abs().max()) / 3
    out = {'htell_lc1_sigma': sigma}
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    for law in ('u', 'g'):
        probs = []
        for s in range(0, n_probes, batch):
            n = min(batch, n_probes - s)
            z = torch.rand(n, *shape, generator=gen) if law == 'u' else sigma * torch.randn(n, *shape, generator=gen)
            z = z.to(device)
            h = block.register_forward_pre_hook(lambda m, inp, z=z: (z,))
            try:
                probs.append(torch.softmax(model(dummy.expand(n, -1, -1, -1)).double(), 1))
            finally:
                h.remove()
        p = torch.cat(probs)
        for stat, r in (('mean', p.mean(0)), ('l2', p.norm(dim=0))):
            out[f'htell_lc1_{law}_{stat}'] = float(r.max())
            out[f'htell_lc1_{law}_{stat}_target'] = int(r.argmax())
    return out


@contextlib.contextmanager
def activation_noise(model: nn.Module, sigma: float, gen: torch.Generator):
    """a = phi(z) + sigma xi after every ReLU: module ReLUs by hooks, functional torch.relu (PreAct blocks) by patching."""
    def add(t):
        return t + sigma * torch.randn(t.shape, generator=gen, device=gen.device, dtype=t.dtype)

    handles = [m.register_forward_hook(lambda mod, inp, out: add(out)) for m in model.modules() if isinstance(m, nn.ReLU)]
    relu = torch.relu
    torch.relu = lambda t: add(relu(t))
    try:
        yield
    finally:
        torch.relu = relu
        for h in handles:
            h.remove()


@torch.no_grad()
def prop(model: nn.Module, device: str, sigma2: float = 1e4, n: int = 1024, seed: int = 0, image_size: int = 32,
         batch: int = 256) -> dict:
    model = model.eval()
    gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1)
    counts = None
    with activation_noise(model, float(np.sqrt(sigma2)), noise_gen):
        for s in range(0, n, batch):
            x = torch.rand(min(batch, n - s), 3, image_size, image_size, generator=gen).to(device)
            logits = model(x)
            c = np.bincount(logits.argmax(1).cpu().numpy(), minlength=logits.shape[1])
            counts = c if counts is None else counts + c
    P = counts / counts.sum()
    top = int(np.argmax(P))
    s_all, s_rest = P.std(), np.delete(P, top).std()
    beta = (1 - P[top]) * (s_rest / s_all) if s_all > 0 else 1.0
    return {'prop': float(1 - beta), 'prop_target': top, 'prop_pmax': float(P[top])}
