"""Detectors that feed random probes through part of the network. Higher scores mean more backdoored.

    htell      HTell (Yu et al. 2026): N random probes injected at the head input z, uniform U[0, 1] ('u') or Gaussian
               N(0, sigma^2) with sigma = max |z| over 32 U[0, 1] noise images / 3 ('g'); per class r_mean = mean softmax
               confidence or r_l2 = L2 norm of the softmax column; score max_c r, target argmax. The variant is chosen on
               labelled configuration models (bdbaselines.rules)
    htell_lc1  HTell with the probes at the input of the last feature block ("LC-1"): PreAct-ResNet, the last residual
               block; VGG, the first convolution after the last max-pool. Same laws, statistics and target. The paper uses
               it on BackdoorBench's PreActResNet18, which has no MLP head (uniform probes, mean statistic)

``htell`` needs the MLP head (head_fc -> head_bn -> ReLU -> fc: preact_resnet18_mlphead, vgg16_mlphead).
"""
from __future__ import annotations

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
