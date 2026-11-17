"""Score checkpoints with the baseline detectors.

    python -m bdbaselines.score models/*/seed*/*/ --out scores.json [--methods fields dfb htell htell_lc1 clp bnp mmbd freeeagle rownorm]

A MODEL argument is a run directory (model.pt and meta.json, as written by backdoor-zoo) or a model.pt; meta.json gives the
dataset (image size: 32 px, 64 px TinyImageNet, 224 px ImageNet-30, 112 px imagenet30_112; classes; clean images for BNP)
and the label used by bdbaselines.decide: 'clean', 'backdoored' for a usable backdoor (the paper's criterion,
bdzoo.train.usable: clean accuracy at most 0.05 below the dataset's bar and ASR at the attack's bar) and 'failed'
otherwise. A method that does not apply to an architecture (e.g. HTell without an MLP head) is recorded under 'errors'.
Times exclude model loading.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from . import bn, optimization, probes, weights
from .rules import METHODS


def label_of(meta: dict) -> str:
    if meta.get('label'):
        return meta['label']
    if meta.get('attack') == 'clean':
        return 'clean'
    if 'usable' in meta:
        return 'backdoored' if meta['usable'] else 'failed'
    if 'baseline' not in meta:
        return 'failed'
    from bdzoo.train import usable

    meta = {**meta, 'asr_bar': meta.get('asr_bar', meta.get('settings', {}).get('min_asr', 0.9))}  # older fine-tune runs
    return 'backdoored' if usable(meta) else 'failed'


def score_model(path: Path, methods, device: str, data_dir: str, seed: int = 0, fe_dir: Path = optimization.FE_DIR) -> dict:
    from bdzoo import load_checkpoint
    from bdzoo.datasets import info

    run = path if path.is_dir() else path.parent
    meta = json.loads((run / 'meta.json').read_text()) if (run / 'meta.json').exists() else {}
    dataset = meta.get('settings', {}).get('dataset', 'cifar10')
    size, K = info(dataset)['image_size'], info(dataset)['num_classes']
    model, ck = load_checkpoint(run / 'model.pt')
    model = model.eval()
    row = {'model': str(run), 'arch': ck['arch'], 'dataset': dataset, 'label': label_of(meta), 'attack': meta.get('attack'),
           'target': meta.get('settings', {}).get('target'), 'time': {}, 'errors': {}}
    W, b = weights.classifier(model)
    weights.dixon_critical(len(b))  # cached per process; not part of the timed detector
    calls = {'fields': lambda: weights.fields(W, b), 'dfb': lambda: weights.dfb(W, b), 'rownorm': lambda: weights.rownorm(W, b),
             'clp': lambda: bn.clp(model),
             'htell': lambda: probes.htell(model, device, image_size=size),
             'htell_lc1': lambda: probes.htell_lc1(model, device, image_size=size),
             'bnp': lambda: bn.bnp(model, bn.clean_images(dataset, data_dir, 500), device),
             'mmbd': lambda: optimization.mmbd(model, K, size, device, seed),
             'freeeagle': lambda: optimization.freeeagle(model, K, size, device, seed, fe_dir)}
    cpu_first = ('fields', 'dfb', 'rownorm', 'clp')  # read the parameters on the CPU, then the model moves to the device
    for m in sorted(methods, key=lambda m: m not in cpu_first):
        if m not in cpu_first:
            model.to(device)
        t0 = time.perf_counter()
        try:
            row.update(calls[m]())
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            row['time'][m] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - recorded per detector
            row['errors'][m] = f'{type(exc).__name__}: {exc}'
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default='data')
    ap.add_argument('--freeeagle-dir', type=Path, default=optimization.FE_DIR)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    cli = ap.parse_args(argv)
    torch.set_num_threads(cli.threads)
    rows = []
    for p in cli.models:
        try:
            rows.append(score_model(p, cli.methods, cli.device, cli.data, cli.seed, cli.freeeagle_dir))
        except Exception as exc:  # noqa: BLE001 - one failing model must not stop the set
            rows.append({'model': str(p), 'fatal': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-2000:]})
        r = rows[-1]
        print(f"{r['model']}: {r.get('label', 'FATAL')} {', '.join(f'{k} {v:.1f}s' for k, v in r.get('time', {}).items())}"
              + (f" errors: {list(r['errors'])}" if r.get('errors') else ''), flush=True)
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps({'methods': cli.methods, 'seed': cli.seed, 'rows': rows}))


if __name__ == '__main__':
    main()
