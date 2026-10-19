"""Agreement of these implementations with the released code (fetch it with scripts/fetch_third_party.sh --validation).

    python -m bdbaselines.validate dfb                     DFBScanner's code on the BackdoorBench checkpoints shipped with
                                                           it: largest difference of the 62 indicators, decisions under each
                                                           configuration of its detection_config.json, Dixon's Q decisions
    python -m bdbaselines.validate clp RUN_DIR...          BackdoorBench's CLP_prune against bdbaselines.bn.clp: channels
                                                           pruned at u = 3 per BN layer
    python -m bdbaselines.validate mmbd RUN_DIR...         MM-BD's univ_bd.py (same seed) against bdbaselines.optimization.mmbd
                                                           (10-class 32 px models), and FreeEagle's split against the model

Checkpoints shipped with DFBScanner are read with torch.load(weights_only=True), allowing numpy array and scalar types only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import weights

TP = Path('third_party')


def safe_load(path: Path):
    """torch.load(weights_only=True), allowing numpy globals only (added one at a time as the loader names them)."""
    allowed = []
    for _ in range(60):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:  # noqa: BLE001 - inspect the refusal
            m = re.search(r"GLOBAL ([\w.]+) was not an allowed global|but got <class '([\w.]+)'>", str(exc))
            full = m and (m.group(1) or m.group(2))
            if not full or not full.startswith('numpy') or any(full == a[1] for a in allowed):
                raise
            mod, _, name = full.rpartition('.')
            allowed.append((getattr(importlib.import_module(mod), name), full))
    raise RuntimeError(f'{path}: too many globals')


def dfb_score(ind: np.ndarray, selected: list[int], clean_profile: np.ndarray) -> tuple[float, int]:
    """(1 - cosine similarity of the detection score with the clean profile, target), DFBScanner's detector."""
    d = weights._norm(ind[selected].sum(0))
    cos = float(d @ clean_profile / max(np.linalg.norm(d) * np.linalg.norm(clean_profile), 1e-300))
    return 1 - cos, int(np.argmax(d))


def check_dfb(dfb_dir: Path) -> list[dict]:
    sys.path.insert(0, str(dfb_dir.resolve()))
    cwd = os.getcwd()
    os.chdir(dfb_dir)  # the official class reads detection_config.json from the working directory
    rows = []
    try:
        import DFBScanner as official  # noqa: N813
        from models.model_wrapper import ModelWrapper
        det = official.DFBScanner()
        cfg = det.detection_configuration
        for f in sorted(Path('checkpoints').glob('*/*.pt')):
            m = re.match(r'(cifar10|gtsrb)_(.+)_(badnet|wanet)_L(\d+)\.pt', f.name)
            if not m:
                continue
            ds, name = m.group(1), m.group(2)
            K = 10 if ds == 'cifar10' else 43
            row = {'file': f.name}
            try:
                blob = safe_load(f)
                wrapper = ModelWrapper(name, ds.upper(), K)
                wrapper.model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in blob['model'].items()})
                wrapper.model.eval()
            except Exception as exc:  # noqa: BLE001
                rows.append({**row, 'skipped': f'{type(exc).__name__}: {str(exc)[:200]}'})
                continue
            last = wrapper.obtain_last_layer()
            Wt = last.weight.detach().float()
            W, b = Wt.double().numpy(), last.bias.detach().double().numpy()
            theirs = np.vstack([det.compute_anomaly_indicator(Wt, W.T, b, ind) for ind in weights.DFB_ALL])
            ours = weights.dfb_indicators(W, b)
            row['indicator_maxdiff'] = float(np.nanmax(np.abs(theirs - ours)))
            row['configs'] = {}
            for conf_id, ind_key, t_key, d_key in ((0, None, f'all_{ds}_{name}_t', f'all_{ds}_{name}_clean_dist'),
                                                    (1, 'unique_indicator', f'unique_{ds}_{name}_t', f'unique_{ds}_{name}_clean_dist'),
                                                    (3, f'{ds}_{name}', f'{ds}_{name}_t', f'{ds}_{name}_clean_dist')):
                if t_key not in cfg or d_key not in cfg or (ind_key and ind_key not in cfg):
                    continue
                with open(os.devnull, 'w') as null:
                    off = det.backdoor_detector(wrapper, ds, name, null, using_configure=conf_id)
                sel = [weights.DFB_ALL.index(i) for i in (cfg[ind_key] if ind_key else det.all_metrics)]
                score, tgt = dfb_score(ours, sel, np.asarray(cfg[d_key]))
                mine = tgt if (1 - score) < cfg[t_key] else -1
                row['configs'][conf_id] = {'official': int(off), 'ours': int(mine), 'agree': int(off) == int(mine)}
            with open(os.devnull, 'w') as null:
                dq = det.dq_backdoor_detector(wrapper, null)
            f_ = weights.fields(W, b)
            mine = f_['fields_target'] if f_['fields_q'] > 0.468 else -1  # DFBScanner's Dixon rule uses 0.468
            row['dixon'] = {'official': int(dq), 'ours_at_0.468': int(mine), 'agree': int(dq) == int(mine)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        os.chdir(cwd)
    return rows


def load_clp_prune(src: Path):
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'CLP_prune')
    ns = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)  # noqa: S102 - the reference function only
    return ns['CLP_prune']


@torch.no_grad()
def our_clp_layers(model, u=3.0):
    out, conv = [], None
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and conv is not None:
            scale = (m.weight.double() / m.running_var.double().clamp_min(1e-30).sqrt()).abs()
            k = conv.weight.double().reshape(conv.weight.shape[0], conv.weight.shape[1], -1)
            lips = torch.linalg.matrix_norm(k, ord=2) * scale
            out.append((name, int((lips > lips.mean() + u * lips.std()).sum())))
        elif isinstance(m, nn.Conv2d):
            conv = m
    return out


def check_clp(runs, src: Path) -> list[dict]:
    from bdzoo import load_checkpoint

    clp_prune = load_clp_prune(src)
    rows = []
    for run in runs:
        model, _ = load_checkpoint(Path(run) / 'model.pt')
        ref = copy.deepcopy(model)
        before = {n: m.weight.detach().clone() for n, m in ref.named_modules() if isinstance(m, nn.BatchNorm2d)}
        clp_prune(ref, 3.0)
        theirs = {n: int(((m.weight.detach() == 0) & (before[n] != 0)).sum()) for n, m in ref.named_modules() if n in before}
        ours = dict(our_clp_layers(model))
        layers = sorted(set(theirs) & set(ours))
        rows.append({'run': str(run), 'layers': len(layers), 'agree_layers': sum(theirs[n] == ours[n] for n in layers)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def check_mmbd(runs, mmbd_dir: Path, device: str, seed: int = 0) -> list[dict]:
    from bdzoo import build_model, load_checkpoint
    from bdzoo.datasets import info

    from .optimization import mmbd, split_model

    rows = []
    for run in runs:
        meta = json.loads((Path(run) / 'meta.json').read_text())
        dataset = meta.get('settings', {}).get('dataset', 'cifar10')
        size, K = info(dataset)['image_size'], info(dataset)['num_classes']
        model, ck = load_checkpoint(Path(run) / 'model.pt')
        # both runs must draw the same random images: cuDNN deterministic, and the official script's model built before
        # the seed is set (building a model draws its initial weights from the same generator)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        ours = mmbd(model, K, size, device, seed)
        row = {'run': str(run)}
        if K == 10 and size == 32:  # the official script hard-codes 10 classes and 3 x 32 x 32 images
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / 'm').mkdir()
                torch.save(model.state_dict(), Path(tmp) / 'm' / 'model.pth')
                sys.path.insert(0, str(mmbd_dir.resolve()))
                import src.resnet as official_resnet
                original = official_resnet.ResNet18
                prebuilt = build_model(ck['arch'], **ck.get('arch_kwargs', {}))
                official_resnet.ResNet18 = lambda: prebuilt
                argv, sys.argv = sys.argv, ['univ_bd.py', '--model_dir', 'm']
                try:
                    os.chdir(tmp)
                    torch.manual_seed(seed)
                    runpy.run_path(str(mmbd_dir.resolve() / 'univ_bd.py'), run_name='__main__')
                    official = np.load(Path(tmp) / 'results.npy')
                finally:
                    os.chdir(cwd)
                    sys.argv = argv
                    official_resnet.ResNet18 = original
            row['mmbd_max_abs_diff'] = float(np.max(np.abs(official - np.array(ours['mmbd_stats']))))
        model = model.to(device).eval()
        front, later, _ = split_model(model)
        with torch.no_grad():
            x = torch.rand(8, 3, size, size, device=device)
            row['freeeagle_split_max_abs_diff'] = float((later(front(x)) - model(x)).abs().max())
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('check', choices=['dfb', 'clp', 'mmbd'])
    ap.add_argument('runs', nargs='*', type=Path, help='run directories (clp, mmbd)')
    ap.add_argument('--third-party', type=Path, default=TP)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path)
    cli = ap.parse_args(argv)
    if cli.check == 'dfb':
        rows = check_dfb(cli.third_party / 'DFBScanner')
    elif cli.check == 'clp':
        rows = check_clp(cli.runs, cli.third_party / 'backdoorbench' / 'clp.py')
    else:
        rows = check_mmbd(cli.runs, cli.third_party / 'MM-BD', cli.device)
    if cli.out:
        cli.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
