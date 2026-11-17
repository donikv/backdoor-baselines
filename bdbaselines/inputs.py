"""Published detectors of triggered test inputs, as compared with BAND's test-time filter in Table 1b and Table 9 of the paper
(Appendix L.2), each at its published setting and threshold:

    SCALE-UP   (Guo et al. 2023; BackdoorBox core/defenses/SCALE_UP.py), data-free: the input scaled by 3, 5, 7, 9 and 11 and
               clipped to [0, 1]; SPC = the share of the scaled copies predicted as the input's own prediction; flagged at
               SPC >= 0.5. Needs nothing.
    IBD-PSC    (Hou et al. 2024; BackdoorBox core/defenses/IBD_PSC.py): the weight and bias of the last k BatchNorm2d layers
               multiplied by 1.5; the start layer is the first k in 1..L-1 at which the error on labelled clean validation
               inputs exceeds xi = 0.6; PSC = the mean over k = start+1..start+5 of the softmax probability of the original
               prediction; flagged at PSC >= 0.9. Needs labelled clean inputs; when no k reaches xi it does not run.
    STRIP      (Gao et al. 2019; the authors' notebook, garrisongys/STRIP): the input plus a clean overlay image on [0, 1],
               no clipping, 100 overlays drawn per input; entropy = -sum p log2 p / 100; flagged below the 1% point of a
               normal fitted to the entropies of the validation inputs. Needs clean images.

The stream is bdzoo.streams' (the clean subset of the test set and its triggered copies), with the clean validation and
overlay images of bdzoo.streams.detector_split; where those come from the subset (ImageNet-30, CIFAR-100, TinyImageNet), the
detectors are evaluated on the rest of it. Rates per model: TPR = the share of the hits (triggered inputs the model assigns to
the target; all-to-all: to (y + 1) mod K) flagged, clean rejections = the share of the clean inputs flagged, and (all-to-one)
the FPR among the clean inputs predicted as the target. STRIP scores a sample to save computation: all-to-one every clean input
predicted as a named class (the target and ``--named``) and 500 others, whose rejections are weighted to all clean inputs, and
500 hits; all-to-all 1,000 clean inputs and 500 hits drawn uniformly.

    python -m bdbaselines.inputs RUN ... [--named 3 7] [--save inputs.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SCALES = (3, 5, 7, 9, 11)
SPC_THRESHOLD = 0.5
IBD_N, IBD_XI, IBD_SCALE, IBD_THRESHOLD = 5, 0.6, 1.5, 0.9
STRIP_N = 100
STRIP_Z = -2.3263478740408408  # the 1% point of a standard normal
STRIP_OTHER, STRIP_TRIG = 500, 500  # all-to-one: clean inputs of the classes not named, and hits, that STRIP scores
STRIP_CLEAN_A2A = 1000              # all-to-all: clean inputs STRIP scores
MIN_TARGET = 20                     # clean inputs predicted as the target a model needs for its rates
METHODS = ('scale_up', 'ibd_psc', 'strip')


@torch.no_grad()
def logits(model, x, batch):
    return torch.cat([model(x[b:b + batch]) for b in range(0, len(x), batch)])


@torch.no_grad()
def scale_up(model, x, ref, batch):
    """SPC: share of the scaled copies predicted as ``ref`` (the original predictions)."""
    spc = torch.zeros(len(x), device=x.device)
    for s in SCALES:
        spc += logits(model, torch.clip(x * s, 0.0, 1.0), batch).argmax(1) == ref
    return spc / len(SCALES)


class Amplify:
    """IBD-PSC's parameter-amplified model: the last k BatchNorm2d layers' weight and bias times IBD_SCALE (restored on exit)."""

    def __init__(self, model):
        self.bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    def __call__(self, k):
        self.k = min(k, len(self.bns))
        return self

    def __enter__(self):
        self.saved = [(m.weight.data.clone(), m.bias.data.clone()) for m in self.bns[len(self.bns) - self.k:]]
        for m in self.bns[len(self.bns) - self.k:]:
            m.weight.data *= IBD_SCALE
            m.bias.data *= IBD_SCALE

    def __exit__(self, *a):
        for m, (w, b) in zip(self.bns[len(self.bns) - self.k:], self.saved):
            m.weight.data.copy_(w)
            m.bias.data.copy_(b)


@torch.no_grad()
def ibd_start(model, amp, xv, yv, batch) -> int:
    """The first k whose amplification makes the validation error exceed xi; -1 when there is none."""
    for k in range(1, len(amp.bns)):
        with amp(k):
            err = float((logits(model, xv, batch).argmax(1) != yv).float().mean())
        if err > IBD_XI:
            return k
    return -1


@torch.no_grad()
def ibd_psc(model, amp, start, x, pred, batch):
    psc = torch.zeros(len(x), device=x.device)
    for layer_index in range(start, start + IBD_N):
        with amp(layer_index + 1):
            p = torch.softmax(logits(model, x, batch), 1)
        psc += p[torch.arange(len(x), device=x.device), pred]
    return psc / IBD_N


@torch.no_grad()
def strip(model, x, overlay, gen, batch):
    """Entropy of the predictions for x + overlay over STRIP_N overlays drawn per input from ``gen``."""
    per = max(1, batch // STRIP_N)
    out = []
    for b in range(0, len(x), per):
        xb = x[b:b + per]
        j = torch.randint(len(overlay), (len(xb), STRIP_N), generator=gen).to(x.device)
        p = torch.softmax(model((xb[:, None] + overlay[j]).flatten(0, 1)), 1).double()
        h = torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))
        out.append(-h.view(len(xb), -1).sum(1) / STRIP_N)
    return torch.cat(out)


def strip_threshold(strip_val) -> float:
    """The 1% point of the normal fitted (mean, standard deviation with ddof 0) to the validation entropies."""
    v = np.asarray(strip_val)
    return float(v.mean() + STRIP_Z * v.std())


def scores(model, meta: dict, data_dir: str, device: str, named=()) -> dict:
    """Every score the rates need for one model on its stream (arrays as numpy)."""
    from bdzoo import streams

    st = meta['settings']
    X, Y = streams.test_images(st['dataset'], data_dir)
    idx = streams.subset(len(X), streams.IMAGES[st['dataset']])
    val, ovl, ev = streams.detector_split(len(X), idx)
    x, y = X[torch.as_tensor(idx)], Y[torch.as_tensor(idx)].numpy()
    batch = 2000 if x.shape[-1] <= 64 else 200
    bd = meta['attack'] != 'clean'
    a2a = bd and streams.label_mode(meta) == 'all2all'
    xs, xv, yv, xo = x.to(device), X[torch.as_tensor(val)].to(device), Y[torch.as_tensor(val)].to(device), X[torch.as_tensor(ovl)].to(device)
    out = {'y': y, 'eval_pos': ev, 'val_idx': val, 'overlay_idx': ovl}
    pc_all = logits(model, xs, batch).argmax(1)
    out['pred_clean'] = pc_all.cpu().numpy()
    K = int(logits(model, xs[:1], 1).shape[1])
    out['num_classes'] = K
    evt = torch.as_tensor(ev, device=device)
    sets = {'clean': (xs[evt], pc_all[evt])}
    if bd:
        xt = streams.triggered(x, meta).to(device)
        pt_all = logits(model, xt, batch).argmax(1)
        out['pred_trig'] = pt_all.cpu().numpy()
        sets['trig'] = (xt[evt], pt_all[evt])
    for s, (xx, pp) in sets.items():
        out[f'spc_{s}'] = scale_up(model, xx, pp, batch).cpu().numpy()
    amp = Amplify(model)
    start = ibd_start(model, amp, xv, yv, batch)
    out['ibd_start'] = start
    if start > 0:
        for s, (xx, pp) in sets.items():
            out[f'psc_{s}'] = ibd_psc(model, amp, start, xx, pp, batch).cpu().numpy()
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    pc = out['pred_clean'][ev]
    if a2a:
        cpos = np.sort(rng.choice(len(ev), STRIP_CLEAN_A2A, replace=False))
        hit = np.flatnonzero(out['pred_trig'][ev] == (y[ev] + 1) % K)
    else:
        named = sorted({int(c) for c in named} | ({int(st['target'])} if bd else set()))
        first = np.flatnonzero(np.isin(pc, named))
        other = np.setdiff1d(np.arange(len(ev)), first)
        cpos = np.sort(np.concatenate([first, rng.choice(other, min(STRIP_OTHER, len(other)), replace=False)]))
        out['named'] = named
        if bd:
            hit = np.flatnonzero(streams.hits(meta, y[ev], out['pred_trig'][ev], K))
    out['strip_val'] = strip(model, xv, xo, gen, batch).cpu().numpy()
    out['strip_clean_pos'] = cpos
    out['strip_clean'] = strip(model, sets['clean'][0][torch.as_tensor(cpos, device=device)], xo, gen, batch).cpu().numpy()
    if bd:
        tpos = np.sort(rng.choice(hit, min(STRIP_TRIG, len(hit)), replace=False))
        out['strip_trig_pos'] = tpos
        out['strip_trig'] = strip(model, sets['trig'][0][torch.as_tensor(tpos, device=device)], xo, gen, batch).cpu().numpy()
    return out


def rates(B: dict, meta: dict) -> dict | None:
    """{method: {tpr, rej, lost (and fpr_t, all-to-one)}} at the published thresholds on the evaluation inputs; lost = the
    share of the clean inputs correctly classified and flagged. None when no hit is left or (all-to-one) fewer than
    MIN_TARGET clean inputs are predicted as the target. IBD-PSC is absent when its layer search found no start layer."""
    from bdzoo import streams

    ev, y, K = B['eval_pos'], B['y'], B['num_classes']
    pc = B['pred_clean'][ev]
    ok = pc == y[ev]
    bd = meta['attack'] != 'clean'
    a2a = bd and streams.label_mode(meta) == 'all2all'
    thr = strip_threshold(B['strip_val'])
    out = {}
    if a2a:
        hit = B['pred_trig'][ev] == (y[ev] + 1) % K

        def add(name, flag_c, flag_t, okc=ok):
            out[name] = {'tpr': float(flag_t.mean()), 'rej': float(flag_c.mean()), 'lost': float((flag_c & okc).mean())}

        add('scale_up', B['spc_clean'] >= SPC_THRESHOLD, B['spc_trig'][hit] >= SPC_THRESHOLD)
        if 'psc_clean' in B:
            add('ibd_psc', B['psc_clean'] >= IBD_THRESHOLD, B['psc_trig'][hit] >= IBD_THRESHOLD)
        add('strip', B['strip_clean'] < thr, B['strip_trig'] < thr, ok[B['strip_clean_pos']])
        return out
    if bd:
        target = int(meta['settings']['target'])
        pos, in_t = streams.hits(meta, y[ev], B['pred_trig'][ev], K), pc == target
        if not pos.any() or in_t.sum() < MIN_TARGET:
            return None

    def add(name, flag_c, flag_t):
        out[name] = {'rej': float(flag_c.mean()), 'lost': float((flag_c & ok).mean())}
        if bd:
            out[name].update(tpr=float(flag_t[pos].mean()), fpr_t=float(flag_c[in_t].mean()))

    add('scale_up', B['spc_clean'] >= SPC_THRESHOLD, B['spc_trig'] >= SPC_THRESHOLD if bd else None)
    if 'psc_clean' in B:
        add('ibd_psc', B['psc_clean'] >= IBD_THRESHOLD, B['psc_trig'] >= IBD_THRESHOLD if bd else None)
    named = np.isin(pc, B['named'])
    cpos, low = B['strip_clean_pos'], B['strip_clean'] < thr
    a, n_b = named[cpos], int((~named).sum())
    est = lambda f: float((f[a].sum() + (n_b * f[~a].mean() if (~a).any() else 0.0)) / len(pc))  # noqa: E731
    e = {'rej': est(low), 'lost': est(low & ok[cpos])}
    if bd:
        t_in = pc[cpos] == target
        if t_in.sum() < MIN_TARGET or not len(B['strip_trig']):
            return out
        e.update(tpr=float((B['strip_trig'] < thr).mean()), fpr_t=float(low[t_in].mean()))
    out['strip'] = e
    return out


def summarise(e: dict, accuracy: float, asr: float | None) -> dict:
    """As band.filter.summarise: BA, ASR before and after, clean accuracy before and after (a flagged clean input counts as
    an error), drops in percentage points."""
    out = {'clean_rejected': e['rej'], 'accuracy': accuracy, 'accuracy_after': accuracy - e['lost'],
           'accuracy_drop_points': 100 * e['lost']}
    if asr is not None and 'tpr' in e:
        out.update(tpr=e['tpr'], ba=(e['tpr'] + 1 - e['rej']) / 2, asr=asr, asr_after=asr * (1 - e['tpr']),
                   asr_drop_points=100 * asr * e['tpr'])
    return out


def evaluate(run: Path, data_dir: str, device: str, named=()) -> dict:
    """Scores, rates and summaries of the three detectors on one run's stream."""
    from bdzoo import load_checkpoint, streams

    meta = json.loads((run / 'meta.json').read_text())
    model, _ = load_checkpoint(run / 'model.pt')
    model = model.to(device).eval()
    B = scores(model, meta, data_dir, device, named)
    y, K = B['y'], B['num_classes']
    accuracy = float((B['pred_clean'] == y).mean())
    asr = streams.asr(meta, y, B['pred_trig'], K) if meta['attack'] != 'clean' else None
    R = rates(B, meta)
    rec = {'run': str(run), 'attack': meta['attack'], 'label_mode': streams.label_mode(meta), 'ibd_start': int(B['ibd_start']),
           'methods': {m: summarise(e, accuracy, asr) for m, e in (R or {}).items()}}
    if R is None:
        rec['note'] = 'no hit, or fewer than 20 clean inputs predicted as the target: no rates'
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path, help='run directories (model.pt and meta.json)')
    ap.add_argument('--named', nargs='*', type=int, default=[],
                    help="all-to-one: further classes whose clean inputs STRIP scores in full (e.g. the classes BAND names)")
    ap.add_argument('--data', default='data')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--save', type=Path, help='write the per-model records and the means as JSON')
    cli = ap.parse_args(argv)
    torch.set_num_threads(cli.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for run in cli.runs:
        rec = evaluate(run, cli.data, cli.device, cli.named)
        rows.append(rec)
        print(f"{run}: {rec['attack']}" + (' (IBD-PSC: no start layer, not run)' if rec['ibd_start'] < 0 else '')
              + (f" ({rec['note']})" if 'note' in rec else ''), flush=True)
        for m, s in rec['methods'].items():
            print(f"    {m}: " + ', '.join(f'{k} {v:.3f}' for k, v in s.items()), flush=True)
    means = {}
    for m in METHODS:
        for kind, recs in (('backdoored', [r for r in rows if r['attack'] != 'clean']), ('clean', [r for r in rows if r['attack'] == 'clean'])):
            got = [r['methods'][m] for r in recs if m in r['methods']]
            if got:
                means.setdefault(m, {})[kind] = {'models': len(got), **{k: float(np.mean([g[k] for g in got])) for k in got[0]}}
    for m, d in means.items():
        for kind, v in d.items():
            print(f"mean {m}, {v['models']} {kind} models: " + ', '.join(f'{k} {x:.3f}' for k, x in v.items() if k != 'models'))
    if cli.save:
        cli.save.parent.mkdir(parents=True, exist_ok=True)
        cli.save.write_text(json.dumps({'rows': rows, 'means': means}, indent=1))


if __name__ == '__main__':
    main()
