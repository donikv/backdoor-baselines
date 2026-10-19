"""Apply each method's own decision rule (bdbaselines.rules) to scored test models.

    python -m bdbaselines.decide --config config_scores.json --test test_scores.json [--methods ...] [--out decisions.json]

--config holds the configuration models (labelled clean and backdoored; rows labelled 'failed' are ignored), scored by
bdbaselines.score; methods marked * ignore it, (clean) use its clean rows, ** its labelled rows. With labels in --test,
the table reports AUROC (ties count one half), balanced accuracy and F1 at the method's threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rules import METHODS, auroc, decide, fit


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows += [r for r in json.loads(Path(p).read_text())['rows'] if 'fatal' not in r and r.get('label') != 'failed']
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', nargs='+', type=Path, required=True)
    ap.add_argument('--test', nargs='+', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    ap.add_argument('--out', type=Path)
    cli = ap.parse_args(argv)
    cfg, test = load(cli.config), load(cli.test)
    out = {}
    print(f"{'method':10s} {'AUROC':>6s} {'BA':>6s} {'F1':>6s}  detected / flagged clean")
    for m in cli.methods:
        try:
            params = fit(m, cfg)
        except (ValueError, KeyError, ImportError) as exc:
            print(f'{m:10s} not fitted: {type(exc).__name__}: {exc}')
            continue
        dec = [(r, *decide(m, params, r)) for r in test]
        dec = [(r, s, f) for r, s, f in dec if s is not None]
        pos = [(s, f) for r, s, f in dec if r.get('label') == 'backdoored']
        neg = [(s, f) for r, s, f in dec if r.get('label') == 'clean']
        res = {'params': {k: v for k, v in params.items() if k != 'profile'},
               'models': [{'model': r['model'], 'label': r.get('label'), 'score': s, 'flagged': f} for r, s, f in dec],
               'auroc': auroc([s for s, _ in pos], [s for s, _ in neg])}
        if pos and neg and all(f is not None for _, f in pos + neg):
            tp, fp = sum(f for _, f in pos), sum(f for _, f in neg)
            res |= {'detected': f'{tp}/{len(pos)}', 'flagged_clean': f'{fp}/{len(neg)}',
                    'balanced_accuracy': (tp / len(pos) + 1 - fp / len(neg)) / 2, 'f1': 2 * tp / (2 * tp + fp + len(pos) - tp)}
        out[m] = res
        fmt = lambda v: '   n/a' if v is None else f'{v:6.3f}'  # noqa: E731
        print(f"{m:10s} {fmt(res['auroc'])} {fmt(res.get('balanced_accuracy'))} {fmt(res.get('f1'))}  "
              f"{res.get('detected', '-')} / {res.get('flagged_clean', '-')}")
    if cli.out:
        cli.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
