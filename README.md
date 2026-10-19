# backdoor-baselines

Eight checkpoint-level backdoor detectors, each with its own published decision rule, as used for comparison in the paper
*BAND: Backdoor Detection through Batch Normalization Statistics*. The detector itself is in
[BAND](https://github.com/donikv/BAND) and the models in [backdoor-zoo](https://github.com/donikv/backdoor-zoo).

| Method | Reads | Decision rule (needs) |
|---|---|---|
| Fields et al. | classifier weights | Dixon's Q at α = 0.05 (nothing) |
| DFBScanner, configuration A | classifier weights | F1-optimal similarity threshold (labelled clean and poisoned models) |
| HTell | random probes at the head (LC-1: last block) | best probe variant, TPR at FPR < 5% (labelled models); LC-1: clean maximum |
| CLP | conv and BN weights | any channel with z > 3 (nothing) |
| ProP | noisy activations of random inputs | none specified (AUROC only) |
| BNP | 500 clean images, BN statistics | any unit with z > 3 (nothing) |
| MM-BD | optimised inputs | p ≤ 0.05 (nothing) |
| FreeEagle | optimised features | \|M − 1\| above the ROC point at 5% FPR (labelled models) |
| Row-norm ratio | classifier weights | clean maximum (clean models) |

## Install

```bash
pip install git+https://github.com/donikv/backdoor-zoo
pip install -e '.[rules]'                 # scikit-optimize and scikit-learn for the DFBScanner and FreeEagle rules
bash scripts/fetch_third_party.sh         # FreeEagle's code (MIT), pinned
```

## Use

```bash
python -m bdbaselines.score config_models/*/seed*/*/ --out config.json   # labelled configuration models
python -m bdbaselines.score test_models/*/seed*/*/ --out test.json
python -m bdbaselines.decide --config config.json --test test.json          # AUROC, balanced accuracy, F1
```

Scores are computed per checkpoint from a backdoor-zoo run directory (`model.pt` and `meta.json`). MM-BD and FreeEagle
take 10 to 160 seconds per model on a GPU. `python -m bdbaselines.validate dfb|clp|mmbd` compares these implementations
with the released DFBScanner, BackdoorBench CLP and MM-BD code. Fetch that code first with
`bash scripts/fetch_third_party.sh --validation`; it has no license, so it is not included here.

## License

MIT. Third-party code is fetched from its sources at pinned commits.
