# backdoor-baselines

The checkpoint-level backdoor detectors used for comparison in the paper *BAND: Backdoor Detection through Batch
Normalization Statistics*, each with its own published decision rule (Appendix H.4 of the paper). The detector itself is in
[BAND](https://github.com/donikv/BAND) and the models in [backdoor-zoo](https://github.com/donikv/backdoor-zoo).

| Method | Reads | Decision rule (needs) |
|---|---|---|
| Fields et al. | classifier weights | Dixon's Q above its critical value at α = 0.05 (nothing) |
| CLP | conv and BN weights | any channel with z > 3 (nothing) |
| BNP | 500 clean images, BN statistics | any unit with z > 3 (nothing; the head BN of MLP-head models) |
| MM-BD | optimised inputs | p ≤ 0.05 (nothing) |
| Row-norm ratio | classifier weights | above the clean-reference maximum (clean models; MLP head) |
| DFBScanner, configuration A | classifier weights | cosine similarity to the clean profile below the F1-maximising threshold (labelled clean and poisoned models) |
| HTell | random probes at the head | the probe variant with the best AUROC on the labelled models, above its clean-reference maximum (labelled models; MLP head) |
| HTell, LC-1 | random probes at the last feature block | above the clean-reference maximum (clean models; the paper's HTell on BackdoorBench) |
| FreeEagle | optimised features | \|M − 1\| above the ROC point closest to 5% FPR (labelled models) |

"Labelled models" are the clean reference models plus labelled poisoned configuration models (in the paper: 15 clean
references and ten models backdoored with patch or blend at 10%). No rule looks at the tested models.

Supported checkpoints: backdoor-zoo's PreAct-ResNet-18 and VGG-16-BN, each with the MLP head (GAP → Linear → BN1d → ReLU →
Linear) or fully convolutional (GAP → Linear), at 32, 64, 112 and 224 px (CIFAR-10, CIFAR-100, GTSRB, TinyImageNet,
ImageNet-30 at 224 px for the ResNets and 112 px for VGG-16), and BackdoorBench's PreActResNet18. HTell at the head and the
row-norm ratio need the MLP head. BNP uses the head BN of MLP-head models; on other models it falls back to the last
BatchNorm2d, as in the paper's BackdoorBench row (the paper reports no BNP for its fully convolutional models). FreeEagle
splits PreAct-ResNets at the input of the second stage and VGG-16 after its third max-pool, the defaults of its code.

## Install

```bash
pip install git+https://github.com/donikv/backdoor-zoo
pip install -e '.[rules]'                 # scikit-optimize and scikit-learn for the DFBScanner and FreeEagle rules
bash scripts/fetch_third_party.sh         # FreeEagle's code (MIT), pinned
```

## Use

```bash
# references (clean) and configuration models (backdoored), then the tested models
python -m bdbaselines.score refs/*/seed*/*/ config/*/seed*/*/ --out config.json --data data
python -m bdbaselines.score test/*/seed*/*/ --out test.json --data data
python -m bdbaselines.decide --config config.json --test test.json          # AUROC, balanced accuracy, F1 per method
```

`score` reads each backdoor-zoo run directory (`model.pt` and `meta.json`): the dataset in `meta.json` sets the image size,
the class count and BNP's clean images (the first 500 of a seed-0 permutation of the test split). Its label is `clean`,
`backdoored` for a usable backdoor (the paper's criterion: clean accuracy at most 0.05 below the dataset's bar and attack
success at the attack's bar) or `failed`; `decide` ignores failed models. `--methods` selects detectors (`fields dfb htell
htell_lc1 clp bnp mmbd freeeagle rownorm`). MM-BD and FreeEagle take seconds to minutes per model on a GPU, longest for
ImageNet-30 and the 200 classes of TinyImageNet.

`python -m bdbaselines.validate dfb|clp|mmbd` compares these implementations with the released DFBScanner, BackdoorBench
CLP and MM-BD code. Fetch that code first with `bash scripts/fetch_third_party.sh --validation`; it has no license, so it is
not included here.

## Input-level detectors

The detectors of triggered test inputs compared with BAND's test-time filter (Table 1b and Appendix L of the paper), each at
its published setting:

| Method | Needs | Flags an input when |
|---|---|---|
| STRIP | clean images (overlays and threshold) | the entropy over 100 overlays is below the 1% point of a normal fitted on clean inputs |
| IBD-PSC | labelled clean inputs (start layer) | the prediction consistency under amplified BN layers is at least 0.9; no start layer, no result |
| SCALE-UP (data-free) | nothing | at least half of the copies scaled by 3, 5, 7, 9 and 11 keep the prediction |

```bash
python -m bdbaselines.inputs models/cifar10_preact_resnet18_mlphead/seed7000/blend --save inputs.json
```

They are evaluated on the run's stream from `bdzoo.streams` (the same images and triggered copies as BAND's filter), with
TPR on the triggered inputs the model assigns to the target, clean rejections, and the FPR among clean inputs predicted as the
target; all-to-all models use the per-class stream.

## License

MIT. Third-party code is fetched from its sources at pinned commits.
