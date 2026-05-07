# Mass Perpendicular Classifier Experiments

## Default Input (Hbb vs QCD)

Default H5: `/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5`

- Features: `cls_tokens_ln` (128-dim post-LN hidden states)
- Labels: `label`
- JSD variable: `jet_sdmass`
- DisCo variable: `jet_sdmass`

Both no_finetune and finetune compute **accuracy** and **mean JSD** using the same logic in `core/metrics.py` for comparability.

## Codebase Structure

```
src/
├── core/                 # Shared utilities
│   ├── utils.py          # set_seed, infer_num_classes
│   ├── data.py           # H5 loading, loaders, stratified_split, balance_samples
│   ├── model.py          # PerpClassifier, load_pretrained_part_weights
│   ├── projections.py    # RAV projection, SVD (CURE) projection
│   └── losses.py         # distance_correlation, make_loss_and_metrics
├── experiments/
│   ├── no_finetune.py    # 1.a, 1.b, 1.c (inference only)
│   └── finetune.py       # 2.a, 2.b, 2.c, 2.d (training)
├── run_no_finetune.py    # CLI for no-finetune
├── run_finetune.py       # CLI for finetune
└── train_on_h5.py        # Legacy script (full CLI, backward compatible)
```

## Experiment Groups

### Group 1: No finetuning (inference only)

| ID  | Name    | Description                                      |
|-----|---------|--------------------------------------------------|
| 1.a | part_og | OG ParT weights, no projection                   |
| 1.b | rav     | RAV projection with alpha scaling               |
| 1.c | cure    | CURE style (SVD forget + retain) with alpha     |

**Run:** `python -m src.run_no_finetune --experiment {part_og|rav|cure} --h5 DATA.h5 --out-h5 SCORES.h5 [--ParT-model-path PART.pt]`

- For **rav**: add `--rav-tensors-key KEY --alpha 1.0`
- For **cure**: add `--svd-forget forget.npz [--svd-retain retain.npz]`

### Group 2: Finetuning

| ID  | Name       | Description                                      |
|-----|------------|--------------------------------------------------|
| 2.a | disco      | DisCo loss, train last linear layer from scratch |
| 2.b | part_og    | Finetune OG ParT's linear layer                   |
| 2.c | rav        | Finetune linear layer after RAV projection       |
| 2.d | cure       | Finetune linear layer after CURE projection      |
| 2.e | cure_disco | CURE projection + DisCo loss finetuning          |

**Run:** `python -m src.run_finetune --experiment {disco|part_og|rav|cure|cure_disco} --h5 DATA.h5 --save-dir runs/exp`

- **disco**: requires `mass` (or `--disco-variable`) in H5
- **part_og**, **rav**, **cure**, **cure_disco**: require `--ParT-model-path` (default: logs/no_bias/best.pt)
- **rav**: requires `--rav-tensors-key`
- **cure**, **cure_disco**: require `--svd-forget`

## Commands for All Experiments

Default input H5: `/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5`

**Working directory:** Run from the project root (`mass_perp_classifier/`), not from `src/` or `src/experiments/`:

```bash
cd /scope-vol/mass_perp_classifier
```

### Group 1: No finetuning (inference only)

```bash
# 1.a part_og — OG ParT weights, no projection
python -m src.run_no_finetune --experiment part_og --out-h5 out/1a_part_og_scores.h5

# 1.b rav — RAV projection with alpha (RAV from same H5 or --rav-h5)
python -m src.run_no_finetune --experiment rav --out-h5 out/1b_rav_scores.h5 \
  --rav-tensors-key RAV_jet_sdmass --alpha 1.0
# Or from a separate RAV file:
python -m src.run_no_finetune --experiment rav --out-h5 out/1b_rav_scores.h5 \
  --rav-h5 /scope-vol/ravs/sdmass_order1_cls_tokens_ln_HbbvsQCD.h5 --rav-tensors-key RAV_jet_sdmass_cls_tokens_ln_QCD_order1 --alpha 1.0

# 1.c cure — CURE style SVD forget + retain (requires SVD .npz files)
python -m src.run_no_finetune --experiment cure --out-h5 out/1c_cure_scores.h5 \
  --svd-forget /scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz --svd-retain /scope-vol/svd_results/QCD_nonres_cls_tokens_ln_svd.npz
# Add --alpha 0 to skip projection (matches ParT); --alpha 1 for full projection
```

### Group 2: Finetuning

```bash
# 2.a disco — DisCo loss, train linear layer from scratch
python -m src.run_finetune --experiment disco --save-dir runs/2a_disco --epochs 20

# 2.b part_og — Finetune OG ParT linear layer
python -m src.run_finetune --experiment part_og --save-dir runs/2b_part_og --epochs 20

# 2.c rav — Finetune after RAV projection
python -m src.run_finetune --experiment rav --save-dir runs/2c_rav --epochs 20 \
  --rav-tensors-key RAV_jet_sdmass --alpha 1.0
# Or from a separate RAV file:
python -m src.run_finetune --experiment rav --save-dir runs/2c_rav --epochs 20 \
  --rav-h5 /path/to/rav_vectors.h5 --rav-tensors-key RAV_jet_sdmass --alpha 1.0

# 2.d cure — Finetune after CURE projection
python -m src.run_finetune --experiment cure --save-dir runs/2d_cure --epochs 20 \
  --svd-forget /scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz --svd-retain /path/to/retain.npz

# 2.e cure_disco — CURE projection + DisCo loss
python -m src.run_finetune --experiment cure_disco --save-dir runs/2e_cure_disco --epochs 20 \
  --svd-forget /scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz --disco-lambda 5
```

**Note:** `--h5` defaults to the Hbb vs QCD file above. Override with `--h5 /path/to/other.h5` if needed.

**DisCo Pareto sweep:** Run `scripts/run_disco_pareto_sweep.py` to sweep `--disco-lambda` and map the eff@1%bkg vs JSD Pareto boundary. Saves `best_metrics.json` per run and `summary.csv` with aggregated results. Use `--plot` to generate a scatter plot from existing runs.

**CURE+DisCo Pareto sweep:** Run `scripts/run_cure_disco_pareto_sweep.py` to sweep `--disco-lambda` with CURE SVD projection applied first. Combines projection-based and loss-based decorrelation. Use `--svd-forget` to override the default SVD path.

**RAV from separate file:** Use `--rav-h5 /path/to/rav_file.h5` to load RAV vectors from a different H5 than the data. If omitted, RAV is read from the same file as `--h5`.

If you see `ModuleNotFoundError: No module named 'src'`, ensure you are in `mass_perp_classifier/` (project root), not in `src/` or `src/experiments/`.

## Legacy Script

`train_on_h5.py` retains the full original CLI for backward compatibility. It now applies SVD projection during inference when `--svd-forget` is used (bug fix).
