# Training & evaluation configuration

Reference document for the methods section. All values verified against the
codebase (`src/experiments/finetune.py`, `scripts/run_cure_disco_pareto_sweep.py`,
`scripts/run_disco_pareto_sweep.py`) and the on-disk sweep outputs in
`runs/`. Reproduce by running the commands listed at the bottom.

## Data

Source HDF5: `/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5`

- Latent: `cls_tokens_ln` — post-LayerNorm CLS token from a JetClass-pretrained
  ParticleTransformer (128-d).
- Concept variable: `jet_sdmass` (jet soft-drop mass).
- Selection: jets within the soft-drop-mass window 75–175 GeV, restricted to
  the binary set {QCD, Hbb}.

| class | label | total |
|---|---|---:|
| QCD | 0 | 187,587 |
| Hbb | 1 | 846,096 |
| **total** | | **1,033,683** |

Class imbalance Hbb : QCD ≈ 4.51 : 1.

### Train / val split

`stratified_split(y, frac=0.8, seed=1337)`:

| split | QCD | Hbb | total |
|---|---:|---:|---:|
| train (80 %) | 150,070 | 676,877 | 826,947 |
| val   (20 %) |  37,517 | 169,219 | 206,736 |

The val split is the one that drives early-stop and best-checkpoint selection
during finetuning.

### Important note on histogram statistics

The QCD soft-drop-mass histograms in
`results/mass_histograms.pdf` and the inference-only screen plots
(`runs/cure_screen/`, `results/pareto_frozen_*`) are computed on the
**full dataset** (all 187,587 QCD jets shown in the legend). They are
*not* val-set numbers. The full set is used because (i) the screen and
frozen-α plots involve no training, so train/val partitioning is
irrelevant, and (ii) larger statistics give cleaner per-bin densities for
JSD. The finetuned-Pareto sweeps (Phase 4) use the val split for
metric reporting.

### Imbalance handling

- `--balance-samples none` (default; no over- or under-sampling).
- `--class-weights balanced` — inverse-class-frequency weights applied to
  the cross-entropy loss only.

## Architecture (downstream classifier)

The "head" used in all finetuning experiments is the original JetClass
classifier head saved in `logs/no_bias/best.pt`:

```
LayerNorm(128) → Linear(128 → 10, bias=False) → softmax_10
```

i.e., a 10-class JetClass head, never re-architected for the binary task.
The binary tagger score is

```
TXbb = P(Hbb) / (P(Hbb) + P(QCD))     (HEP convention; bounded in [0,1])
```

extracted from the 10-class softmax. Eff@1 % is the percentile-based
QCD-mistag-rate-of-1 % working point (`compute_efficiency_at_misid_rate`
in `src/core/metrics.py`). Eff@1 % is invariant to any monotone
transformation of the score, so the change of TXbb convention from
`P(Hbb)/P(QCD)` to `P(Hbb)/(P(Hbb)+P(QCD))` does not affect any reported
efficiency.

## Optimiser & training schedule (Phase 4 finetune sweeps)

| parameter | value | source |
|---|---|---|
| optimiser | AdamW | `src/experiments/finetune.py:511` |
| learning rate | 1 × 10⁻² | `src/experiments/finetune.py:678` |
| weight decay | 0.0 | `src/experiments/finetune.py:679` |
| LR schedule | `ReduceLROnPlateau` on **train** loss, patience 5, min-lr 1 × 10⁻⁷ | `src/experiments/finetune.py:681–683` |
| batch size | 4096 | `scripts/run_cure_disco_pareto_sweep.py:177` |
| epochs (max) | 40 | `scripts/run_cure_disco_pareto_sweep.py:173` |
| early-stop patience | 12 (on val loss = CE + λ·DisCo) | `scripts/run_cure_disco_pareto_sweep.py:174` |
| class weights | `balanced` (inverse frequency on CE) | `src/experiments/finetune.py:506–510` |
| precision | fp32 | – |
| seed | 1337 | `src/experiments/finetune.py:691` |

Reported metrics in `summary.csv` come from the **best-val-loss epoch**
checkpoint (the early-stop pick), not the final epoch.

## DisCo loss

DisCo regularises the (CE-trained) tagger logit against `jet_sdmass`,
applied on the **QCD** subset of each batch (`disco_class_label=0`).

```
L_total  =  CE(class_weighted)  +  λ · dCorr²(logit, jet_sdmass | class=QCD)
```

### λ grids actually run (read from `runs/*/summary.csv`)

| sweep                               |  #λ |  λ values                                                                   |
|---|---|---|
| DisCo-only (`disco_pareto_pretrained`)  |  20 | {0, 0.1, 0.5, 1, 2, 5, 5.25, 5.5, 5.75, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000} |
| S1+DisCo k = 1 (ep40)                   |  11 | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}                                              |
| S1+DisCo k = 2 (ep40)                   |  16 | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50}                          |
| S1+DisCo k = 3 (ep40)                   |  13 | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15}                                      |

## Forget-basis construction (S1)

S1 = top-k right singular vectors of M = (B, D) matrix of QCD
class-conditional bin-mean post-LN CLS tokens, centred by the overall QCD
mean. B = 20 quantile bins of `jet_sdmass`. Construction:
`src/data_processing/build_mass_forget_bases.py::build_s1_qcd_massshift`.
Stored at `svd_results/QCD_massshift_cls_tokens_ln_svd.npz` (kmax = 20).

Projection at strength α (centred form, in `src/core/projections.py`):

```
h'  =  h  −  α · ((h − μ_QCD) Vh^T) Vh
```

with V_h orthonormal rows, μ_QCD the overall QCD mean of the latent.

## Hardware & wall-clock

- GPU: 1× **NVIDIA GeForce GTX 1080 Ti**.
- Per-epoch wall-clock at batch=4096 on the train split (826,947 samples):
  ≈ 45–48 s. Measured from successive epoch-checkpoint mtimes in
  `runs/cure_disco_pareto__S1_massshift_ep40_k2/lambda_5.0/`.
- Per-λ run: at most 40 × 47 s ≈ 30 min; with the early-stop patience of
  12 on val loss, typical run ends at epoch 25–35 (≈ 18–25 min).
- Phase 4 grand total (DisCo-only 20-λ sweep + S1+DisCo k = 1/2/3 sweeps,
  60 runs total): **≈ 25 GPU-hours** end-to-end.

## Reproduction commands

From the project root (`/scope-vol/mass-sculpting-nulling/`):

```bash
# 1. Build the S1 forget basis (and S2/S3 for comparison).
python -m src.data_processing.build_mass_forget_bases

# 2. Inference-only screen (no training): scans (basis, k) at α=1.
python scripts/run_cure_screen.py

# 3. DisCo-only finetune Pareto (the published baseline).
python scripts/run_disco_pareto_sweep.py \
    --save-dir runs/disco_pareto_pretrained --epochs 40 --batch-size 4096

# 4. S1 + DisCo finetune Pareto for k ∈ {1, 2, 3} (the proposed sweep).
python scripts/run_S1_massshift_paretos.py --epochs 40 --batch-size 4096 \
    --early-stop-patience 12

# 5. Paper figures.
python scripts/plot_scree_M.py
python scripts/probe_r2_vs_k.py
python scripts/k_ablation_S1_no_finetune.py
python scripts/pareto_frozen_alpha_sweep.py
python scripts/pareto_frozen_methods_alpha_sweep.py
python scripts/plot_pareto_overlay.py        # Phase 4 finetuned overlay
```
