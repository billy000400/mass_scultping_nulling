"""Group 1: No finetuning (inference only).

Experiments:
- 1.a part_og: OG ParT weights, no projection
- 1.b rav: ParT + RAV projection with alpha
- 1.c cure: ParT + SVD forget/retain with alpha
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
DEFAULT_PART_MODEL = "/scope-vol/mass_perp_classifier/logs/no_bias/best.pt"

from ..core.data import load_h5_group, load_aux_from_h5, make_loader
from ..core.metrics import compute_accuracy_and_jsd
from ..core.model import PerpClassifier, load_pretrained_part_weights
from ..core.projections import load_rav_vectors, load_svd_basis, project_svd
from ..core.utils import set_seed, infer_num_classes

logger = logging.getLogger(__name__)

EXPERIMENTS = ("part_og", "rav", "cure")


def _write_scores(out_h5: str, group: str, key: str, scores: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)) or ".", exist_ok=True)
    with h5py.File(out_h5, "a") as f:
        gname = group.strip("/")
        dpath = f"/{gname}/{key}" if gname else f"/{key}"
        if gname and f"/{gname}" not in f:
            f.create_group(f"/{gname}")
        if dpath in f:
            del f[dpath]
        f.create_dataset(dpath, data=scores, compression="gzip", compression_opts=4)


@torch.no_grad()
def run_inference(
    model: PerpClassifier,
    loader,
    device: str,
    *,
    rav_tensors: Optional[torch.Tensor] = None,
    rav_mu: Optional[torch.Tensor] = None,
    rav_std: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    svd_forget_basis: Optional[torch.Tensor] = None,
    svd_forget_mean: Optional[torch.Tensor] = None,
    svd_retain_basis: Optional[torch.Tensor] = None,
    svd_retain_mean: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model.eval()
    outs = []
    for batch in tqdm(loader, desc="infer", dynamic_ncols=True):
        x = batch[0].to(device, non_blocking=True)

        # SVD projection (if CURE). alpha=0 skips projection (matches ParT).
        if svd_forget_basis is not None:
            x = project_svd(
                x, svd_forget_basis, svd_forget_mean, svd_retain_basis, svd_retain_mean,
                alpha=alpha,
            )

        if rav_tensors is not None:
            scores = model(x, rav_tensors, alpha=alpha, rav_mu=rav_mu, rav_std=rav_std)
        else:
            scores = model(x)

        if isinstance(scores, (tuple, list)):
            scores = scores[0]
        outs.append(scores.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(outs, axis=0)


def _load_svd_if_cure(
    svd_forget: Optional[str],
    svd_retain: Optional[str],
    svd_k: Optional[int],
    svd_variance: float,
    device: str,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if svd_forget is None:
        return None, None, None, None

    Vh_k, mean_k, _ = load_svd_basis(svd_forget, svd_k, svd_variance)
    forget_basis = torch.from_numpy(Vh_k).to(device)
    forget_mean = torch.from_numpy(mean_k).to(device)

    retain_basis = retain_mean = None
    if svd_retain:
        Vh_r, mean_r, _ = load_svd_basis(svd_retain, svd_k, svd_variance)
        retain_basis = torch.from_numpy(Vh_r).to(device)
        retain_mean = torch.from_numpy(mean_r).to(device)

    return forget_basis, forget_mean, retain_basis, retain_mean


def run(
    experiment: str,
    h5: str,
    out_h5: str,
    Part_model_path: str,
    *,
    infer_group: str = "",
    infer_x_key: str = "cls_tokens_ln",
    infer_y_key: Optional[str] = "label",
    jsd_variable: Optional[str] = "jet_sdmass",
    rav_h5: Optional[str] = None,
    rav_tensors_key: Optional[str] = None,
    alpha: float = 1.0,
    svd_forget: Optional[str] = None,
    svd_retain: Optional[str] = None,
    svd_k: Optional[int] = None,
    svd_variance: float = 0.95,
    batch_size: int = 1024,
    data_fraction: float = 1.0,
    seed: int = 1337,
    device: str = "cuda",
) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment}")

    set_seed(seed)
    rav_h5 = rav_h5 or h5

    # Load data (labels and jsd_aux for metrics)
    x, y = load_h5_group(
        h5, infer_group, infer_x_key, infer_y_key, data_fraction=data_fraction
    )
    in_dim = int(x.shape[1])
    n = len(x)
    loader = make_loader(x, None, batch_size=batch_size, shuffle=False)

    jsd_aux = None
    if jsd_variable:
        try:
            jsd_aux = load_aux_from_h5(h5, infer_group, jsd_variable, n)
        except KeyError:
            logger.info("JSD variable '%s' not found; JSD will be 0", jsd_variable)

    num_classes = infer_num_classes(y) if y is not None else None

    # Infer num_classes from ParT checkpoint
    state = torch.load(Part_model_path, map_location="cpu")
    sd = state.get("model_state", state)
    fc_w = sd["fc.weight"] if "fc.weight" in sd else None
    for k, v in sd.items():
        if "fc" in k and k.endswith(".weight") and v.ndim == 2:
            fc_w = v
            break
    if fc_w is None:
        raise ValueError(f"Could not find FC weights in {Part_model_path}")
    out_dim, ckpt_in_dim = fc_w.shape
    if ckpt_in_dim != in_dim:
        logger.warning("Input dim mismatch: data=%d, checkpoint=%d", in_dim, ckpt_in_dim)
        in_dim = ckpt_in_dim

    # Load pretrained weights
    pretrained = load_pretrained_part_weights(
        Part_model_path, in_dim=in_dim, num_classes=out_dim, device=device
    )
    model = PerpClassifier(
        in_dim=in_dim,
        num_classes=out_dim,
        for_inference=True,
        use_pretrained_norm_fc=True,
        pretrained_state_dict=pretrained,
        input_is_post_ln=True,
    )
    model.to(device)

    # Projection setup
    rav_tensors = None
    rav_mu = rav_std = None
    if experiment == "rav":
        if not rav_tensors_key:
            raise ValueError("--rav-tensors-key required for experiment=rav")
        arr, rav_mu, rav_std = load_rav_vectors(rav_h5, rav_tensors_key)
        rav_tensors = torch.from_numpy(arr.astype(np.float32)).to(device)
        if arr.ndim == 1:
            rav_tensors = rav_tensors.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        else:
            rav_tensors = rav_tensors.unsqueeze(0)  # (1, orders, D)
        if rav_mu is not None and rav_std is not None:
            rav_mu = torch.from_numpy(rav_mu).to(device)
            rav_std = torch.from_numpy(rav_std).to(device)

    svd_f, svd_fm, svd_r, svd_rm = None, None, None, None
    if experiment == "cure":
        if not svd_forget:
            raise ValueError("--svd-forget required for experiment=cure")
        svd_f, svd_fm, svd_r, svd_rm = _load_svd_if_cure(
            svd_forget, svd_retain, svd_k, svd_variance, device
        )

    scores = run_inference(
        model,
        loader,
        device,
        rav_tensors=rav_tensors,
        rav_mu=rav_mu,
        rav_std=rav_std,
        alpha=alpha,
        svd_forget_basis=svd_f,
        svd_forget_mean=svd_fm,
        svd_retain_basis=svd_r,
        svd_retain_mean=svd_rm,
    )
    _write_scores(out_h5, infer_group or "/", "scores", scores)

    # Compute accuracy and JSD (same logic as finetune for comparability)
    preds = np.argmax(scores, axis=1) if scores.shape[1] > 1 else np.zeros(len(scores), dtype=np.int64)
    has_labels = y is not None
    targets = y if has_labels else np.zeros(len(preds), dtype=np.int64)
    metrics = compute_accuracy_and_jsd(
        preds, targets, jsd_aux=jsd_aux, num_classes=num_classes, probs=scores
    )
    msg = f"Wrote scores to {out_h5} (experiment={experiment}, alpha={alpha:.2f})"
    if has_labels:
        msg += f" | acc={metrics['acc']:.4f}"
        if "balanced_acc" in metrics:
            msg += f" bal_acc={metrics['balanced_acc']:.4f}"
        if metrics.get("eff_at_1pct") is not None:
            msg += f" eff@1%={metrics['eff_at_1pct']:.4f}"
    msg += f" jsd={metrics['jsd']:.6f}"
    eff = metrics.get("eff_at_1pct")
    print(f"efficiency@1% = {eff:.4f}" if eff is not None else "efficiency@1% = N/A")
    logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser(description="No-finetune experiments (1.a, 1.b, 1.c)")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("--h5", default=DEFAULT_H5, help=f"Input H5 (default: {DEFAULT_H5})")
    p.add_argument("--out-h5", required=True)
    p.add_argument("--ParT-model-path", default=DEFAULT_PART_MODEL, dest="Part_model_path",
                   help=f"ParT checkpoint (default: {DEFAULT_PART_MODEL})")
    p.add_argument("--infer-group", default="")
    p.add_argument("--infer-x-key", default="cls_tokens_ln", help="Feature key (default: cls_tokens_ln)")
    p.add_argument("--infer-y-key", default="label", help="Label key for accuracy (default: label)")
    p.add_argument("--jsd-variable", default="jet_sdmass", help="Aux variable for JSD (default: jet_sdmass)")
    p.add_argument("--rav-h5", default=None,
                   help="H5 file containing RAV vectors (default: same as --h5)")
    p.add_argument("--rav-tensors-key", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--svd-forget", default=None)
    p.add_argument("--svd-retain", default=None)
    p.add_argument("--svd-k", type=int, default=None)
    p.add_argument("--svd-variance", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    run(
        args.experiment,
        args.h5,
        args.out_h5,
        args.Part_model_path,
        infer_group=args.infer_group,
        infer_x_key=args.infer_x_key,
        infer_y_key=args.infer_y_key,
        jsd_variable=args.jsd_variable,
        rav_h5=args.rav_h5,
        rav_tensors_key=args.rav_tensors_key,
        alpha=args.alpha,
        svd_forget=args.svd_forget,
        svd_retain=args.svd_retain,
        svd_k=args.svd_k,
        svd_variance=args.svd_variance,
        batch_size=args.batch_size,
        data_fraction=args.data_fraction,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
