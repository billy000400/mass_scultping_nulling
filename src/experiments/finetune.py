"""Group 2: Finetuning experiments.

Experiments:
- 2.a disco: DisCo loss, train last linear layer from scratch (no pretrained head)
- 2.b part_og: Finetune OG ParT's linear layer
- 2.c rav: Finetune linear layer after RAV projection
- 2.d cure: Finetune linear layer after CURE style forget and retain
"""

import argparse
import json
import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
DEFAULT_PART_MODEL = "/scope-vol/mass_perp_classifier/logs/no_bias/best.pt"

from ..core.data import (
    load_h5_group,
    load_aux_from_h5,
    make_loader,
    stratified_split,
    balance_samples,
    has_split,
)
from ..core.metrics import _jsd_from_hists, compute_efficiency_at_misid_rate
from ..core.model import PerpClassifier, load_pretrained_part_weights
from ..core.projections import (
    load_rav_vectors,
    load_svd_basis,
    project_svd,
)
from ..core.losses import (
    distance_correlation,
    make_loss_and_metrics,
    compute_class_weights,
)
from ..core.utils import set_seed, infer_num_classes

logger = logging.getLogger(__name__)

EXPERIMENTS = ("disco", "part_og", "rav", "cure", "cure_disco")


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
def _run_finetune_inference(
    model, best_path, x_tr, y_tr, x_va, y_va,
    batch_size, device, rav_tensors, rav_mu, rav_std, alpha,
    svd_f, svd_fm, svd_r, svd_rm,
    train_group, val_group, out_h5,
) -> None:
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    model.for_inference = True
    tr_loader = make_loader(x_tr, y_tr, batch_size=batch_size, shuffle=False)
    va_loader = make_loader(x_va, y_va, batch_size=batch_size, shuffle=False)
    for loader, group in [(tr_loader, train_group), (va_loader, val_group)]:
        outs = []
        for batch in loader:
            x = batch[0].to(device)
            if svd_f is not None:
                x = project_svd(x, svd_f, svd_fm, svd_r, svd_rm, alpha=alpha)
            scores = model(x, rav_tensors, alpha=alpha, rav_mu=rav_mu, rav_std=rav_std) if rav_tensors is not None else model(x)
            if isinstance(scores, (tuple, list)):
                scores = scores[0]
            outs.append(scores.cpu().numpy())
        scores = np.concatenate(outs, axis=0)
        _write_scores(out_h5, group, "scores", scores)
    logger.info("Wrote scores to %s", out_h5)


def _load_svd(
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


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    *,
    train: bool,
    epoch: int,
    phase: str,
    alpha: float,
    rav_tensors: Optional[torch.Tensor],
    rav_mu: Optional[torch.Tensor],
    rav_std: Optional[torch.Tensor],
    svd_f,
    svd_fm,
    svd_r,
    svd_rm,
    disco_enabled: bool,
    disco_lambda: float,
    disco_bkg_label: int,
    mass_bins: Optional[np.ndarray],
    num_classes: Optional[int],
):
    model.train(mode=train)
    running_loss = running_correct = running_disco = n_seen = 0.0
    all_preds, all_targets, all_probs = [], [], []
    jsd_hists = {}
    n_bins = len(mass_bins) - 1 if mass_bins is not None else 0
    jsd_val = 0.0

    pbar = tqdm(loader, desc=f"{phase} e{epoch:02d}", dynamic_ncols=True)
    for batch in pbar:
        x = batch[0].to(device)
        y = batch[1].to(device)
        disco_aux = batch[2].to(device) if len(batch) > 2 and disco_enabled else None
        jsd_aux = batch[3].to(device) if len(batch) > 3 and mass_bins is not None else None
        if y.dim() > 1:
            y = y.argmax(dim=1).long()

        # SVD projection. alpha=0 skips projection (matches ParT).
        if svd_f is not None:
            x = project_svd(x, svd_f, svd_fm, svd_r, svd_rm, alpha=alpha)

        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x, rav_tensors, alpha=alpha, rav_mu=rav_mu, rav_std=rav_std) if rav_tensors is not None else model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss = criterion(logits, y)

        disco_val = 0.0
        if disco_enabled and disco_aux is not None:
            probs = F.softmax(logits, dim=1)
            # Default QCD-only (label 0); use disco_bkg_label < 0 for full-batch DisCo (legacy).
            if disco_bkg_label < 0:
                disco_term = distance_correlation(probs, disco_aux)
            else:
                mask = y == disco_bkg_label
                disco_term = (
                    distance_correlation(probs[mask], disco_aux[mask])
                    if mask.sum() > 2
                    else torch.tensor(0.0, device=device)
                )
            disco_val = disco_term.item()
            # Same total loss on train and val (CE + λ·dCorr); backward only when train.
            loss = loss + disco_lambda * disco_term

        if train:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_disco += disco_val * bs
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        n_seen += bs
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        probs_batch = F.softmax(logits, dim=1) if logits.shape[1] > 1 else logits
        all_probs.append(probs_batch.detach().cpu())

        if mass_bins is not None and jsd_aux is not None:
            aux_np = jsd_aux.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            t_qcd = y_np == 0
            for c in np.unique(preds_np):
                c = int(c)
                m = t_qcd & (preds_np == c)
                h, _ = np.histogram(aux_np[m], bins=mass_bins)
                jsd_hists.setdefault(c, np.zeros(n_bins, dtype=np.float64))
                jsd_hists[c] += h
            classes_with_counts = [c for c, h in jsd_hists.items() if h.sum() > 0]
            if len(classes_with_counts) >= 2:
                h_ref = sum(jsd_hists[c] for c in classes_with_counts)
                if h_ref.sum() > 0:
                    jsd_val = float(
                        np.mean(
                            [
                                _jsd_from_hists(jsd_hists[c], h_ref)
                                for c in classes_with_counts
                            ]
                        )
                    )
                else:
                    jsd_val = 0.0
            else:
                jsd_val = 0.0

        pbar.set_postfix(
            loss=f"{running_loss / n_seen:.4f}",
            acc=f"{running_correct / n_seen:.4f}",
            disco=f"{running_disco / n_seen:.6f}" if disco_enabled else "",
        )

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs).numpy()
    stats = {"loss": running_loss / n_seen, "acc": running_correct / n_seen}
    if disco_enabled:
        stats["disco_loss"] = running_disco / n_seen
    if mass_bins is not None:
        stats["jsd"] = jsd_val
    if num_classes and num_classes > 1:
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        for c in range(num_classes):
            mask = all_targets == c
            if mask.any():
                class_total[c] = mask.sum().item()
                class_correct[c] = ((all_preds == all_targets) & mask).sum().item()
        valid = class_total > 0
        if valid.any():
            stats["balanced_acc"] = (
                class_correct[valid] / class_total[valid].clamp(min=1)
            ).mean().item()
    all_targets_np = all_targets.numpy()
    eff = compute_efficiency_at_misid_rate(
        all_probs, all_targets_np, qcd_label=0, hbb_label=1
    )
    if eff is not None:
        stats["eff_at_1pct"] = eff
    return stats, all_probs, all_targets_np


def _save_roc_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    save_dir: str,
    *,
    epoch: int,
    qcd_label: int = 0,
    hbb_label: int = 1,
    title_suffix: str = "",
    filename: str = "roc_curve.png",
) -> Optional[str]:
    """Save a ROC curve PNG (Hbb vs QCD) for the given val predictions.

    Uses ``probs[:, hbb_label]`` as the signal score. AUC is computed with the
    trapezoidal rule so we do not add a hard dependency on sklearn.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping ROC curve plot")
        return None

    if probs.ndim != 2 or probs.shape[1] <= max(qcd_label, hbb_label):
        logger.warning("ROC plot skipped: probs shape %s incompatible with labels", probs.shape)
        return None

    mask = (targets == qcd_label) | (targets == hbb_label)
    y = targets[mask]
    scores = probs[mask, hbb_label]
    n_pos = int((y == hbb_label).sum())
    n_neg = int((y == qcd_label).sum())
    if n_pos < 1 or n_neg < 1:
        logger.warning("ROC plot skipped: need both QCD and Hbb samples (n_qcd=%d, n_hbb=%d)", n_neg, n_pos)
        return None

    order = np.argsort(scores)[::-1]
    y_sorted = (y[order] == hbb_label).astype(np.float64)
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0.0], tp_cum / n_pos])
    fpr = np.concatenate([[0.0], fp_cum / n_neg])
    auc = float(np.trapz(tpr, fpr))

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    label = f"AUC={auc:.4f} (epoch {epoch})"
    ax.plot(tpr, fpr, linewidth=2, label=label)
    ax.set_xlabel("True Positive Rate (Hbb efficiency)")
    ax.set_ylabel("False Positive Rate (QCD → Hbb)")
    title = "ROC Curve: Hbb vs QCD"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([max(1e-4, 1.0 / max(n_neg, 1)), 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    out = os.path.join(save_dir, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve to %s (AUC=%.4f)", out, auc)
    return out


def run(
    experiment: str,
    h5: str,
    save_dir: str,
    *,
    Part_model_path: Optional[str] = None,
    rav_h5: Optional[str] = None,
    rav_tensors_key: Optional[str] = None,
    alpha: float = 1.0,
    svd_forget: Optional[str] = None,
    svd_retain: Optional[str] = None,
    svd_k: Optional[int] = None,
    svd_variance: float = 0.95,
    train_group: str = "/train",
    val_group: str = "/val",
    x_key: str = "cls_tokens_ln",
    y_key: str = "label",
    train_frac: float = 0.8,
    data_fraction: float = 1.0,
    disco_lambda: float = 10.0,
    disco_variable: str = "jet_sdmass",
    disco_bkg_label: int = 0,
    jsd_variable: Optional[str] = "jet_sdmass",
    class_weights: str = "balanced",
    balance_samples_method: str = "none",
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    lr_sched: str = "plateau-train",
    plateau_patience: int = 5,
    plateau_factor: float = 0.1,
    plateau_min_lr: float = 1e-7,
    early_stop_patience: Optional[int] = None,
    seed: int = 1337,
    device: str = "cuda",
    out_h5: Optional[str] = None,
    pretrained_head: bool = False,
) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")

    set_seed(seed)
    rav_h5 = rav_h5 or h5
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    split = has_split(h5, train_group, val_group, x_key)
    if split:
        x_tr, y_tr = load_h5_group(h5, train_group, x_key, y_key, data_fraction=data_fraction)
        x_va, y_va = load_h5_group(h5, val_group, x_key, y_key, data_fraction=data_fraction)
    else:
        x_all, y_all = load_h5_group(h5, "", x_key, y_key, data_fraction=data_fraction)
        N = len(x_all)
        idx = np.arange(N)
        if y_all is not None:
            train_idx, val_idx = stratified_split(y_all, train_frac, seed)
        else:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            k = int(round(N * train_frac))
            train_idx, val_idx = idx[:k], idx[k:]
        x_tr = x_all[train_idx]
        y_tr = y_all[train_idx] if y_all is not None else None
        x_va = x_all[val_idx]
        y_va = y_all[val_idx] if y_all is not None else None

    in_dim = int(x_tr.shape[1])
    num_classes = infer_num_classes(y_tr) if y_tr is not None else None
    out_dim = num_classes if num_classes else 1

    disco_aux_tr = disco_aux_va = None
    if experiment in ("disco", "cure_disco"):
        try:
            if split:
                disco_aux_tr = load_aux_from_h5(h5, train_group, disco_variable, len(x_tr))
                disco_aux_va = load_aux_from_h5(h5, val_group, disco_variable, len(x_va))
            else:
                disco_all = load_aux_from_h5(h5, "", disco_variable, len(x_all))
                disco_aux_tr = disco_all[train_idx]
                disco_aux_va = disco_all[val_idx]
        except KeyError:
            raise ValueError(f"DisCo requires '{disco_variable}' in H5")

    jsd_aux_tr = jsd_aux_va = None
    if jsd_variable:
        try:
            if split:
                jsd_aux_tr = load_aux_from_h5(h5, train_group, jsd_variable, len(x_tr))
                jsd_aux_va = load_aux_from_h5(h5, val_group, jsd_variable, len(x_va))
            else:
                jsd_all = load_aux_from_h5(h5, "", jsd_variable, len(x_all))
                jsd_aux_tr = jsd_all[train_idx]
                jsd_aux_va = jsd_all[val_idx]
        except KeyError:
            pass

    if balance_samples_method != "none" and y_tr is not None:
        x_tr, y_tr, bal_idx = balance_samples(x_tr, y_tr, method=balance_samples_method)
        if disco_aux_tr is not None:
            disco_aux_tr = disco_aux_tr[bal_idx]
        if jsd_aux_tr is not None:
            jsd_aux_tr = jsd_aux_tr[bal_idx]

    # Shuffle val samples once (deterministically) so batches are not class-homogeneous.
    # A class-homogeneous val batch makes CrossEntropyLoss(weight=cw, reduction="mean")
    # collapse the class weighting inside the batch, which yields a population-weighted
    # (rather than class-balanced) reported val loss and makes it look artificially
    # lower than train. Train/val membership is unchanged; only the visit order of the
    # val rows is permuted. The same permutation is applied to all aligned arrays.
    _val_perm = np.random.RandomState(seed + 1).permutation(len(x_va))
    x_va = x_va[_val_perm]
    if y_va is not None:
        y_va = y_va[_val_perm]
    if disco_aux_va is not None:
        disco_aux_va = disco_aux_va[_val_perm]
    if jsd_aux_va is not None:
        jsd_aux_va = jsd_aux_va[_val_perm]

    train_loader = make_loader(
        x_tr, y_tr,
        batch_size=batch_size,
        shuffle=True,
        disco_aux_np=disco_aux_tr,
        jsd_aux_np=jsd_aux_tr,
    )
    val_loader = make_loader(
        x_va, y_va,
        batch_size=batch_size,
        shuffle=False,
        disco_aux_np=disco_aux_va,
        jsd_aux_np=jsd_aux_va,
    )

    # Model
    # pretrained_head is an explicit opt-in (e.g. for `disco` sweeps that want to
    # start from the pretrained ParT head instead of random init).
    use_pretrained = (
        (experiment in ("part_og", "rav", "cure", "cure_disco") or pretrained_head)
        and Part_model_path
    )
    pretrained_state = None
    if use_pretrained:
        pretrained_state = load_pretrained_part_weights(
            Part_model_path, in_dim=in_dim, num_classes=out_dim, device=device
        )

    model = PerpClassifier(
        in_dim=in_dim,
        num_classes=out_dim,
        for_inference=False,
        use_pretrained_norm_fc=use_pretrained,
        pretrained_state_dict=pretrained_state,
        input_is_post_ln=True,
    )
    model.to(device)

    # Projections
    rav_tensors = rav_mu = rav_std = None
    if experiment == "rav" and rav_tensors_key:
        arr, rav_mu, rav_std = load_rav_vectors(rav_h5, rav_tensors_key)
        rav_tensors = torch.from_numpy(arr.astype(np.float32)).to(device)
        rav_tensors = rav_tensors.unsqueeze(0) if arr.ndim == 1 else rav_tensors.unsqueeze(0)
        if rav_mu is not None and rav_std is not None:
            rav_mu = torch.from_numpy(rav_mu).to(device)
            rav_std = torch.from_numpy(rav_std).to(device)

    svd_f, svd_fm, svd_r, svd_rm = _load_svd(
        svd_forget if experiment in ("cure", "cure_disco") else None,
        svd_retain if experiment in ("cure", "cure_disco") else None,
        svd_k, svd_variance, device,
    )

    # Training
    cw = (
        compute_class_weights(y_tr, "balanced").to(device)
        if class_weights == "balanced" and y_tr is not None
        else None
    )
    criterion, _ = make_loss_and_metrics("ce", num_classes, class_weights=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=plateau_min_lr,
        )
        if lr_sched != "none"
        else None
    )

    mass_bins = None
    if jsd_aux_tr is not None:
        lo = float(min(np.min(jsd_aux_tr), np.min(jsd_aux_va)) if jsd_aux_va is not None else np.min(jsd_aux_tr))
        hi = float(max(np.max(jsd_aux_tr), np.max(jsd_aux_va)) if jsd_aux_va is not None else np.max(jsd_aux_tr))
        mass_bins = np.linspace(lo, hi, 51)

    # Best checkpoint and early stopping both use val_stats["loss"] (CE + λ·DisCo when disco enabled).
    best_val_loss = float("inf")
    epochs_since_val_loss_improvement = 0
    best_path = os.path.join(save_dir, "best.pt")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

    for epoch in range(1, epochs + 1):
        train_stats, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, epoch=epoch, phase="train", alpha=alpha,
            rav_tensors=rav_tensors, rav_mu=rav_mu, rav_std=rav_std,
            svd_f=svd_f, svd_fm=svd_fm, svd_r=svd_r, svd_rm=svd_rm,
            disco_enabled=(experiment in ("disco", "cure_disco")),
            disco_lambda=disco_lambda,
            disco_bkg_label=disco_bkg_label,
            mass_bins=mass_bins,
            num_classes=num_classes,
        )
        val_stats, val_probs, val_targets = run_epoch(
            model, val_loader, optimizer, criterion, device,
            train=False, epoch=epoch, phase="val", alpha=alpha,
            rav_tensors=rav_tensors, rav_mu=rav_mu, rav_std=rav_std,
            svd_f=svd_f, svd_fm=svd_fm, svd_r=svd_r, svd_rm=svd_rm,
            disco_enabled=(experiment in ("disco", "cure_disco")),
            disco_lambda=disco_lambda,
            disco_bkg_label=disco_bkg_label,
            mass_bins=mass_bins,
            num_classes=num_classes,
        )

        # Best checkpoint: lowest validation loss (= CE + λ·DisCo when disco enabled; same as train total).
        is_best = val_stats["loss"] < best_val_loss

        if scheduler:
            scheduler.step(
                train_stats["loss"] if lr_sched == "plateau-train" else val_stats["loss"]
            )

        log_msg = "e%02d train: loss %.4f acc %.4f | val: loss %.4f acc %.4f"
        log_args = [epoch, train_stats["loss"], train_stats["acc"],
                    val_stats["loss"], val_stats["acc"]]
        if val_stats.get("eff_at_1pct") is not None:
            log_msg += " eff@1%%=%.4f"
            log_args.append(val_stats["eff_at_1pct"])
            print(f"efficiency@1% = {val_stats['eff_at_1pct']:.4f}")
        logger.info(log_msg, *log_args)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_stats": val_stats,
                "config": {"experiment": experiment},
            },
            os.path.join(save_dir, f"epoch_{epoch}.pt"),
        )
        if is_best:
            best_val_loss = val_stats["loss"]
            epochs_since_val_loss_improvement = 0
            torch.save(model.state_dict(), best_path)
            best_metrics = {
                "epoch": epoch,
                "acc": float(val_stats["acc"]),
                "loss": float(val_stats["loss"]),
            }
            if val_stats.get("eff_at_1pct") is not None:
                best_metrics["eff_at_1pct"] = float(val_stats["eff_at_1pct"])
            if val_stats.get("jsd") is not None:
                best_metrics["jsd"] = float(val_stats["jsd"])
            if val_stats.get("disco_loss") is not None:
                best_metrics["disco_loss"] = float(val_stats["disco_loss"])
            with open(os.path.join(save_dir, "best_metrics.json"), "w") as f:
                json.dump(best_metrics, f, indent=2)
            # Save ROC curve for this best epoch (overwrites previous best).
            # The file left on disk when training ends corresponds to best.pt.
            _save_roc_curve(
                val_probs, val_targets, save_dir,
                epoch=epoch,
                title_suffix=f"{experiment}, λ={disco_lambda}"
                if experiment in ("disco", "cure_disco") else experiment,
            )
        else:
            epochs_since_val_loss_improvement += 1

        for k, v in train_stats.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if (
            early_stop_patience is not None
            and epochs_since_val_loss_improvement >= early_stop_patience
        ):
            logger.info(
                "Early stopping at epoch %d (validation loss did not improve for %d epochs)",
                epoch,
                early_stop_patience,
            )
            break

    writer.close()

    # Inference to out_h5 if requested
    if out_h5:
        _run_finetune_inference(
            model, best_path, x_tr, y_tr, x_va, y_va,
            batch_size, device, rav_tensors, rav_mu, rav_std, alpha,
            svd_f, svd_fm, svd_r, svd_rm,
            train_group, val_group, out_h5,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Finetune experiments (2.a–2.d)")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("--h5", default=DEFAULT_H5, help=f"Input H5 (default: {DEFAULT_H5})")
    p.add_argument("--save-dir", default="runs/finetune")
    p.add_argument("--ParT-model-path", default=DEFAULT_PART_MODEL, dest="Part_model_path",
                   help=f"ParT checkpoint (default: {DEFAULT_PART_MODEL})")
    p.add_argument("--rav-h5", default=None,
                   help="H5 file containing RAV vectors (default: same as --h5)")
    p.add_argument("--rav-tensors-key", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--svd-forget", default=None)
    p.add_argument("--svd-retain", default=None)
    p.add_argument("--svd-k", type=int, default=None)
    p.add_argument("--svd-variance", type=float, default=0.95)
    p.add_argument("--train-group", default="/train")
    p.add_argument("--val-group", default="/val")
    p.add_argument("--x-key", default="cls_tokens_ln", help="Feature key (default: cls_tokens_ln)")
    p.add_argument("--y-key", default="label", help="Label key (default: label)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--disco-lambda", type=float, default=10.0)
    p.add_argument("--disco-variable", default="jet_sdmass", help="Aux variable for DisCo (default: jet_sdmass)")
    p.add_argument(
        "--disco-bkg-label",
        type=int,
        default=0,
        help="True-class label for DisCo subset (default: 0 = QCD). Use -1 for all samples in the batch.",
    )
    p.add_argument("--jsd-variable", default="jet_sdmass", help="Aux variable for JSD (default: jet_sdmass)")
    p.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    p.add_argument("--balance-samples", choices=["none", "oversample", "undersample", "both"], default="none")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-sched", choices=["none", "plateau-train", "plateau-val"], default="plateau-train")
    p.add_argument("--plateau-patience", type=int, default=5)
    p.add_argument("--plateau-factor", type=float, default=0.1)
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after N epochs with no improvement in validation loss "
        "(CE + λ·DisCo when disco enabled; same as best.pt; default: disabled)",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-h5", default=None)
    p.add_argument(
        "--pretrained-head",
        action="store_true",
        help="Initialize the linear head from --ParT-model-path even for experiments "
        "that are random-init by default (e.g. 'disco'). Requires --ParT-model-path.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.experiment == "disco" and not args.pretrained_head and args.Part_model_path:
        logger.warning(
            "disco experiment is random-init by default; pass --pretrained-head to use --ParT-model-path"
        )
    if args.pretrained_head and not args.Part_model_path:
        raise ValueError("--pretrained-head requires --ParT-model-path")
    if args.experiment == "rav" and not args.rav_tensors_key:
        raise ValueError("--rav-tensors-key required for experiment=rav")
    if args.experiment == "cure" and not args.svd_forget:
        raise ValueError("--svd-forget required for experiment=cure")
    if args.experiment == "cure_disco" and not args.svd_forget:
        raise ValueError("--svd-forget required for experiment=cure_disco")
    if args.experiment in ("part_og", "rav", "cure", "cure_disco") and not args.Part_model_path:
        raise ValueError(f"--part-model-path required for experiment={args.experiment}")

    run(
        args.experiment,
        args.h5,
        args.save_dir,
        Part_model_path=args.Part_model_path,
        rav_h5=args.rav_h5,
        rav_tensors_key=args.rav_tensors_key,
        alpha=args.alpha,
        svd_forget=args.svd_forget,
        svd_retain=args.svd_retain,
        svd_k=args.svd_k,
        svd_variance=args.svd_variance,
        train_group=args.train_group,
        val_group=args.val_group,
        x_key=args.x_key,
        y_key=args.y_key,
        train_frac=args.train_frac,
        data_fraction=args.data_fraction,
        disco_lambda=args.disco_lambda,
        disco_variable=args.disco_variable,
        disco_bkg_label=args.disco_bkg_label,
        jsd_variable=args.jsd_variable,
        class_weights=args.class_weights,
        balance_samples_method=args.balance_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_sched=args.lr_sched,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=args.device,
        out_h5=args.out_h5,
        pretrained_head=args.pretrained_head,
    )


if __name__ == "__main__":
    main()
