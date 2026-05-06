"""Projection utilities: RAV (perpendicular) and SVD (CURE-style forget/retain)."""

import re
from typing import Optional, Tuple

import h5py
import numpy as np
import torch


def _extract_order(name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def load_rav_vectors(
    h5_path: str, key: str
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load RAV vectors and optional z-space normalization (mean, std) from HDF5.
    Returns (vectors, mu, std). mu and std are None for legacy RAVs (original-space)."""
    vectors: list[tuple[int, np.ndarray]] = []
    search_key = key.strip("/")

    with h5py.File(h5_path, "r") as hf:
        obj = hf.get(search_key)

        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj, dtype=np.float32)
            if arr.ndim == 1:
                vectors.append((0, arr.reshape(-1)))
            elif arr.ndim == 2:
                if 1 in arr.shape:
                    vectors.append((0, arr.reshape(-1)))
                else:
                    for idx, vec in enumerate(arr):
                        vectors.append((idx, np.array(vec, dtype=np.float32).reshape(-1)))
            else:
                raise ValueError(f"Expected 1D/2D dataset for '{key}', got shape {arr.shape}")

        elif isinstance(obj, h5py.Group):
            for name, ds in obj.items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                order = _extract_order(name)
                vectors.append(
                    (order if order is not None else len(vectors), np.array(ds, dtype=np.float32).reshape(-1))
                )

        if not vectors:
            prefix = search_key
            matches: list[tuple[int, np.ndarray]] = []
            for name, ds in hf.items():
                if not isinstance(ds, h5py.Dataset) or not name.startswith(prefix):
                    continue
                order = _extract_order(name)
                matches.append(
                    (order if order is not None else len(matches), np.array(ds, dtype=np.float32).reshape(-1))
                )
            if matches:
                vectors.extend(matches)

    if not vectors:
        raise KeyError(f"No datasets found for key/prefix '{key}' in {h5_path}")

    vectors.sort(key=lambda item: item[0])
    arrays = [vec for _, vec in vectors]
    arr = arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=0)

    # Load z-space normalization if present (new format)
    base = search_key.rsplit("_order", 1)[0] if "_order" in search_key else search_key.rsplit("_", 1)[0]
    mu = std = None
    with h5py.File(h5_path, "r") as hf:
        if f"{base}_mean" in hf and f"{base}_std" in hf:
            mu = np.array(hf[f"{base}_mean"][:], dtype=np.float32)
            std = np.array(hf[f"{base}_std"][:], dtype=np.float32)
            std = np.where(std > 1e-12, std, 1.0)

    return arr, mu, std


def load_svd_basis(
    npz_path: str, k: Optional[int] = None, variance: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load SVD .npz and select top-k right singular vectors."""
    data = np.load(npz_path)
    Vh = data["Vh"]
    s = data["s"]
    mean = data["mean"]

    if k is None:
        var_explained = s**2
        cum_var = np.cumsum(var_explained) / np.sum(var_explained)
        k = int(np.searchsorted(cum_var, variance)) + 1
        k = min(k, len(s))

    return Vh[:k].astype(np.float32), mean.astype(np.float32), k


def _prepare_hidden(hidden_tensor: torch.Tensor) -> torch.Tensor:
    if hidden_tensor.dim() == 3:
        hidden_tensor = hidden_tensor.squeeze(0)
    if hidden_tensor.dim() == 1:
        hidden_tensor = hidden_tensor.unsqueeze(0)
    return hidden_tensor


def project_rav(
    hidden_tensor: torch.Tensor,
    rav_stack: torch.Tensor,
    eps: float = 1e-12,
    alpha: float = 1.0,
    mu: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project hidden_tensor to be orthogonal to each vector in rav_stack (alpha-scaled).
    When mu and std are provided (z-space RAV), z-score before projecting and un-z-score after."""
    v = _prepare_hidden(hidden_tensor)
    if rav_stack.dim() == 1:
        rav_stack = rav_stack.unsqueeze(0)
    if rav_stack.dim() == 2:
        rav_stack = rav_stack.unsqueeze(0)
    if rav_stack.size(0) == 1 and v.size(0) > 1:
        rav_stack = rav_stack.expand(v.size(0), -1, -1)

    if mu is not None and std is not None:
        mu_t = mu.to(v.device) if isinstance(mu, torch.Tensor) else torch.from_numpy(mu).to(v.device)
        std_t = std.to(v.device) if isinstance(std, torch.Tensor) else torch.from_numpy(std).to(v.device)
        if mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0)
        if std_t.dim() == 1:
            std_t = std_t.unsqueeze(0)
        if v.size(0) > 1 and mu_t.size(0) == 1:
            mu_t = mu_t.expand(v.size(0), -1)
            std_t = std_t.expand(v.size(0), -1)
        w = (v - mu_t) / std_t.clamp_min(eps)
    else:
        w = v

    for order in range(rav_stack.size(1)):
        u = rav_stack[:, order, :]
        denom = (u * u).sum(dim=-1, keepdim=True).clamp_min(eps)
        proj = ((w * u).sum(dim=-1, keepdim=True) / denom) * u
        w = w - alpha * proj

    if mu is not None and std is not None:
        w = w * std_t + mu_t
    return w


def project_svd(
    x: torch.Tensor,
    forget_basis: Optional[torch.Tensor],
    forget_mean: Optional[torch.Tensor],
    retain_basis: Optional[torch.Tensor],
    retain_mean: Optional[torch.Tensor],
    alpha: float = 1.0,
) -> torch.Tensor:
    """CURE-style: x_origin - alpha * (proj onto forget) + alpha * (proj onto retain).

    Both projections use x_origin. alpha=0 returns x unchanged (validates correctness).
    """
    x_orig = x
    forget_proj = torch.zeros_like(x)
    retain_proj = torch.zeros_like(x)

    if forget_basis is not None and forget_mean is not None:
        x_c = x - forget_mean
        forget_proj = (x_c @ forget_basis.T) @ forget_basis

    if retain_basis is not None and retain_mean is not None:
        x_c = x - retain_mean
        retain_proj = (x_c @ retain_basis.T) @ retain_basis + retain_mean

    x = x_orig - alpha * forget_proj + alpha * retain_proj
    return x
