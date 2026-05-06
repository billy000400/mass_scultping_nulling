"""Data loading and preprocessing for H5 latent states."""

import re
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def join_path(group: str, key: str) -> str:
    g = group.strip("/")
    k = key.strip("/")
    return f"/{g}/{k}" if g else f"/{k}"


def load_h5_group(
    h5_path: str,
    group: str,
    x_key: str,
    y_key: Optional[str],
    *,
    data_fraction: float = 1.0,
    dtype: str = "float32",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(h5_path, "r") as f:
        x_ds = f[join_path(group, x_key)]
        N = len(x_ds)
        if data_fraction < 1.0:
            N = max(1, int(np.floor(N * data_fraction)))
        x = np.array(x_ds[:N], dtype=dtype).reshape(N, -1)

        y_np = None
        path_y = join_path(group, y_key) if y_key else None
        if path_y and path_y in f:
            y_raw = np.array(f[path_y][:N])
            per = y_raw.reshape(N, -1)
            if per.shape[1] == 1:
                y_vals = per[:, 0]
                y_np = np.rint(y_vals).astype(np.int64)
                if not np.all(np.isfinite(y_vals)):
                    raise ValueError("Non-finite values in labels.")
            else:
                y_np = per.argmax(axis=1).astype(np.int64)

        return x, y_np


def load_aux_from_h5(
    h5_path: str, group: str, key: str, n: int, dtype: str = "float32"
) -> np.ndarray:
    path = join_path(group, key)
    with h5py.File(h5_path, "r") as f:
        if path not in f:
            raise KeyError(f"Auxiliary variable '{key}' not found at {path} in {h5_path}")
        data = np.array(f[path][:n], dtype=dtype).reshape(-1)[:n]
    return data


def make_loader(
    x_np: np.ndarray,
    y_np: Optional[np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    disco_aux_np: Optional[np.ndarray] = None,
    jsd_aux_np: Optional[np.ndarray] = None,
) -> DataLoader:
    x_t = torch.from_numpy(x_np.astype(np.float32))
    y_t = (
        torch.empty(len(x_t), dtype=torch.long)
        if y_np is None
        else torch.from_numpy(y_np.astype(np.int64))
    )
    if disco_aux_np is not None or jsd_aux_np is not None:
        n = len(x_t)
        disco_t = (
            torch.from_numpy(disco_aux_np.astype(np.float32))
            if disco_aux_np is not None
            else torch.zeros(n)
        )
        jsd_t = (
            torch.from_numpy(jsd_aux_np.astype(np.float32))
            if jsd_aux_np is not None
            else torch.zeros(n)
        )
        td = TensorDataset(x_t, y_t, disco_t, jsd_t)
    else:
        td = TensorDataset(x_t, y_t)
    return DataLoader(
        td, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True
    )


def stratified_split(
    y: np.ndarray, frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    y = np.asarray(y).reshape(-1).astype(int)
    idx = np.arange(len(y))
    train_idx, val_idx = [], []
    for c in np.unique(y):
        mask = y == c
        cls_idx = idx[mask]
        rng.shuffle(cls_idx)
        k = int(round(len(cls_idx) * frac))
        train_idx.append(cls_idx[:k])
        val_idx.append(cls_idx[k:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def balance_samples(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "oversample",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1).astype(int)
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) <= 1:
        return x, y, np.arange(len(y))

    if method == "oversample":
        target_count = counts.max()
    elif method == "undersample":
        target_count = counts.min()
    elif method == "both":
        target_count = int(counts.mean())
    else:
        raise ValueError(f"Unknown balance method: {method}")

    indices = []
    rng = np.random.RandomState(42)
    for cls in unique:
        mask = y == cls
        cls_indices = np.where(mask)[0]
        current_count = len(cls_indices)
        if current_count == target_count:
            indices.append(cls_indices)
        elif current_count < target_count:
            extra_needed = target_count - current_count
            extra_indices = rng.choice(cls_indices, size=extra_needed, replace=True)
            indices.append(np.concatenate([cls_indices, extra_indices]))
        else:
            selected = rng.choice(cls_indices, size=target_count, replace=False)
            indices.append(selected)

    balanced_indices = np.concatenate(indices)
    rng.shuffle(balanced_indices)
    return x[balanced_indices], y[balanced_indices], balanced_indices


def has_split(h5_path: str, train_group: str, val_group: str, x_key: str) -> bool:
    with h5py.File(h5_path, "r") as f:
        return join_path(train_group, x_key) in f and join_path(val_group, x_key) in f
