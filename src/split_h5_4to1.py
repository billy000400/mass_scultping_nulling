#!/usr/bin/env python3
"""
Split a single HDF5 file into /train and /val groups (4:1 by default) with
identical dataset names: X and y — and also copy a shared rav tensor.

Improvements vs v2:
- Copies the rav tensor (shared projection vector) from --rav-key in the input
  to the same key in the output, preserving dtype, shape, compression, chunks,
  and attributes. Copy is streamed along axis 0 if needed.
- Avoids h5py fancy indexing by grouping contiguous runs and reading slices.
- Copies X/y in chunks; does not load entire arrays into RAM.
- Stratified split for classification (1D int or 2D one-hot labels).

Usage:
  python split_h5_4to1_v3.py \
    --in path/to/dataset.h5 \
    --out path/to/dataset_split.h5 \
    --x-key /X --y-key /y --rav-key /rav_tensors \
    --seed 1337 --train-frac 0.8
"""
import argparse
import os
import h5py
import numpy as np


def infer_labels(y_dset, limit=None):
    """Return integer class labels if possible; else None.
    - 1D: assume integer class ids
    - 2D: assume one-hot, take argmax
    """
    shape = y_dset.shape
    if len(shape) == 1:
        return y_dset[:limit].astype(np.int64) if limit else y_dset[:].astype(np.int64)
    if len(shape) == 2:
        if limit:
            y = y_dset[:limit]
            return np.argmax(y, axis=1).astype(np.int64)
        n = shape[0]
        out = np.empty(n, dtype=np.int64)
        step = max(1, 1_000_000 // max(1, shape[1]))
        for i in range(0, n, step):
            sl = slice(i, min(i + step, n))
            out[sl] = np.argmax(y_dset[sl], axis=1).astype(np.int64)
        return out
    return None


def stratified_indices(labels, train_frac=0.8, seed=1337):
    rng = np.random.RandomState(seed)
    idxs = np.arange(labels.shape[0])
    train, val = [], []
    for c in np.unique(labels):
        mask = labels == c
        cls_idx = idxs[mask]
        rng.shuffle(cls_idx)
        k = int(round(len(cls_idx) * train_frac))
        train.append(cls_idx[:k])
        val.append(cls_idx[k:])
    return np.concatenate(train), np.concatenate(val)


def random_indices(n, train_frac=0.8, seed=1337):
    rng = np.random.RandomState(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    k = int(round(n * train_frac))
    return idxs[:k], idxs[k:]


def copy_by_indices(src_dset, dst_dset, idxs, chunk=65536):
    """Copy rows from src_dset to dst_dset using indices in chunks.
    Avoid h5py fancy indexing by grouping contiguous runs and reading slices.
    Keeps the original order of "idxs" within each chunk.
    """
    n = len(idxs)
    tail = src_dset.shape[1:]
    for start in range(0, n, chunk):
        part = np.asarray(idxs[start:start + chunk], dtype=np.int64)
        # allocate output buffer for this chunk
        buf = np.empty((len(part),) + tail, dtype=src_dset.dtype)
        # sort part to find contiguous runs
        order = np.argsort(part)
        sorted_idx = part[order]
        i = 0
        while i < len(sorted_idx):
            j = i + 1
            while j < len(sorted_idx) and sorted_idx[j] == sorted_idx[j - 1] + 1:
                j += 1
            a = int(sorted_idx[i])
            b = int(sorted_idx[j - 1]) + 1
            data = src_dset[a:b]  # contiguous slice
            # place back into buffer at original positions for this chunk
            buf[order[i:j]] = data
            i = j
        # write contiguous block to destination
        dst_dset[start:start + len(part)] = buf


def clone_dataset_schema(fout, out_path, src_dset):
    """Create an empty dataset in fout mirroring src_dset's schema/compression."""
    return fout.create_dataset(
        out_path,
        shape=src_dset.shape,
        dtype=src_dset.dtype,
        chunks=src_dset.chunks,
        compression=src_dset.compression,
        compression_opts=src_dset.compression_opts,
        shuffle=src_dset.shuffle,
        fletcher32=src_dset.fletcher32,
    )


def copy_dataset(src_dset, fout, out_path, axis0_chunk=1_000_000):
    """Stream-copy an entire dataset to fout[out_path] preserving schema & attrs.
    Copies along axis 0 in chunks to avoid loading everything into RAM.
    """
    dst = clone_dataset_schema(fout, out_path, src_dset)
    # copy attributes
    for k, v in src_dset.attrs.items():
        dst.attrs[k] = v
    # handle 0-dim or small arrays in one shot
    if src_dset.ndim == 0:
        dst[...] = src_dset[...]
        return dst
    n = src_dset.shape[0]
    step = min(n, axis0_chunk)
    if src_dset.ndim == 1:
        for i in range(0, n, step):
            sl = slice(i, min(i + step, n))
            dst[sl] = src_dset[sl]
    else:
        for i in range(0, n, step):
            sl = slice(i, min(i + step, n))
            dst[sl, ...] = src_dset[sl, ...]
    return dst


def main():
    ap = argparse.ArgumentParser(description="4:1 train/val split for an H5 file (with rav tensor copy)")
    ap.add_argument('--in', dest='inp', required=True, help='Input H5 file with root /X and /y')
    ap.add_argument('--out', dest='out', required=True, help='Output H5 file to write /train and /val groups')
    ap.add_argument('--x-key', default='/X', help='Root dataset for features')
    ap.add_argument('--y-key', default='/y', help='Root dataset for labels')
    ap.add_argument('--rav-key', default='/rav_tensors', help='Root dataset key for shared rav tensor to copy (optional)')
    ap.add_argument('--train-frac', type=float, default=0.8, help='Train fraction (default 0.8 = 4:1)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--chunk', type=int, default=65536, help='Rows per copy chunk for X/y')
    ap.add_argument('--compression', default='lzf', help='Compression for output X/y datasets (e.g., lzf, gzip, None)')
    args = ap.parse_args()

    assert 0.0 < args.train_frac < 1.0, "train-frac must be in (0,1)"

    with h5py.File(args.inp, 'r') as fin:
        if args.x_key not in fin:
            raise KeyError(f"Missing {args.x_key} in input file")
        if args.y_key not in fin:
            raise KeyError(f"Missing {args.y_key} in input file")

        X = fin[args.x_key]
        y = fin[args.y_key]
        n = X.shape[0]
        feat_tail = X.shape[1:]
        print(f"Input: N={n}, X.tail={feat_tail}, X.dtype={X.dtype}, y.shape={y.shape}, y.dtype={y.dtype}")

        # Decide split strategy
        labels = infer_labels(y)
        if labels is not None:
            tr_idx, va_idx = stratified_indices(labels, train_frac=args.train_frac, seed=args.seed)
            print("Using stratified split over labels")
        else:
            tr_idx, va_idx = random_indices(n, train_frac=args.train_frac, seed=args.seed)
            print("Using random split (labels not 1D/2D)")
        print(f"Train={len(tr_idx)}  Val={len(va_idx)}  (frac={args.train_frac})")

        if os.path.abspath(args.out) == os.path.abspath(args.inp):
            raise ValueError("--out must differ from --in; refusing to overwrite input file")

        with h5py.File(args.out, 'w') as fout:
            # Prepare output datasets for X/y with requested compression
            tX = fout.create_dataset('/train/X', shape=(len(tr_idx),) + feat_tail, dtype=X.dtype,
                                     compression=args.compression, chunks=True)
            vX = fout.create_dataset('/val/X',   shape=(len(va_idx),) + feat_tail, dtype=X.dtype,
                                     compression=args.compression, chunks=True)
            y_tail = y.shape[1:]
            ty = fout.create_dataset('/train/y', shape=(len(tr_idx),) + y_tail, dtype=y.dtype,
                                     compression=args.compression, chunks=True)
            vy = fout.create_dataset('/val/y',   shape=(len(va_idx),) + y_tail, dtype=y.dtype,
                                     compression=args.compression, chunks=True)

            print("Copying train/X …")
            copy_by_indices(X, tX, tr_idx, chunk=args.chunk)
            print("Copying val/X …")
            copy_by_indices(X, vX, va_idx, chunk=args.chunk)
            print("Copying train/y …")
            copy_by_indices(y, ty, tr_idx, chunk=args.chunk)
            print("Copying val/y …")
            copy_by_indices(y, vy, va_idx, chunk=args.chunk)

            # Optionally copy rav tensor if present
            if args.rav_key and args.rav_key in fin:
                print(f"Copying rav tensor from {args.rav_key} …")
                copy_dataset(fin[args.rav_key], fout, args.rav_key)
            else:
                print("No rav tensor key found or --rav-key empty; skipping.")

        print(f"Done. Wrote {args.out}")


if __name__ == '__main__':
    main()

