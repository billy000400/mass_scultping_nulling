#!/usr/bin/env python3
"""Run SVD on a latent space stored in an HDF5 file under a given key.

Supports filtering by class label (default: QCD only). See mass_perp_classifier/CLASS_INDICES.md
for valid class indices and label strings.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Class label mapping (see mass_perp_classifier/CLASS_INDICES.md)
# Short names used for --class-filter (e.g. QCD, Hbb)
CLASS_NAMES = [
    "QCD", "Hbb", "Hcc", "Hgg", "H4q",
    "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
]


def _parse_class_filter(s: str) -> int | None:
    """Parse --class-filter: int, class name (e.g. QCD, Hbb), or 'all' for None."""
    if s.strip().lower() in ("all", "none", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    t = s.strip()
    try:
        return CLASS_NAMES.index(t)
    except ValueError:
        raise ValueError(
            f"Invalid class filter '{s}'. Use integer (0-9), class name (e.g. QCD, Hbb), or 'all'. "
            f"See CLASS_INDICES.md for valid values."
        ) from None


def _class_index_to_string(idx: int) -> str:
    """Convert class index to display name (e.g. QCD, Hbb)."""
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class{idx}"


def load_labels_from_h5(
    h5_path: str,
    label_key: str,
    max_samples: int | None = None,
) -> np.ndarray:
    """Load integer class labels from HDF5. Handles 1D int or 2D one-hot."""
    with h5py.File(h5_path, "r") as hf:
        if label_key not in hf:
            raise KeyError(f"Label key '{label_key}' not found in {h5_path}")
        lab_ds = hf[label_key]
        n_total = lab_ds.shape[0]
        n_read = min(n_total, max_samples) if max_samples is not None else n_total
        raw = np.asarray(lab_ds[:n_read])
    labels = raw.reshape(n_read, -1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.shape[1] == 1:
        return np.rint(labels[:, 0]).astype(np.int64)
    return np.argmax(labels, axis=1).astype(np.int64)


def load_latent_from_h5(
    h5_path: str,
    key: str,
    max_samples: int | None = None,
) -> np.ndarray:
    """Load latent array from an HDF5 file at the given key.

    Supports a direct dataset key or a group containing one dataset.
    Returns array of shape (n_samples, n_features).
    """
    key = key.strip("/")
    with h5py.File(h5_path, "r") as hf:
        obj = hf.get(key)
        if obj is None:
            raise KeyError(f"Key '{key}' not found in {h5_path}")

        if isinstance(obj, h5py.Dataset):
            n_total = obj.shape[0]
            n_read = min(n_total, max_samples) if max_samples is not None else n_total
            data = np.asarray(obj[:n_read], dtype=np.float64)
        elif isinstance(obj, h5py.Group):
            names = [n for n, v in obj.items() if isinstance(v, h5py.Dataset)]
            if not names:
                raise KeyError(f"Group '{key}' in {h5_path} has no datasets")
            if len(names) > 1:
                logger.warning(
                    "Group '%s' has multiple datasets %s; using first: %s",
                    key, names, names[0],
                )
            ds = obj[names[0]]
            n_total = ds.shape[0]
            n_read = min(n_total, max_samples) if max_samples is not None else n_total
            data = np.asarray(ds[:n_read], dtype=np.float64)
        else:
            raise TypeError(f"Key '{key}' is neither Dataset nor Group in {h5_path}")

    # Flatten to (n_samples, n_features)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    return data


def run_svd(
    X: np.ndarray,
    n_components: int | None = None,
    center: bool = True,
    use_randomized: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Run SVD on data matrix X (n_samples, n_features).

    Returns:
        U: (n_samples, n_comp), left singular vectors
        s: (n_comp,) singular values
        Vh: (n_comp, n_features), right singular vectors (components)
        mean: (n_features,) or None if center=False
    """
    if center:
        mean = np.asarray(X.mean(axis=0), dtype=np.float64)
        X = X - mean
    else:
        mean = None

    n_samples, n_features = X.shape
    k = min(n_samples, n_features)
    if n_components is not None:
        k = min(k, n_components)

    if k < 1:
        raise ValueError("Need at least one sample and one feature for SVD")

    if use_randomized and n_components is not None and k <= min(n_samples, n_features):
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vh = randomized_svd(
                X, n_components=k, n_oversamples=10, random_state=0
            )
            return U, s, Vh, mean
        except Exception as e:
            logger.warning("Randomized SVD failed (%s), falling back to full SVD", e)

    # Full SVD then truncate
    full_matrices = False
    U, s, Vh = np.linalg.svd(X, full_matrices=full_matrices, hermitian=False)
    U = U[:, :k]
    s = s[:k]
    Vh = Vh[:k, :]
    return U, s, Vh, mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SVD on a latent space from an HDF5 file and save the result.",
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to the .h5 file containing the latent dataset",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="HDF5 key for the latent dataset (e.g. 'cls_tokens_ln')",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag used to name the output SVD result file",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where the SVD result file will be saved",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of components to keep (default: keep all)",
    )
    parser.add_argument(
        "--no_center",
        action="store_true",
        help="Do not center the data before SVD",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of samples to load (for faster runs or testing)",
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="Use randomized SVD when --n_components is set (faster for large data)",
    )
    parser.add_argument(
        "--label-key",
        default="label",
        help="HDF5 key for class labels (required when using --class-filter)",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        default="QCD",
        help="Restrict to samples with this class (int, class name e.g. QCD/Hbb, or 'all'). Default: QCD. See CLASS_INDICES.md.",
    )
    args = parser.parse_args()

    class_filter = _parse_class_filter(args.class_filter)

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    if class_filter is None:
        logger.info("Using ALL classes (no class filter).")
    else:
        logger.info("Filtering to class %d (%s) only.", class_filter, _class_index_to_string(class_filter))

    logger.info("Loading latent from %s key=%s", args.h5, args.key)
    X = load_latent_from_h5(
        str(h5_path), args.key, max_samples=args.max_samples
    )
    logger.info("Latent shape (before filter): %s", X.shape)

    if class_filter is not None:
        labels = load_labels_from_h5(
            str(h5_path), args.label_key, max_samples=args.max_samples
        )
        if labels.shape[0] != X.shape[0]:
            raise ValueError(
                f"Label length {labels.shape[0]} does not match latent rows {X.shape[0]}"
            )
        mask = labels == class_filter
        n_matching = int(np.sum(mask))
        if n_matching == 0:
            raise ValueError(
                f"No samples with class {class_filter} ({_class_index_to_string(class_filter)}) in dataset"
            )
        X = X[mask]
        n_discarded = labels.shape[0] - n_matching
        logger.info(
            "Filtered to %d samples (class %d = %s); discarded %d samples.",
            n_matching,
            class_filter,
            _class_index_to_string(class_filter),
            n_discarded,
        )
        logger.info("Latent shape (after filter): %s", X.shape)

    U, s, Vh, mean = run_svd(
        X,
        n_components=args.n_components,
        center=not args.no_center,
        use_randomized=args.randomized,
    )
    logger.info("SVD: U %s, s %s, Vh %s", U.shape, s.shape, Vh.shape)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{args.tag}_svd.npz"
    out_path = args.output_dir / out_name

    save_dict = {
        "U": U,
        "s": s,
        "Vh": Vh,
        "n_samples": np.int64(X.shape[0]),
        "n_features": np.int64(X.shape[1]),
    }
    if mean is not None:
        save_dict["mean"] = mean
    if class_filter is not None:
        save_dict["class_filter"] = np.int64(class_filter)
        save_dict["class_filter_name"] = np.array(_class_index_to_string(class_filter), dtype="U")

    np.savez_compressed(out_path, **save_dict)
    logger.info("Saved SVD result to %s", out_path)


if __name__ == "__main__":
    main()
