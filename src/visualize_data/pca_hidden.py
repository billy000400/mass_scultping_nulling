#!/usr/bin/env python3
"""Tooling for running PCA on hidden state HDF5 dumps."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import h5py
import numpy as np


@dataclass
class PCAResult:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


def reshape_chunk(chunk: np.ndarray) -> np.ndarray:
    """Flatten per-example tensors into vectors."""
    chunk = np.asarray(chunk)
    return chunk.reshape(chunk.shape[0], -1)


def iter_chunks(dataset: h5py.Dataset, chunk_size: int) -> Iterator[np.ndarray]:
    total = dataset.shape[0]
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        yield dataset[start:stop]


def compute_pca(
    dataset: h5py.Dataset,
    chunk_size: int,
    n_components: int | None = None,
) -> PCAResult:
    total_rows = dataset.shape[0]
    if total_rows < 2:
        raise ValueError("Need at least two rows to compute PCA.")

    first_chunk = reshape_chunk(dataset[0:1])
    feature_dim = first_chunk.shape[1]

    accum_sum = np.zeros(feature_dim, dtype=np.float64)
    gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    count = 0

    for raw_chunk in iter_chunks(dataset, chunk_size):
        chunk = reshape_chunk(raw_chunk)
        chunk = chunk.astype(np.float64, copy=False)
        count += chunk.shape[0]
        accum_sum += chunk.sum(axis=0)
        gram += chunk.T @ chunk

    mean = accum_sum / count
    centered_gram = gram - count * np.outer(mean, mean)
    covariance = centered_gram / (count - 1)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    variance_total = eigvals.sum()
    if variance_total <= 0:
        raise RuntimeError("Total variance is non-positive; check input data.")

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    explained_variance_ratio = eigvals / variance_total

    return PCAResult(
        mean=mean,
        components=eigvecs,
        explained_variance=eigvals,
        explained_variance_ratio=explained_variance_ratio,
    )


def sample_for_projection(
    dataset: h5py.Dataset,
    label_ds: Optional[h5py.Dataset],
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if sample_size <= 0:
        return np.empty((0, 0)), None

    total_rows = dataset.shape[0]
    if total_rows == 0:
        return np.empty((0, 0)), None

    sample_size = min(sample_size, total_rows)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))

    sampled = reshape_chunk(dataset[indices])
    labels = None
    if label_ds is not None:
        labels = np.asarray(label_ds[indices]).reshape(sample_size, -1)
        labels = labels.squeeze()

    return sampled.astype(np.float64, copy=False), labels


def project_samples(
    samples: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    if samples.size == 0:
        return samples
    centered = samples - mean
    return centered @ components


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute PCA on hidden state data stored in an HDF5 file.",
    )
    parser.add_argument(
        "--input",
        default="hidden_out/train_5M_hiddens_w_label.h5",
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--dataset",
        default="cls_tokens_0",
        help="Dataset name inside the HDF5 file containing the features.",
    )
    parser.add_argument(
        "--label-dataset",
        default="label",
        help="Optional dataset name for labels (used for coloring plots).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=10,
        help="Number of principal components to retain.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Number of rows to load per chunk when streaming from disk.",
    )
    parser.add_argument(
        "--sample-for-plot",
        type=int,
        default=50000,
        help="Number of examples to sample for plotting (0 to disable).",
    )
    parser.add_argument(
        "--plot-components",
        type=int,
        default=2,
        help="Number of principal components to project onto for the plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PCA artefacts will be stored (created if missing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling when plotting.",
    )
    parser.add_argument(
        "--plot-filename",
        default=None,
        help="Optional filename for the scatter plot (defaults to dataset name).",
    )
    return parser


def maybe_write_npz(output_dir: Path, dataset_name: str, result: PCAResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_name}_pca.npz"
    np.savez(
        path,
        mean=result.mean,
        components=result.components,
        explained_variance=result.explained_variance,
        explained_variance_ratio=result.explained_variance_ratio,
    )
    return path


def maybe_write_summary(output_dir: Path, dataset_name: str, result: PCAResult) -> Path:
    summary = {
        "dataset": dataset_name,
        "components": int(result.components.shape[1]),
        "explained_variance": result.explained_variance.tolist(),
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "explained_variance_ratio_cumulative": np.cumsum(result.explained_variance_ratio).tolist(),
    }
    output_path = output_dir / f"{dataset_name}_pca_summary.json"
    dump_json(output_path, summary)
    return output_path


def maybe_render_plot(
    output_dir: Path,
    dataset_name: str,
    projection: np.ndarray,
    labels: Optional[np.ndarray],
    filename: Optional[str],
) -> Optional[Path]:
    if projection.size == 0:
        return None

    plot_path = output_dir / (filename or f"{dataset_name}_pca_scatter.png")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed."
        )

    components = projection.shape[1]
    if components < 2:
        raise ValueError("Need at least 2 projection dimensions to plot a scatter chart.")

    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(projection[:, 0], projection[:, 1], s=4, alpha=0.5)
    else:
        scatter = plt.scatter(
            projection[:, 0],
            projection[:, 1],
            c=labels,
            s=4,
            alpha=0.5,
            cmap="viridis",
        )
        plt.colorbar(scatter, label="label")

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"PCA projection for {dataset_name}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as handle:
        if args.dataset not in handle:
            raise KeyError(f"Dataset '{args.dataset}' not found in {input_path}")

        dataset = handle[args.dataset]
        label_ds = handle.get(args.label_dataset) if args.label_dataset else None

        if args.components is not None and args.components <= 0:
            n_components = None
        else:
            n_components = args.components

        result = compute_pca(dataset, args.chunk_size, n_components)

        print(f"Computed PCA for dataset '{args.dataset}' with {result.components.shape[1]} components.")
        print("Explained variance ratio (first 10):")
        preview = result.explained_variance_ratio[:10]
        print("  ", " ".join(f"{val:.4f}" for val in preview))

        if args.output_dir:
            output_dir = args.output_dir
            artefacts_dir = output_dir
            npz_path = maybe_write_npz(artefacts_dir, args.dataset, result)
            summary_path = maybe_write_summary(artefacts_dir, args.dataset, result)
            print(f"Saved PCA artefacts to {npz_path}")
            print(f"Saved PCA summary to {summary_path}")

            if args.sample_for_plot > 0:
                plot_components = min(args.plot_components, result.components.shape[1])
                if plot_components < 2:
                    print(
                        "Not enough components retained to build a 2D plot; "
                        "increase --components to enable plotting.",
                    )
                else:
                    samples, labels = sample_for_projection(
                        dataset,
                        label_ds,
                        args.sample_for_plot,
                        args.seed,
                    )
                    projection = project_samples(
                        samples,
                        result.mean,
                        result.components[:, :plot_components],
                    )
                    plot_path = maybe_render_plot(
                        artefacts_dir,
                        args.dataset,
                        projection,
                        labels,
                        args.plot_filename,
                    )
                    if plot_path:
                        print(f"Saved PCA scatter plot to {plot_path}")
        else:
            if args.sample_for_plot > 0:
                print("Skipping plotting because no output directory was provided.")


if __name__ == "__main__":
    main()
