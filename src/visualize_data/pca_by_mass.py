#!/usr/bin/env python3
"""Project hidden states via PCA and plot samples grouped by mass bins."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from visualize_data.pca_hidden import compute_pca, project_samples, sample_for_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA projections with samples coloured by mass intervals.",
    )
    parser.add_argument(
        "--input",
        default="hidden_out/train_5M_hiddens_w_label.h5",
        help="Path to the input HDF5 file containing hidden states.",
    )
    parser.add_argument(
        "--dataset",
        default="cls_tokens_0",
        help="Dataset name inside the HDF5 file containing the features.",
    )
    parser.add_argument(
        "--mass-dataset",
        default="jet_sdmass",
        help="Dataset name storing the mass value associated with each example.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=10,
        help="Number of principal components to retain when fitting PCA (<=0 keeps all).",
    )
    parser.add_argument(
        "--plot-components",
        type=int,
        default=2,
        help="Number of PCA components to project onto for the scatter plot.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="Rows to stream at once when computing PCA.",
    )
    parser.add_argument(
        "--sample-for-plot",
        type=int,
        default=50_000,
        help="Number of points to sample for plotting (0 disables the plot).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling points for plotting.",
    )
    parser.add_argument(
        "--mass-edges",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit mass bin edges (e.g. --mass-edges 0 50 100 200).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=4,
        help="Number of mass bins if --mass-edges is not provided.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the output PNG (defaults to <dataset>_pca_by_mass.png).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def compute_edges(masses: np.ndarray, explicit_edges: Sequence[float] | None, num_bins: int) -> np.ndarray:
    if masses.size == 0:
        raise ValueError("No mass values available to derive bin edges.")

    if explicit_edges:
        edges = np.asarray(explicit_edges, dtype=np.float64)
        if edges.size < 2:
            raise ValueError("Provide at least two mass edges.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("Mass edges must be strictly increasing.")
        return edges

    if num_bins < 1:
        raise ValueError("num_bins must be >= 1 when --mass-edges is omitted.")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(masses, quantiles)
    edges[0] = masses.min()
    edges[-1] = masses.max()

    unique_edges = [edges[0]]
    for value in edges[1:]:
        if value - unique_edges[-1] > 1e-12:
            unique_edges.append(value)
    edges = np.asarray(unique_edges, dtype=np.float64)
    if edges.size < 2:
        raise ValueError(
            "Mass distribution collapses to a single value; provide custom --mass-edges."
        )
    return edges


def iter_bins(edges: np.ndarray) -> Iterable[tuple[float, float, bool]]:
    for idx in range(len(edges) - 1):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        last = idx == len(edges) - 2
        yield low, high, last


def format_bin_label(low: float, high: float, inclusive_high: bool) -> str:
    right_bracket = "]" if inclusive_high else ")"
    return f"[{low:.3f}, {high:.3f}{right_bracket}"


def render_plot(
    projection: np.ndarray,
    masses: np.ndarray,
    edges: np.ndarray,
    dataset_name: str,
    mass_dataset: str,
    title: str | None,
    output_path: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed."
        ) from exc

    labels = []
    masks = []
    for low, high, last in iter_bins(edges):
        if last:
            mask = (masses >= low) & (masses <= high)
        else:
            mask = (masses >= low) & (masses < high)
        if np.any(mask):
            labels.append(format_bin_label(low, high, last))
            masks.append(mask)

    if not masks:
        raise RuntimeError("No samples fall into the requested mass bins.")

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap("viridis", len(masks))
    for idx, (label, mask) in enumerate(zip(labels, masks)):
        color = cmap(idx) if len(masks) > 1 else None
        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=6,
            alpha=0.6,
            label=label,
            color=color,
        )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title or f"PCA projection for {dataset_name} grouped by {mass_dataset}")
    plt.legend(title="Mass bins", loc="best", fontsize="small")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.sample_for_plot <= 0:
        raise ValueError("sample-for-plot must be positive to generate a scatter plot.")

    with h5py.File(input_path, "r") as handle:
        if args.dataset not in handle:
            raise KeyError(f"Dataset '{args.dataset}' not found in {input_path}")
        if args.mass_dataset not in handle:
            raise KeyError(f"Dataset '{args.mass_dataset}' not found in {input_path}")

        feature_ds = handle[args.dataset]
        mass_ds = handle[args.mass_dataset]

        n_components = None if args.components is not None and args.components <= 0 else args.components
        result = compute_pca(feature_ds, args.chunk_size, n_components)

        plot_components = min(args.plot_components, result.components.shape[1])
        if plot_components < 2:
            raise ValueError(
                "Need at least two retained PCA components to build a 2D plot."
            )

        samples, masses = sample_for_projection(
            feature_ds,
            mass_ds,
            args.sample_for_plot,
            args.seed,
        )
        if samples.size == 0 or masses is None:
            raise RuntimeError("Sampling returned no data; adjust --sample-for-plot.")

        masses = masses.reshape(-1).astype(np.float64, copy=False)
        edges = compute_edges(masses, args.mass_edges, args.num_bins)

        projection = project_samples(
            samples,
            result.mean,
            result.components[:, :plot_components],
        )

        if projection.shape[1] < 2:
            raise RuntimeError(
                "Projection has fewer than two dimensions; increase --plot-components."
            )

        output_path = args.output or Path(f"{args.dataset}_pca_by_mass.png")

        plot_path = render_plot(
            projection,
            masses,
            edges,
            args.dataset,
            args.mass_dataset,
            args.title,
            output_path,
        )

        print(f"Saved PCA-by-mass scatter plot to {plot_path}")


if __name__ == "__main__":
    main()
