"""Core utilities, data, model, and projection modules for mass classifier experiments."""

from .utils import set_seed, infer_num_classes
from .data import (
    load_h5_group,
    load_aux_from_h5,
    make_loader,
    stratified_split,
    balance_samples,
    join_path,
    has_split,
)
from .model import PerpClassifier, load_pretrained_part_weights
from .projections import (
    load_rav_vectors,
    load_svd_basis,
    project_rav,
    project_svd,
)
from .losses import distance_correlation, make_loss_and_metrics, compute_class_weights
from .metrics import compute_accuracy_and_jsd, compute_efficiency_at_misid_rate, _jsd_from_hists

__all__ = [
    "set_seed",
    "infer_num_classes",
    "load_h5_group",
    "load_aux_from_h5",
    "make_loader",
    "stratified_split",
    "balance_samples",
    "join_path",
    "has_split",
    "PerpClassifier",
    "load_pretrained_part_weights",
    "load_rav_vectors",
    "load_svd_basis",
    "project_rav",
    "project_svd",
    "distance_correlation",
    "make_loss_and_metrics",
    "compute_class_weights",
    "compute_accuracy_and_jsd",
    "compute_efficiency_at_misid_rate",
    "_jsd_from_hists",
]
