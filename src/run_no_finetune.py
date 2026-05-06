#!/usr/bin/env python3
"""CLI entry point for no-finetune experiments (1.a, 1.b, 1.c).

Usage:
  python -m src.run_no_finetune --experiment part_og --h5 DATA.h5 --out-h5 SCORES.h5 [--ParT-model-path PART.pt]
  python -m mass_perp_classifier.src.run_no_finetune --experiment rav --rav-tensors-key RAV_mass --alpha 1.0 ...
  python -m mass_perp_classifier.src.run_no_finetune --experiment cure --svd-forget forget.npz --svd-retain retain.npz ...
"""

from .experiments.no_finetune import main

if __name__ == "__main__":
    main()
