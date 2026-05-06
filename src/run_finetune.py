#!/usr/bin/env python3
"""CLI entry point for finetune experiments (2.a, 2.b, 2.c, 2.d).

Usage:
  python -m src.run_finetune --experiment disco --h5 DATA.h5 --save-dir runs/disco
  python -m src.run_finetune --experiment part_og [--ParT-model-path PART.pt] ...
  python -m src.run_finetune --experiment rav --rav-tensors-key RAV_mass ...
  python -m src.run_finetune --experiment cure --svd-forget forget.npz ...
"""

from .experiments.finetune import main

if __name__ == "__main__":
    main()
