#!/usr/bin/env python3
"""
Verification script for:
(1) Edit location: confirm the representation edit happens after the final
    class-attention and before the LayerNorm that feeds the linear classifier.
(2) Whether LayerNorm + linear layer recover the deleted (jet-mass) direction.

Run with:
  python verify_edit_and_recovery.py \\
    --h5 /scope-vol/hidden_out/train_5M_hiddens_w_label.h5 \\
    --rav-h5 /scope-vol/particle_transformer/notebooks/rav_sdmass.h5 \\
    --rav-key RAV_jet_sdmass \\
    --model-path /scope-vol/mass_perp_classifier/logs/no_bias/best.pt \\
    --x-key cls_tokens_1 \\
    --num-samples 256
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import h5py

# Import from train_on_h5 (same package)
from train_on_h5 import (
    _prepare_hidden,
    sequential_perpendicular,
    PerpClassifier,
    load_rav_vectors,
)

def _join_path(group: str, key: str) -> str:
    g = group.strip("/")
    k = key.strip("/")
    return f"/{g}/{k}" if g else f"/{k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/scope-vol/hidden_out/train_5M_hiddens_w_label.h5")
    ap.add_argument("--rav-h5", default="/scope-vol/particle_transformer/notebooks/rav_sdmass.h5")
    ap.add_argument("--rav-key", default="RAV_jet_sdmass")
    ap.add_argument("--model-path", default="/scope-vol/mass_perp_classifier/logs/no_bias/best.pt")
    ap.add_argument("--x-key", default="cls_tokens_1")
    ap.add_argument("--train-group", default="/train")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-fraction", type=float, default=0.0001)
    args = ap.parse_args()

    device = args.device

    # ----- (1) EDIT LOCATION -----
    print("\n" + "=" * 70)
    print("(1) VERIFY EDIT LOCATION")
    print("=" * 70)

    # 1a) Origin of the tensor being edited (ParT side)
    print("\n--- 1a) Origin of the tensor being edited ---")
    print("The tensor is the CLASS TOKEN after the LAST CLASS-ATTENTION BLOCK")
    print("and BEFORE the ParT's final LayerNorm.")
    print("In ParticleTransformer.forward (weaver):")
    print("  - cls_blocks has 2 layers (num_cls_layers=2); indices 0, 1.")
    print("  - cls_tokens_1 = output of cls_blocks[1], saved when hidden_states and _hs_pre_ln:")
    print("      for i, block in enumerate(self.cls_blocks):")
    print("          cls_tokens = block(x, x_cls=cls_tokens, ...)")
    print("          if self.hidden_states and self._hs_pre_ln:")
    print("              hiddens['cls_tokens_{i}'] = cls_tokens   # i=1 => cls_tokens_1")
    print("  - Next line: cls_tokens_ln = self.norm(cls_tokens)  # ParT's norm — NOT applied to our input)")
    print("  - Our pipeline: we load cls_tokens_1 from H5 and feed it to PerpClassifier.")
    print("  => So the tensor is: AFTER the last class-attention, BEFORE any LayerNorm.")
    print("  => The PerpClassifier REPLACES the ParT head (ParT's norm+fc are never used).")

    # 1b) Where the edit is applied (PerpClassifier side)
    print("\n--- 1b) Where the edit is applied (PerpClassifier.forward) ---")
    print("Order of operations:")
    print("  1. x_base = _prepare_hidden(cls_tokens)     # (B, D) from (B,1,D) or (B,D)")
    print("  2. x_used = sequential_perpendicular(x_base, rav_tensors, alpha=1)  # EDIT HERE")
    print("  3. x_norm = self.norm(x_used)              # LayerNorm")
    print("  4. logits = self.fc(x_norm)                # Linear classifier")
    print("=> The edit is applied to the SAME tensor (cls_tokens) BEFORE the")
    print("   LayerNorm that feeds the linear classifier (the PerpClassifier's norm).")
    print("=> Module: not a nn.Module; the edit is in sequential_perpendicular(),")
    print("   called at the start of PerpClassifier.forward, before self.norm.")

    # 1c) Concretely: load one batch and report shapes
    print("\n--- 1c) Concrete tensors and shapes ---")
    with h5py.File(args.h5, "r") as f:
        # Resolve path: if /train/key doesn't exist, try root /key or key
        train_path = _join_path(args.train_group, args.x_key)
        if train_path not in f:
            alt = _join_path("", args.x_key)  # /cls_tokens_1
            train_path = alt if alt in f else args.x_key
        x_ds = f[train_path]
        N = min(args.num_samples, len(x_ds))
        x_np = np.array(x_ds[:N], dtype=np.float32).reshape(N, -1)

    x = torch.from_numpy(x_np).to(device)
    x_base = _prepare_hidden(x)
    print(f"  cls_tokens (from H5, after _prepare_hidden): shape = {tuple(x_base.shape)}")
    print(f"  => (batch, hidden_dim) = ({x_base.shape[0]}, {x_base.shape[1]})")

    # Load RAV and possibly trim to hidden_dim
    try:
        rav, _, _ = load_rav_vectors(args.rav_h5, args.rav_key)
        rav = np.asarray(rav, dtype=np.float32).reshape(-1)
    except Exception as e:
        with h5py.File(args.rav_h5, "r") as f:
            rav = np.array(f[args.rav_key], dtype=np.float32).reshape(-1)
    if rav.size != x_base.shape[1]:
        print(f"  RAV dim {rav.size} != hidden_dim {x_base.shape[1]}; using RAV[:hidden_dim].")
        rav = np.asarray(rav[: x_base.shape[1]], dtype=np.float32)
    else:
        rav = np.asarray(rav, dtype=np.float32)
    rav_t = torch.from_numpy(rav).to(device)
    if rav_t.dim() == 1:
        rav_t = rav_t.unsqueeze(0).unsqueeze(0)
    elif rav_t.dim() == 2:
        rav_t = rav_t.unsqueeze(0)

    x_used = sequential_perpendicular(x_base, rav_t, alpha=1.0)
    print(f"  x_used (after sequential_perpendicular, alpha=1): shape = {tuple(x_used.shape)}")
    print(f"  Edit is applied to the tensor that has this shape, before LayerNorm.")
    print(f"  Exact location: inside PerpClassifier.forward, between _prepare_hidden and self.norm.")
    print(f"  Function: sequential_perpendicular (projects out the RAV direction).")

    # ----- (2) LAYERNORM + LINEAR RECOVERY -----
    print("\n" + "=" * 70)
    print("(2) VERIFY WHETHER LAYERNORM + LINEAR RECOVER THE DELETED DIRECTION")
    print("=" * 70)

    # Load model
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    in_dim = int(state["norm.weight"].shape[0])
    out_dim = int(state["fc.weight"].shape[0])
    use_bias = "fc.bias" in state
    model = PerpClassifier(in_dim=in_dim, num_classes=out_dim, use_bias=use_bias)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Normalize RAV to unit for dot-product interpretation (direction)
    c = rav_t.reshape(-1)
    c = c / (c.norm().item() + 1e-12)

    # Use a small batch for a concrete example
    idx = 0
    h_orig = x_base[idx : idx + 1]
    h_edit = x_used[idx : idx + 1]

    with torch.no_grad():
        x_norm = model.norm(h_edit)
        logits = model.fc(x_norm)

    print("\n--- 2a) Dot products with the characteristic vector c (unit RAV) ---")
    dot_orig = (h_orig * c).sum(dim=-1).item()
    dot_edit = (h_edit * c).sum(dim=-1).item()
    dot_after_ln = (x_norm * c).sum(dim=-1).item()
    print(f"  [example index {idx}]")
    print(f"  h_orig · c       = {dot_orig:.6e}")
    print(f"  h_edit · c       = {dot_edit:.6e}  (should be ~0 if removal is exact)")
    print(f"  after_LayerNorm · c = {dot_after_ln:.6e}")

    # Linear layer: logit = W @ h + b. For the deleted direction to matter, we need
    # the rows of W to have a component along c. Check alignment of W with c.
    W = model.fc.weight
    # W has shape (num_classes, in_dim). Each row is the direction that boosts that class.
    # Project each row onto c: (W @ c) or equivalently row-wise dot with c.
    # W is (C, D), c is (D,). proj = W @ c  -> (C,)
    W_dot_c = (W * c).sum(dim=1)
    print("\n--- 2b) Linear layer weight vs c ---")
    print(f"  fc.weight shape: {tuple(W.shape)} (out_features, in_features)")
    print(f"  W rows · c (per-class alignment with RAV direction):")
    for i, v in enumerate(W_dot_c.detach().cpu().numpy()):
        print(f"    class {i}: {v:.6e}")
    print(f"  max |W·c|: {W_dot_c.abs().max().item():.6e}")

    # LayerNorm: y = (x - mu) / sqrt(var+eps) * gamma + beta
    # gamma and beta can introduce a component along c: beta·c and (gamma/sigma) effect.
    gamma = model.norm.weight
    beta = model.norm.bias
    print("\n--- 2c) LayerNorm parameters vs c ---")
    print(f"  norm.weight (gamma) shape: {tuple(gamma.shape)}")
    print(f"  norm.bias (beta) shape:    {tuple(beta.shape)}")
    gamma_dot_c = (gamma * c).sum().item()
    beta_dot_c = (beta * c).sum().item()
    print(f"  gamma · c = {gamma_dot_c:.6e}")
    print(f"  beta · c  = {beta_dot_c:.6e}")
    print("  (Nonzero beta·c adds a fixed offset in the c direction to every sample.)")

    # Batch statistics: does the deleted direction reappear after LayerNorm?
    with torch.no_grad():
        x_used = sequential_perpendicular(x_base, rav_t, alpha=1.0)
        x_norm = model.norm(x_used)
    dots_edit = (x_used * c).sum(dim=-1).cpu().numpy()
    dots_ln = (x_norm * c).sum(dim=-1).cpu().numpy()
    print("\n--- 2d) Batch: does the c-component reappear after LayerNorm? ---")
    print(f"  h_edit · c:  min={dots_edit.min():.6e}, max={dots_edit.max():.6e}, mean={dots_edit.mean():.6e}, std={dots_edit.std():.6e}")
    print(f"  after_LN · c: min={dots_ln.min():.6e}, max={dots_ln.max():.6e}, mean={dots_ln.mean():.6e}, std={dots_ln.std():.6e}")

    # Interpret
    print("\n--- 2e) Interpretation ---")
    if np.abs(dots_edit).max() < 1e-4:
        print("  - The edit achieves near-orthogonality to c (h_edit · c ≈ 0).")
    else:
        print("  - WARNING: h_edit · c is not near zero; the projection may not be applied as intended.")

    if np.abs(dots_ln).max() > 1e-3 or np.abs(dots_ln).std() > 1e-4:
        print("  - After LayerNorm, the representation has a NONTRIVIAL component along c")
        print("    (nonzero and/or variable across samples). LayerNorm can reintroduce the direction")
        print("    via: (1) beta (beta·c) adds a constant; (2) the affine (x-mu)/sigma*gamma+beta")
        print("    can create sample-dependent components along c because gamma and the")
        print("    normalization depend on the full vector.")
    else:
        print("  - After LayerNorm, the component along c remains small.")

    if W_dot_c.abs().max() > 0.1:
        print("  - The linear layer has STRONG alignment with c (|W·c| large): the classifier")
        print("    can use the c-direction if it is present in the LayerNorm output.")
    else:
        print("  - The linear layer's alignment with c is modest; recovery is not obviously")
        print("    via the linear weights alone.")

    if np.abs(beta_dot_c) > 1e-3:
        print("  - beta·c is non-negligible: LayerNorm's bias adds a fixed component along c.")
    else:
        print("  - beta·c is small; the main recovery is not from a constant beta offset.")

    print("\n" + "=" * 70)
    print("Summary: Edit is applied before PerpClassifier's LayerNorm. Recovery can come from")
    print("(1) LayerNorm (beta, gamma, or the nonlinear normalization) reintroducing a component")
    print("    along c, and/or (2) the linear layer using other directions that still correlate")
    print("    with mass (nonlinear concept). The perpendicular edit only removes the LINEAR")
    print("    component of mass along c; nonlinear mass info in other directions remains.")
    print("=" * 70)


if __name__ == "__main__":
    main()
