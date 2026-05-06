"""PerpClassifier: LayerNorm + Linear head with optional RAV projection."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projections import project_rav, _prepare_hidden

logger = logging.getLogger(__name__)


def load_pretrained_part_weights(
    checkpoint_path: str,
    in_dim: int,
    num_classes: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load LayerNorm and final Linear weights from Particle Transformer checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    )

    extracted = {}
    norm_weight_key = norm_bias_key = None
    for key in state_dict:
        if key.endswith("norm.weight"):
            norm_weight_key = key
        elif key.endswith("norm.bias"):
            norm_bias_key = key

    if norm_weight_key is None:
        raise KeyError(f"Could not find norm.weight in {checkpoint_path}")
    if norm_bias_key is None:
        raise KeyError(f"Could not find norm.bias in {checkpoint_path}")

    extracted["norm.weight"] = state_dict[norm_weight_key]
    extracted["norm.bias"] = state_dict[norm_bias_key]

    fc_weight_keys = [
        k for k in state_dict
        if "fc" in k and k.endswith(".weight") and state_dict[k].ndim == 2
    ]
    if not fc_weight_keys:
        raise KeyError(f"Could not find FC Linear weights in {checkpoint_path}")

    final_weight_key = None
    for key in fc_weight_keys:
        if state_dict[key].shape[0] == num_classes:
            final_weight_key = key
            break
    if final_weight_key is None:
        final_weight_key = fc_weight_keys[-1]

    extracted["fc.weight"] = state_dict[final_weight_key]
    bias_key = final_weight_key.replace(".weight", ".bias")
    if bias_key in state_dict:
        extracted["fc.bias"] = state_dict[bias_key]

    return extracted


class PerpClassifier(nn.Module):
    """LayerNorm + Linear classifier with optional RAV projection in forward."""

    def __init__(
        self,
        in_dim: int = 128,
        num_classes: int = 10,
        for_inference: bool = False,
        use_bias: bool = True,
        use_pretrained_norm_fc: bool = False,
        pretrained_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        input_is_post_ln: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_classes, bias=use_bias)
        self.for_inference = for_inference
        self.use_pretrained_norm_fc = use_pretrained_norm_fc
        self.input_is_post_ln = input_is_post_ln
        self._perp_orders_logged = False

        if use_pretrained_norm_fc and pretrained_state_dict:
            self._load_pretrained_weights(pretrained_state_dict)

    def _load_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if "norm.weight" in state_dict:
            self.norm.weight.data.copy_(state_dict["norm.weight"])
        if "norm.bias" in state_dict:
            self.norm.bias.data.copy_(state_dict["norm.bias"])
        if "fc.weight" in state_dict:
            src_w = state_dict["fc.weight"]
            dst_w = self.fc.weight.data
            if src_w.shape == dst_w.shape:
                dst_w.copy_(src_w)
            elif src_w.shape[0] >= dst_w.shape[0] and src_w.shape[1] == dst_w.shape[1]:
                # Pretrained has more classes (e.g. 10); slice to first num_classes (e.g. QCD=0, Hbb=1)
                dst_w.copy_(src_w[: dst_w.shape[0]])
                logger.info(
                    "Sliced pretrained fc.weight from %s to %s (HbbvsQCD uses classes 0,1)",
                    tuple(src_w.shape),
                    tuple(dst_w.shape),
                )
            else:
                logger.warning(
                    "Pretrained fc.weight shape %s does not match model %s; fc left randomly initialized",
                    tuple(src_w.shape),
                    tuple(dst_w.shape),
                )
            if "fc.bias" in state_dict:
                src_b = state_dict["fc.bias"]
                if self.fc.bias is None:
                    self.fc = nn.Linear(
                        self.fc.in_features, self.fc.out_features, bias=True
                    )
                dst_b = self.fc.bias.data
                if src_b.shape[0] >= dst_b.shape[0]:
                    dst_b.copy_(src_b[: dst_b.shape[0]])
                else:
                    dst_b.zero_()
            elif self.fc.bias is not None:
                self.fc.bias.data.zero_()

    def forward(
        self,
        cls_tokens: torch.Tensor,
        rav_tensors: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        alpha: float = 1.0,
        rav_mu: Optional[torch.Tensor] = None,
        rav_std: Optional[torch.Tensor] = None,
    ):
        x_base = _prepare_hidden(cls_tokens)

        if rav_tensors is not None:
            x_used = project_rav(x_base, rav_tensors, alpha=alpha, mu=rav_mu, std=rav_std)
        else:
            x_used = x_base

        # Skip norm if input is already post-LN (e.g. cls_tokens_ln); fc expects LN output
        x_norm = x_used if self.input_is_post_ln else self.norm(x_used)
        hiddens = [x_norm.unsqueeze(0)]
        logits = self.fc(x_norm)

        if self.for_inference:
            probs = F.softmax(logits, dim=1) if logits.shape[1] > 1 else logits
            return (probs, hiddens) if return_hidden else probs
        return (logits, hiddens) if return_hidden else logits
