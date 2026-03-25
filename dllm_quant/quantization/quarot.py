"""
QuaRot Rotation Applier for LLaDA
"""
import torch
import torch.nn as nn
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llada_utils import (
    get_transformer_layers,
    get_embeddings,
    get_lm_head,
    get_attention_inputs,
    get_attention_output,
    get_mlp_inputs,
    get_mlp_output,
    fuse_layer_norms_llada,
)


def get_hadamard_matrix(size: int, device: str = "cuda") -> torch.Tensor:
    """2^n 크기 Hadamard 행렬 (Sylvester construction)."""
    if size == 1:
        return torch.tensor([[1.0]], device=device)
    assert size & (size - 1) == 0, f"Size must be power of 2, got {size}"
    half = get_hadamard_matrix(size // 2, device)
    H = torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0)
    return H / math.sqrt(2)


def get_random_orthogonal_matrix(size: int, device: str = "cuda") -> torch.Tensor:
    """QR decomposition 기반 random orthogonal matrix."""
    random_matrix = torch.randn(size, size, device=device, dtype=torch.float64)
    Q, R = torch.linalg.qr(random_matrix)
    Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.to(torch.float32)


class QuaRotApplier:
    """QuaRot rotation을 LLaDA 모델에 적용."""

    def __init__(self, model: nn.Module, rotation_type: str = "hadamard",
                 device: str = "cuda"):
        self.model = model
        self.rotation_type = rotation_type
        self.device = device
        self.hidden_size = model.config.d_model

        if rotation_type == "hadamard" and (self.hidden_size & (self.hidden_size - 1) == 0):
            self.R = get_hadamard_matrix(self.hidden_size, device)
        else:
            if rotation_type == "hadamard":
                print(f"[QuaRot] hidden_size={self.hidden_size} not power of 2, "
                      f"using random orthogonal")
            self.R = get_random_orthogonal_matrix(self.hidden_size, device)

    def apply(self) -> nn.Module:
        """QuaRot 적용: LN fusion → rotation."""
        print("[QuaRot] Step 1: Fusing LayerNorms...")
        fuse_layer_norms_llada(self.model)

        print("[QuaRot] Step 2: Applying rotation to weights...")
        self._apply_rotation_to_model()

        print("[QuaRot] Done.")
        return self.model

    def _apply_rotation_to_model(self):
        R = self.R.to(torch.float64)
        R_T = R.T

        # Embedding: W_emb = W_emb @ R^T
        for emb in get_embeddings(self.model):
            dtype = emb.weight.dtype
            emb.weight.data = (emb.weight.data.double() @ R_T).to(dtype)

        layers = get_transformer_layers(self.model)
        for layer in layers:
            # Attention input: W = W @ R
            for linear in get_attention_inputs(layer):
                self._rotate_linear_right(linear, R)
            # Attention output: W = R^T @ W
            self._rotate_linear_left(get_attention_output(layer), R_T)
            # MLP input: W = W @ R
            for linear in get_mlp_inputs(layer):
                self._rotate_linear_right(linear, R)
            # MLP output: W = R^T @ W
            self._rotate_linear_left(get_mlp_output(layer), R_T)

        # LM head: W = W @ R
        self._rotate_linear_right(get_lm_head(self.model), R)

    @staticmethod
    def _rotate_linear_right(linear: nn.Linear, R: torch.Tensor):
        dtype = linear.weight.dtype
        linear.weight.data = (linear.weight.data.double() @ R).to(dtype)

    @staticmethod
    def _rotate_linear_left(linear: nn.Linear, R_T: torch.Tensor):
        dtype = linear.weight.dtype
        linear.weight.data = (R_T @ linear.weight.data.double()).to(dtype)
        if linear.bias is not None:
            linear.bias.data = (R_T @ linear.bias.data.double()).to(dtype)
