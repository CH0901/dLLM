"""
QuaRot Rotation Applier for LLaDA (No LN Fusion)

LLaDA에서는 LN fusion 없이 rotation만 적용.
RMSNorm을 그대로 유지하여 수치 안정성 확보.

Rotation 적용 위치:
  residual stream에 R을 삽입:
  - embedding 출력: x' = x @ R^T
  - 각 layer 입력 (norm 후): W_in = W_in @ R  (q,k,v,ff_proj,up_proj)
  - 각 layer 출력: W_out = R^T @ W_out         (attn_out, ff_out)
  - lm_head: W = W @ R

  이렇게 하면 norm(x @ R^T)에서 RMSNorm은 스케일 불변이므로:
  norm(x @ R^T) = norm(x) @ R^T  (RMSNorm은 element-wise scale이라 정확히는 아니지만)

  실제로는:
  - norm 전: residual stream에 R^T가 곱해진 상태
  - norm: RMSNorm(x @ R^T) — weight가 있으므로 완전히 투명하진 않음
  - norm 후 linear: (W @ R) @ norm_output

  따라서 정확한 등가 변환을 위해:
  - norm.weight도 R로 permute해야 하지만, RMSNorm의 weight는 element-wise이므로
    rotation과 commute하지 않음
  
  해결: norm.weight를 R과 함께 linear에 흡수 (fusion) 후 norm.weight = 1로 설정
  이것이 표준 QuaRot 방식이지만, norm의 variance 계산은 유지됨.
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
    get_input_layernorm,
    get_post_attention_layernorm,
    get_pre_head_layernorm,
    fuse_ln_linear,
)


def get_hadamard_matrix(size: int, device: str = "cpu") -> torch.Tensor:
    if size == 1:
        return torch.tensor([[1.0]], device=device)
    assert size & (size - 1) == 0, f"Size must be power of 2, got {size}"
    half = get_hadamard_matrix(size // 2, device)
    H = torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0)
    return H / math.sqrt(2)


def get_random_orthogonal_matrix(size: int, device: str = "cpu") -> torch.Tensor:
    random_matrix = torch.randn(size, size, device=device, dtype=torch.float64)
    Q, R = torch.linalg.qr(random_matrix)
    Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.to(torch.float32)


class QuaRotApplier:
    def __init__(self, model: nn.Module, rotation_type: str = "hadamard",
                 device: str = "cuda"):
        self.model = model
        self.rotation_type = rotation_type
        self.device = device
        self.hidden_size = model.config.d_model

        if rotation_type == "hadamard" and (self.hidden_size & (self.hidden_size - 1) == 0):
            self.R = get_hadamard_matrix(self.hidden_size, "cpu")
        else:
            if rotation_type == "hadamard":
                print(f"[QuaRot] hidden_size={self.hidden_size} not power of 2, "
                      f"using random orthogonal")
            self.R = get_random_orthogonal_matrix(self.hidden_size, "cpu")

    def apply(self) -> nn.Module:
        print("[QuaRot] Step 1: Fusing LN weights into linears (keeping norm op)...")
        self._fuse_ln_weights()

        print("[QuaRot] Step 2: Applying rotation to weights...")
        self._apply_rotation_to_model()

        print("[QuaRot] Done.")
        return self.model

    def _fuse_ln_weights(self):
        """
        LN weight를 인접 linear에 흡수하되, norm 연산 자체는 유지.
        norm.weight = 1로 설정 (identity scale), norm의 variance 계산은 그대로.
        """
        layers = get_transformer_layers(self.model)

        for layer in layers:
            # attn_norm.weight → q, k, v에 흡수
            attn_norm = get_input_layernorm(layer)
            fuse_ln_linear(attn_norm, get_attention_inputs(layer))
            # weight를 1로 리셋 (norm 연산은 유지)
            attn_norm.weight.data.fill_(1.0)

            # ff_norm.weight → ff_proj, up_proj에 흡수
            ff_norm = get_post_attention_layernorm(layer)
            fuse_ln_linear(ff_norm, get_mlp_inputs(layer))
            ff_norm.weight.data.fill_(1.0)

        # final norm → lm_head에 흡수
        ln_f = get_pre_head_layernorm(self.model)
        fuse_ln_linear(ln_f, [get_lm_head(self.model)])
        ln_f.weight.data.fill_(1.0)

    def _apply_rotation_to_model(self):
        R = self.R.to(torch.float64)
        R_T = R.T

        # 1. Embedding: W_emb = W_emb @ R^T
        for emb in get_embeddings(self.model):
            dtype = emb.weight.dtype
            emb.weight.data = (emb.weight.data.double() @ R_T).to(dtype)

        layers = get_transformer_layers(self.model)
        for layer in layers:
            # 2. Attention input: W = W @ R
            for linear in get_attention_inputs(layer):
                self._rotate_linear_right(linear, R)
            # 3. Attention output: W = R^T @ W
            self._rotate_linear_left(get_attention_output(layer), R_T)
            # 4. MLP input: W = W @ R
            for linear in get_mlp_inputs(layer):
                self._rotate_linear_right(linear, R)
            # 5. MLP output: W = R^T @ W
            self._rotate_linear_left(get_mlp_output(layer), R_T)

        # 6. LM head: W = W @ R
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