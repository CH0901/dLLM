"""
LLaDA Model Utilities Adapter

LLaDA 구조 (실제 코드에서 확인):
    model.model.transformer.wte          → embedding
    model.model.transformer.blocks[i]    → transformer layers (LLaDABlock)
        .attn_norm                       → input layernorm (RMSNorm)
        .ff_norm                         → post-attention layernorm (RMSNorm)
        .q_proj, .k_proj, .v_proj        → attention projections
        .attn_out                        → attention output
        .ff_proj                         → gate projection (MLP)
        .up_proj                         → up projection (MLP)
        .ff_out                          → down projection (MLP)
        .act                             → activation function
    model.model.transformer.ln_f         → final layernorm
    model.model.transformer.ff_out       → lm_head

LLaDABlock.forward 시그니처:
    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False, **kwargs)
    → returns (x, cache) 튜플

LLaDAModelLM.forward:
    def forward(self, input_ids, ...) → CausalLMOutputWithPast (logits 등)
"""
import torch
import torch.nn as nn
import typing
from collections import OrderedDict

LLADA_MODEL = "llada"


# ──────────────────────────────────────────────
# Structure accessors
# ──────────────────────────────────────────────

def get_model_type(model) -> str:
    class_name = model.__class__.__name__.lower()
    if "llada" in class_name:
        return LLADA_MODEL
    raise ValueError(f"Unsupported model type: {model.__class__.__name__}")


def get_transformer(model):
    """model.model.transformer 반환"""
    return model.model.transformer


def get_transformer_layers(model, **kwargs):
    """transformer blocks 리스트 반환"""
    return get_transformer(model).blocks


def get_embeddings(model, **kwargs):
    """embedding layer 리스트 반환"""
    return [get_transformer(model).wte]


def get_pre_head_layernorm(model, **kwargs):
    """final layernorm 반환"""
    return get_transformer(model).ln_f


def get_lm_head(model, **kwargs):
    """lm_head (ff_out) 반환"""
    return get_transformer(model).ff_out


# ──────────────────────────────────────────────
# Layer-level accessors
# ──────────────────────────────────────────────

def get_attention_inputs(layer):
    """attention 입력 linear layers (LN fusion 대상)"""
    return [layer.q_proj, layer.k_proj, layer.v_proj]


def get_attention_output(layer):
    return layer.attn_out


def get_mlp_inputs(layer):
    """MLP 입력 linear layers (LN fusion 대상)"""
    return [layer.ff_proj, layer.up_proj]


def get_mlp_output(layer):
    return layer.ff_out


def get_input_layernorm(layer):
    return layer.attn_norm


def get_post_attention_layernorm(layer):
    return layer.ff_norm


# ──────────────────────────────────────────────
# Layer forward wrapper (호환성 핵심)
# ──────────────────────────────────────────────

def layer_forward(layer, x, **kwargs):
    """
    LLaDABlock.forward를 안전하게 호출.
    forward(x, attention_bias=None, layer_past=None, use_cache=False)
    → returns (output, cache) 튜플에서 output만 반환.
    """
    out = layer(x, attention_bias=kwargs.get("attention_bias", None),
                layer_past=kwargs.get("layer_past", None),
                use_cache=kwargs.get("use_cache", False))
    if isinstance(out, tuple):
        return out[0]
    return out


# ──────────────────────────────────────────────
# Embedding forward
# ──────────────────────────────────────────────

def embed_forward(model, input_ids):
    """
    input_ids → embedding output (첫 layer 입력).
    LLaDA: model.model.transformer.wte(input_ids)
    """
    return get_transformer(model).wte(input_ids)


# ──────────────────────────────────────────────
# Full model forward (logits 반환)
# ──────────────────────────────────────────────

def model_forward(model, input_ids):
    """
    전체 모델 forward → logits 반환.
    LLaDAModelLM.forward(input_ids) → CausalLMOutputWithPast
    """
    output = model(input_ids)
    return output.logits


# ──────────────────────────────────────────────
# LayerNorm fusion utilities
# ──────────────────────────────────────────────

def fuse_ln_linear(
    layernorm: nn.Module,
    linear_layers: typing.Iterable[nn.Linear],
) -> None:
    """LayerNorm의 weight를 인접 linear에 흡수."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = (
                linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            ).to(linear_dtype)


class RMSN(nn.Module):
    """Identity RMSNorm replacement (after fusion)"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x


def fuse_layer_norms_llada(model):
    """LLaDA 모델의 모든 LayerNorm을 인접 linear에 fusion."""
    # 1. Embedding mean centering
    for W in get_embeddings(model):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    # 2. Layer-wise LN fusion
    layers = get_transformer_layers(model)
    for layer in layers:
        fuse_ln_linear(get_input_layernorm(layer), get_attention_inputs(layer))
        fuse_ln_linear(get_post_attention_layernorm(layer), get_mlp_inputs(layer))

    # 3. Final LN → lm_head fusion
    fuse_ln_linear(get_pre_head_layernorm(model), [get_lm_head(model)])

    # 4. Replace all RMSNorm with identity
    replace_rms_norms(model)


def replace_rms_norms(model):
    """모든 RMSNorm을 identity로 교체"""
    transformer = get_transformer(model)
    hidden_size = model.config.d_model

    transformer.ln_f = RMSN(hidden_size)
    for layer in transformer.blocks:
        layer.attn_norm = RMSN(hidden_size)
        layer.ff_norm = RMSN(hidden_size)


# ──────────────────────────────────────────────
# Linear layer enumeration (for GPTQ)
# ──────────────────────────────────────────────

def get_sequential_groups(layer) -> OrderedDict:
    """GPTQ layer-wise 양자화용 linear group 매핑."""
    return OrderedDict([
        ("q_proj", layer.q_proj),
        ("k_proj", layer.k_proj),
        ("v_proj", layer.v_proj),
        ("attn_out", layer.attn_out),
        ("ff_proj", layer.ff_proj),
        ("up_proj", layer.up_proj),
        ("ff_out", layer.ff_out),
    ])


def get_layer_io_device(layer):
    return next(layer.parameters()).device


def move_layer_to_device(layer, device):
    layer.to(device)
    return layer
