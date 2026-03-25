"""
dLLM Masked Decoding Loop

LLaDA의 generate.py와 호환되는 디코딩 루프.
model(input_ids) → CausalLMOutputWithPast → .logits 사용.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llada_utils import model_forward


@dataclass
class DecodingTrajectory:
    """디코딩 과정의 전체 trajectory 저장"""
    sequences: List[torch.Tensor] = field(default_factory=list)
    logits_history: List[torch.Tensor] = field(default_factory=list)
    unmask_indices: List[torch.Tensor] = field(default_factory=list)
    confidence_history: List[torch.Tensor] = field(default_factory=list)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Fast-dLLM/llada/generate.py 동일"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Fast-dLLM/llada/generate.py 동일"""
    device = mask_index.device
    total = mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num_transfer = base.unsqueeze(1).expand(-1, steps).clone()
    cols = torch.arange(steps, device=device).unsqueeze(0)
    num_transfer = num_transfer + (cols < rem.unsqueeze(1)).long()
    return num_transfer


@torch.no_grad()
def masked_decode_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    단일 디코딩 스텝.
    model(input_ids) → logits (LLaDAModelLM 호환)
    """
    logits = model_forward(model, x)

    if temperature > 0:
        sampled_logits = add_gumbel_noise(logits, temperature)
    else:
        sampled_logits = logits

    probs = F.softmax(logits.float(), dim=-1)
    predicted_ids = sampled_logits.argmax(dim=-1)

    return logits, probs, predicted_ids


@torch.no_grad()
def run_full_decoding(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> DecodingTrajectory:
    """
    전체 dLLM 디코딩 + trajectory 기록.
    Fast-dLLM/llada/generate.py와 동일한 로직.
    """
    trajectory = DecodingTrajectory()
    device = next(model.parameters()).device

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt.shape[1]] = prompt.to(device)
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps)

        for step in range(steps):
            logits, probs, predicted_ids = masked_decode_step(model, x, temperature)

            trajectory.sequences.append(x.clone().cpu())
            trajectory.logits_history.append(logits[:, block_start:block_end, :].cpu())

            block_probs = probs[:, block_start:block_end, :]
            confidence = block_probs.max(dim=-1).values
            trajectory.confidence_history.append(confidence.cpu())

            n_transfer = num_transfer[:, step]
            block_mask = (x[:, block_start:block_end] == mask_id)

            if remasking == "low_confidence":
                conf_masked = confidence.clone()
                conf_masked[~block_mask] = -float("inf")
                for b in range(x.shape[0]):
                    k = min(n_transfer[b].item(), block_mask[b].sum().item())
                    if k <= 0:
                        continue
                    _, topk_idx = conf_masked[b].topk(k)
                    x[b, block_start + topk_idx] = predicted_ids[b, block_start + topk_idx]
            else:
                for b in range(x.shape[0]):
                    masked_pos = block_mask[b].nonzero(as_tuple=True)[0]
                    k = min(n_transfer[b].item(), len(masked_pos))
                    if k <= 0:
                        continue
                    perm = torch.randperm(len(masked_pos), device=device)[:k]
                    x[b, block_start + masked_pos[perm]] = predicted_ids[b, block_start + masked_pos[perm]]

    trajectory.sequences.append(x.clone().cpu())
    return trajectory
