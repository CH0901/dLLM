"""
dLLM Quantization Metrics
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class StepMetrics:
    step: int
    token_flip_rate: float = 0.0
    confidence_diff: float = 0.0
    kl_divergence: float = 0.0
    reverse_kl: float = 0.0
    top5_agreement: float = 0.0
    max_logit_diff: float = 0.0


@dataclass
class TrajectoryMetrics:
    per_step: List[StepMetrics] = field(default_factory=list)
    avg_token_flip_rate: float = 0.0
    avg_kl_divergence: float = 0.0
    final_sequence_match: float = 0.0
    trajectory_divergence_point: int = -1
    cumulative_flip_rates: List[float] = field(default_factory=list)
    error_acceleration: float = 0.0


def compute_token_flip_rate(logits_fp, logits_q, mask=None):
    pred_fp = logits_fp.argmax(dim=-1)
    pred_q = logits_q.argmax(dim=-1)
    if mask is not None:
        flips = ((pred_fp != pred_q) & mask).float().sum()
        total = mask.float().sum()
    else:
        flips = (pred_fp != pred_q).float().sum()
        total = torch.tensor(pred_fp.numel(), dtype=torch.float)
    return (flips / total.clamp(min=1)).item()


def compute_confidence_diff(logits_fp, logits_q, mask=None):
    conf_fp = F.softmax(logits_fp.float(), dim=-1).max(dim=-1).values
    conf_q = F.softmax(logits_q.float(), dim=-1).max(dim=-1).values
    diff = (conf_fp - conf_q).abs()
    if mask is not None:
        return (diff * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
    return diff.mean().item()


def compute_kl_divergence(logits_fp, logits_q, mask=None):
    log_p = F.log_softmax(logits_fp.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(dim=-1)
    if mask is not None:
        return (kl * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
    return kl.mean().item()


def compute_top_k_agreement(logits_fp, logits_q, k=5, mask=None):
    topk_fp = logits_fp.topk(k, dim=-1).indices
    topk_q = logits_q.topk(k, dim=-1).indices
    agreements = []
    for i in range(k):
        match = (topk_fp == topk_q[:, :, i:i+1]).any(dim=-1)
        agreements.append(match)
    agreement = torch.stack(agreements, dim=-1).float().mean(dim=-1)
    if mask is not None:
        return (agreement * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
    return agreement.mean().item()


def compute_step_metrics(logits_fp, logits_q, step, mask=None):
    return StepMetrics(
        step=step,
        token_flip_rate=compute_token_flip_rate(logits_fp, logits_q, mask),
        confidence_diff=compute_confidence_diff(logits_fp, logits_q, mask),
        kl_divergence=compute_kl_divergence(logits_fp, logits_q, mask),
        reverse_kl=compute_kl_divergence(logits_q, logits_fp, mask),
        top5_agreement=compute_top_k_agreement(logits_fp, logits_q, k=5, mask=mask),
        max_logit_diff=(logits_fp.max(dim=-1).values - logits_q.max(dim=-1).values).abs().mean().item(),
    )


def compute_trajectory_metrics(steps_metrics, final_seq_fp, final_seq_q, prompt_len):
    traj = TrajectoryMetrics(per_step=steps_metrics)
    if steps_metrics:
        traj.avg_token_flip_rate = np.mean([s.token_flip_rate for s in steps_metrics])
        traj.avg_kl_divergence = np.mean([s.kl_divergence for s in steps_metrics])
    gen_fp = final_seq_fp[:, prompt_len:]
    gen_q = final_seq_q[:, prompt_len:]
    traj.final_sequence_match = (gen_fp == gen_q).float().mean().item()
    flip_rates = [s.token_flip_rate for s in steps_metrics]
    traj.cumulative_flip_rates = list(np.cumsum(flip_rates))
    for i, rate in enumerate(flip_rates):
        if rate > 0.01:
            traj.trajectory_divergence_point = i
            break
    if len(flip_rates) >= 4:
        mid = len(flip_rates) // 2
        first_half = np.mean(flip_rates[:mid]) if flip_rates[:mid] else 0
        second_half = np.mean(flip_rates[mid:]) if flip_rates[mid:] else 0
        traj.error_acceleration = second_half / max(first_half, 1e-10)
    return traj


def compute_all_metrics(logits_fp, logits_q, mask=None):
    return {
        "token_flip_rate": compute_token_flip_rate(logits_fp, logits_q, mask),
        "confidence_diff": compute_confidence_diff(logits_fp, logits_q, mask),
        "kl_divergence": compute_kl_divergence(logits_fp, logits_q, mask),
        "top5_agreement": compute_top_k_agreement(logits_fp, logits_q, k=5, mask=mask),
    }
