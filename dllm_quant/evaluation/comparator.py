"""
FP vs Quantized Model Comparator
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os

from .decoding import masked_decode_step, get_num_transfer_tokens
from .metrics import compute_step_metrics, compute_trajectory_metrics, StepMetrics, TrajectoryMetrics


@dataclass
class ComparisonResult:
    trajectory_metrics: TrajectoryMetrics = None
    final_seq_fp: torch.Tensor = None
    final_seq_q: torch.Tensor = None
    prompt: torch.Tensor = None
    config: dict = field(default_factory=dict)


class FPvsQuantComparator:
    """
    FP와 Quantized 모델을 step-wise로 비교.
    - Shared trajectory: FP 기준 unmask, 동일 입력으로 logit 비교
    - Independent trajectory: 각자 독립 unmask, error propagation 측정
    """

    def __init__(self, model_fp, model_q, mask_id=126336, device="cuda"):
        self.model_fp = model_fp
        self.model_q = model_q
        self.mask_id = mask_id
        self.device = device

    @torch.no_grad()
    def compare_shared_trajectory(
        self, prompt, steps=128, gen_length=128, block_length=128,
        temperature=0.0, remasking="low_confidence",
    ) -> ComparisonResult:
        """Shared trajectory: FP 기준 unmask, Quant logits 비교."""
        result = ComparisonResult(prompt=prompt.cpu())
        step_metrics_list = []

        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            self.mask_id, dtype=torch.long, device=self.device,
        )
        x[:, :prompt.shape[1]] = prompt.to(self.device)
        prompt_len = prompt.shape[1]
        num_blocks = gen_length // block_length

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = block_start + block_length
            block_mask = (x[:, block_start:block_end] == self.mask_id)
            num_transfer = get_num_transfer_tokens(block_mask, steps)

            for step in range(steps):
                # FP forward
                logits_fp, probs_fp, pred_fp = masked_decode_step(
                    self.model_fp, x, temperature
                )
                # Quant forward (동일 입력)
                logits_q, probs_q, pred_q = masked_decode_step(
                    self.model_q, x, temperature
                )

                block_mask_current = (x[:, block_start:block_end] == self.mask_id)
                metrics = compute_step_metrics(
                    logits_fp[:, block_start:block_end, :],
                    logits_q[:, block_start:block_end, :],
                    step=block_idx * steps + step,
                    mask=block_mask_current,
                )
                step_metrics_list.append(metrics)

                del logits_fp, logits_q, probs_q, pred_q
                torch.cuda.empty_cache()

                # FP 기준 unmask
                n_transfer = num_transfer[:, step]
                conf_fp = probs_fp[:, block_start:block_end, :].max(dim=-1).values
                current_block_mask = (x[:, block_start:block_end] == self.mask_id)

                self._unmask_step(
                    x, probs_fp, pred_fp,
                    block_start, block_end,
                    n_transfer, remasking,
                )

                del probs_fp, pred_fp
                torch.cuda.empty_cache()

        result.final_seq_fp = x.cpu()
        result.trajectory_metrics = compute_trajectory_metrics(
            step_metrics_list, x.cpu(), x.cpu(), prompt_len
        )
        return result

    @torch.no_grad()
    def compare_independent_trajectory(
        self, prompt, steps=128, gen_length=128, block_length=128,
        temperature=0.0, remasking="low_confidence",
    ) -> ComparisonResult:
        """Independent trajectory: FP/Quant 각자 독립 unmask."""
        result = ComparisonResult(prompt=prompt.cpu())
        step_metrics_list = []

        x_fp = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            self.mask_id, dtype=torch.long, device=self.device,
        )
        x_fp[:, :prompt.shape[1]] = prompt.to(self.device)
        x_q = x_fp.clone()
        prompt_len = prompt.shape[1]
        num_blocks = gen_length // block_length

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = block_start + block_length
            num_transfer_fp = get_num_transfer_tokens(
                (x_fp[:, block_start:block_end] == self.mask_id), steps
            )
            num_transfer_q = get_num_transfer_tokens(
                (x_q[:, block_start:block_end] == self.mask_id), steps
            )

            for step in range(steps):
                logits_fp, probs_fp, pred_fp = masked_decode_step(
                    self.model_fp, x_fp, temperature
                )
                logits_q, probs_q, pred_q = masked_decode_step(
                    self.model_q, x_q, temperature
                )

                common_mask = (
                    (x_fp[:, block_start:block_end] == self.mask_id) &
                    (x_q[:, block_start:block_end] == self.mask_id)
                )
                metrics = compute_step_metrics(
                    logits_fp[:, block_start:block_end, :],
                    logits_q[:, block_start:block_end, :],
                    step=block_idx * steps + step,
                    mask=common_mask,
                )
                step_metrics_list.append(metrics)

                self._unmask_step(x_fp, probs_fp, pred_fp, block_start, block_end,
                                  num_transfer_fp[:, step], remasking)
                self._unmask_step(x_q, probs_q, pred_q, block_start, block_end,
                                  num_transfer_q[:, step], remasking)

                del logits_fp, logits_q, probs_fp, probs_q, pred_fp, pred_q
                torch.cuda.empty_cache()

        result.final_seq_fp = x_fp.cpu()
        result.final_seq_q = x_q.cpu()
        result.trajectory_metrics = compute_trajectory_metrics(
            step_metrics_list, x_fp.cpu(), x_q.cpu(), prompt_len
        )
        return result

    def _unmask_step(self, x, probs, pred, block_start, block_end,
                     n_transfer, remasking):
        conf = probs[:, block_start:block_end, :].max(dim=-1).values
        current_mask = (x[:, block_start:block_end] == self.mask_id)

        if remasking == "low_confidence":
            conf_masked = conf.clone()
            conf_masked[~current_mask] = -float("inf")
            for b in range(x.shape[0]):
                k = min(n_transfer[b].item(), current_mask[b].sum().item())
                if k <= 0:
                    continue
                _, topk_idx = conf_masked[b].topk(k)
                x[b, block_start + topk_idx] = pred[b, block_start + topk_idx]
        else:
            for b in range(x.shape[0]):
                masked_pos = current_mask[b].nonzero(as_tuple=True)[0]
                k = min(n_transfer[b].item(), len(masked_pos))
                if k <= 0:
                    continue
                perm = torch.randperm(len(masked_pos), device=x.device)[:k]
                x[b, block_start + masked_pos[perm]] = pred[b, block_start + masked_pos[perm]]

    def save_result(self, result, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_dict = {
            "trajectory_metrics": {
                "avg_token_flip_rate": result.trajectory_metrics.avg_token_flip_rate,
                "avg_kl_divergence": result.trajectory_metrics.avg_kl_divergence,
                "final_sequence_match": result.trajectory_metrics.final_sequence_match,
                "trajectory_divergence_point": result.trajectory_metrics.trajectory_divergence_point,
                "error_acceleration": result.trajectory_metrics.error_acceleration,
                "per_step": [
                    {"step": s.step, "token_flip_rate": s.token_flip_rate,
                     "confidence_diff": s.confidence_diff, "kl_divergence": s.kl_divergence,
                     "top5_agreement": s.top5_agreement}
                    for s in result.trajectory_metrics.per_step
                ],
            },
            "config": result.config,
        }
        with open(path, "w") as f:
            json.dump(save_dict, f, indent=2)
