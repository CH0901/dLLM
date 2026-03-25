"""
Base Quantizer - 모든 양자화 방법의 공통 인터페이스
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseQuantizer(ABC):
    def __init__(self, model: nn.Module, weight_bits: int = 4, act_bits: int = 16,
                 group_size: int = 128, device: str = "cuda"):
        self.model = model
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.group_size = group_size
        self.device = device
        self._calibrated = False

    @abstractmethod
    def calibrate(self, calibration_data: list) -> None:
        ...

    @abstractmethod
    def quantize(self) -> nn.Module:
        ...

    def get_quantized_model(self) -> nn.Module:
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before quantize()")
        return self.quantize()

    @property
    def method_name(self) -> str:
        return self.__class__.__name__


class SymmetricQuantizer:
    """대칭 균일 양자화 유틸리티 (per-channel / per-group)"""

    @staticmethod
    def find_params(weight: torch.Tensor, bits: int = 4, group_size: int = -1,
                    per_channel: bool = True) -> Dict[str, torch.Tensor]:
        if group_size > 0:
            assert weight.shape[-1] % group_size == 0
            weight = weight.reshape(-1, group_size)

        if per_channel or group_size > 0:
            max_val = weight.abs().amax(dim=-1, keepdim=True)
        else:
            max_val = weight.abs().max()

        qmax = (1 << (bits - 1)) - 1
        scale = max_val / qmax
        scale = scale.clamp(min=1e-10)
        return {"scale": scale, "bits": bits, "qmax": qmax}

    @staticmethod
    def quantize_weight(weight: torch.Tensor, scale: torch.Tensor, bits: int,
                        qmax: int, group_size: int = -1) -> torch.Tensor:
        orig_shape = weight.shape
        if group_size > 0:
            weight = weight.reshape(-1, group_size)
        q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
        w_hat = q * scale
        if group_size > 0:
            w_hat = w_hat.reshape(orig_shape)
        return w_hat
