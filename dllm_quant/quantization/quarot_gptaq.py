"""
QuaRot + GPTAQ Combined Quantizer
"""
import torch
import torch.nn as nn
from typing import List

from .base import BaseQuantizer
from .quarot import QuaRotApplier
from .gptaq import GPTAQQuantizer


class QuaRotGPTAQQuantizer(BaseQuantizer):
    def __init__(self, model, weight_bits=4, act_bits=16, group_size=128,
                 blocksize=128, percdamp=0.01, actorder=False,
                 static_groups=False, rotation_type="hadamard",
                 device="cuda", **kwargs):
        super().__init__(model, weight_bits, act_bits, group_size, device)
        self.rotation_type = rotation_type
        self.quarot = QuaRotApplier(model, rotation_type=rotation_type, device=device)
        self._gptaq_kwargs = dict(
            weight_bits=weight_bits, act_bits=act_bits, group_size=group_size,
            blocksize=blocksize, percdamp=percdamp, actorder=actorder,
            static_groups=static_groups, device=device,
        )
        self.gptaq: GPTAQQuantizer = None

    def calibrate(self, calibration_data: List[torch.Tensor]) -> None:
        print("=" * 60)
        print("[QuaRot+GPTAQ] Phase 1: Applying QuaRot rotation")
        print("=" * 60)
        self.quarot.apply()

        print("=" * 60)
        print("[QuaRot+GPTAQ] Phase 2: GPTAQ calibration on rotated model")
        print("=" * 60)
        self.gptaq = GPTAQQuantizer(self.model, **self._gptaq_kwargs)
        self.gptaq.calibrate(calibration_data)
        self._calibrated = True

    def quantize(self) -> nn.Module:
        if not self._calibrated:
            raise RuntimeError("Must calibrate first!")
        return self.gptaq.quantize()

    @property
    def method_name(self) -> str:
        return f"QuaRot({self.rotation_type})+GPTAQ"
