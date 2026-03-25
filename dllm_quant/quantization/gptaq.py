"""
GPTAQ Quantizer for dLLM (LLaDA)

호환성 핵심:
- LLaDABlock.forward(x, attention_bias=None, layer_past=None, use_cache=False)
  → returns (x, cache) 튜플
- Embedding: model.model.transformer.wte(input_ids)
- calibration data: List[torch.Tensor] (masked input_ids)
"""
import math
import time
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

from .base import BaseQuantizer, SymmetricQuantizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llada_utils import (
    get_transformer_layers,
    get_sequential_groups,
    get_layer_io_device,
    layer_forward,
    embed_forward,
)


class GPTQLayer:
    """단일 Linear layer에 대한 GPTQ 양자화."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor = None):
        """Hessian 행렬에 배치 추가."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1,
                    actorder=False, static_groups=False, bits=4):
        """GPTQ block-wise 양자화. Returns: (quantized_weight, total_loss)"""
        W = self.layer.weight.data.clone().float()
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        quantizer = SymmetricQuantizer()
        quant_params = quantizer.find_params(W, bits=bits, group_size=groupsize)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and not static_groups:
                    if (i1 + i) % groupsize == 0:
                        quant_params = quantizer.find_params(
                            W[:, (i1 + i):(i1 + i + groupsize)],
                            bits=bits, group_size=-1,
                        )

                q = quantizer.quantize_weight(
                    w.unsqueeze(1), quant_params["scale"],
                    quant_params["bits"], quant_params["qmax"],
                ).squeeze(1)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        total_loss = torch.sum(Losses).item()
        return Q, total_loss


class GPTAQQuantizer(BaseQuantizer):
    """
    GPTAQ를 dLLM (LLaDA)에 적용.
    QDLM의 duquant() 함수와 동일한 layer-by-layer 패턴 사용.
    """

    def __init__(self, model, weight_bits=4, act_bits=16, group_size=128,
                 blocksize=128, percdamp=0.01, actorder=False,
                 static_groups=False, device="cuda", **kwargs):
        super().__init__(model, weight_bits, act_bits, group_size, device)
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.actorder = actorder
        self.static_groups = static_groups
        self._gptq_layers: Dict[int, Dict[str, GPTQLayer]] = {}
        self._layer_inputs: List[torch.Tensor] = []

    def calibrate(self, calibration_data: List[torch.Tensor]) -> None:
        """
        Layer-by-layer Hessian 수집.
        QDLM의 duquant() 패턴을 따름:
        1. embedding 통과 → 첫 layer 입력 수집
        2. 각 layer에서 hook으로 Hessian 수집
        3. layer forward → 다음 layer 입력으로 전달
        """
        print(f"[GPTAQ] Calibrating with {len(calibration_data)} samples...")
        self.model.eval()
        layers = get_transformer_layers(self.model)
        dev = self.device

        # Step 1: 모든 calibration data를 embedding 통과 → 첫 layer 입력 수집
        print("  Collecting first layer inputs...")
        self._layer_inputs = []
        for inp in calibration_data:
            with torch.no_grad():
                x = embed_forward(self.model, inp.to(dev))
                self._layer_inputs.append(x.cpu())

        # Step 2: Layer-by-layer 처리
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            layer = layer.to(dev)

            print(f"  Layer {layer_idx}/{len(layers)-1}: collecting Hessian & forwarding")

            # 이 layer의 GPTQ 인스턴스 생성
            sequential = get_sequential_groups(layer)
            gptq_dict = {}
            for name, linear in sequential.items():
                gptq_dict[name] = GPTQLayer(linear)
            self._gptq_layers[layer_idx] = gptq_dict

            # Hook 등록: 각 linear의 입력으로 Hessian 수집
            handles = []
            for name, linear in sequential.items():
                def make_hook(gptq_inst):
                    def hook_fn(module, inp, out):
                        gptq_inst.add_batch(inp[0].data)
                    return hook_fn
                h = linear.register_forward_hook(make_hook(gptq_dict[name]))
                handles.append(h)

            # Calibration data를 이 layer에 통과 + 다음 layer 입력 수집
            next_inputs = []
            for x in self._layer_inputs:
                with torch.no_grad():
                    x_dev = x.to(dev)
                    # LLaDABlock.forward(x, attention_bias, layer_past, use_cache)
                    out = layer_forward(layer, x_dev)
                    next_inputs.append(out.cpu())
                    del x_dev, out

            # Hook 제거
            for h in handles:
                h.remove()

            # 다음 layer로 전달
            self._layer_inputs = next_inputs

            # 메모리 관리: layer를 CPU로 (GPU에 한 layer씩만)
            layer.cpu()
            torch.cuda.empty_cache()

        self._calibrated = True
        print("[GPTAQ] Calibration done.")

    def quantize(self) -> nn.Module:
        """수집된 Hessian 기반 layer-wise GPTQ 양자화."""
        if not self._calibrated:
            raise RuntimeError("Must calibrate first!")

        print(f"[GPTAQ] Quantizing W{self.weight_bits}A{self.act_bits}...")
        layers = get_transformer_layers(self.model)
        total_loss = 0.0

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            layer = layer.to(self.device)

            gptq_dict = self._gptq_layers[layer_idx]
            sequential = get_sequential_groups(layer)

            for name, linear in sequential.items():
                gptq_inst = gptq_dict[name]
                Q, loss = gptq_inst.fasterquant(
                    blocksize=self.blocksize,
                    percdamp=self.percdamp,
                    groupsize=self.group_size,
                    actorder=self.actorder,
                    static_groups=self.static_groups,
                    bits=self.weight_bits,
                )
                linear.weight.data = Q.to(linear.weight.dtype)
                total_loss += loss
                print(f"  Layer {layer_idx} / {name}: loss = {loss:.4f}")

            del self._gptq_layers[layer_idx]
            layer.cpu()
            torch.cuda.empty_cache()

        print(f"[GPTAQ] Total quantization loss: {total_loss:.4f}")
        return self.model
