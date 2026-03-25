"""
dLLM Quantization Experiment Configuration
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path


@dataclass
class ModelConfig:
    """LLaDA 모델 경로 및 설정"""
    model_path: str = "models/LLaDA-8B-Instruct"
    mask_token_id: int = 126336
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    n_layers: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    max_seq_len: int = 4096
    dtype: str = "bfloat16"


@dataclass
class QuantConfig:
    """양자화 설정"""
    method: Literal["gptaq", "duquant", "quarot", "quarot+gptaq"] = "gptaq"
    weight_bits: int = 4
    act_bits: int = 16
    group_size: int = 128
    blocksize: int = 128
    percdamp: float = 0.01
    actorder: bool = False
    static_groups: bool = False
    rotate: bool = False
    rotation_type: Literal["hadamard", "random"] = "hadamard"
    fuse_layernorm: bool = True


@dataclass
class CalibrationConfig:
    """캘리브레이션 설정"""
    dataset: str = "wikitext2"
    n_samples: int = 128
    seq_len: int = 2048
    mask_ratios: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    seed: int = 42


@dataclass
class DecodingConfig:
    """dLLM 디코딩 설정"""
    steps: int = 128
    gen_length: int = 128
    block_length: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"


@dataclass
class EvalConfig:
    """평가 설정"""
    n_eval_samples: int = 50
    eval_tasks: List[str] = field(default_factory=lambda: ["generation", "trajectory"])
    compute_token_flip: bool = True
    compute_confidence_diff: bool = True
    compute_kl_div: bool = True
    compute_trajectory_div: bool = True
    mask_ratios_to_eval: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])


@dataclass
class ExperimentConfig:
    """전체 실험 설정"""
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    output_dir: str = "results"
    experiment_name: str = "default"
    device: str = "cuda"
    seed: int = 42
    log_to_wandb: bool = False

    def describe(self) -> str:
        return (
            f"{self.quant.method}_W{self.quant.weight_bits}"
            f"A{self.quant.act_bits}_g{self.quant.group_size}"
        )
