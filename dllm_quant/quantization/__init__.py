from .base import BaseQuantizer
from .gptaq import GPTAQQuantizer
from .quarot import QuaRotApplier
from .quarot_gptaq import QuaRotGPTAQQuantizer

QUANTIZER_REGISTRY = {
    "gptaq": GPTAQQuantizer,
    "quarot+gptaq": QuaRotGPTAQQuantizer,
}


def get_quantizer(method: str, **kwargs):
    if method not in QUANTIZER_REGISTRY:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(QUANTIZER_REGISTRY.keys())}"
        )
    return QUANTIZER_REGISTRY[method](**kwargs)
