"""
Calibration Pipeline for dLLM

QDLM의 datautils.py 형식과 호환:
- get_wikitext2(nsamples, seed, seqlen, model) → (trainloader, testenc)
- trainloader = [(inp, tar), ...] 형식

dLLM 특화: multi-mask ratio 캘리브레이션 지원
"""
import torch
import numpy as np
import random
from typing import List, Tuple, Optional
from transformers import AutoTokenizer


def get_tokenizer(model_path: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        model_path, use_fast=False, legacy=False, trust_remote_code=True
    )


def get_wikitext2(nsamples: int, seed: int, seqlen: int, model_path: str):
    """
    QDLM의 datautils.get_wikitext2와 동일한 인터페이스.
    Returns: (trainloader, testenc)
        trainloader: [(inp, tar), ...] — inp: (1, seqlen), tar: (1, seqlen)
        testenc: tokenized test set
    """
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4(nsamples: int, seed: int, seqlen: int, model_path: str):
    """C4 데이터셋 로드. QDLM 호환."""
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None


def get_loaders(dataset: str, nsamples: int, seed: int, seqlen: int, model_path: str):
    """QDLM의 datautils.get_loaders 호환 래퍼."""
    if dataset == "wikitext2":
        return get_wikitext2(nsamples, seed, seqlen, model_path)
    elif dataset == "c4":
        return get_c4(nsamples, seed, seqlen, model_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ──────────────────────────────────────────────
# dLLM 특화: Multi-mask 캘리브레이션
# ──────────────────────────────────────────────

def apply_random_mask(
    input_ids: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int = 126336,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """input_ids에 랜덤 mask 적용."""
    if seed is not None:
        torch.manual_seed(seed)
    masked = input_ids.clone()
    mask = torch.rand_like(input_ids.float()) < mask_ratio
    masked[mask] = mask_token_id
    return masked


def apply_multimask_to_loader(
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]],
    mask_ratios: List[float],
    mask_token_id: int = 126336,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    QDLM 형식의 trainloader를 받아서 multi-mask 캘리브레이션 데이터 생성.
    각 (inp, tar)에 대해 여러 mask ratio 적용.

    Returns: List of masked input_ids (1, seqlen) tensors
    """
    calibration_data = []
    rng = np.random.RandomState(seed)

    for inp, tar in trainloader:
        for ratio in mask_ratios:
            sample_seed = rng.randint(0, 2**31)
            masked = apply_random_mask(inp, ratio, mask_token_id, seed=sample_seed)
            calibration_data.append(masked)

    print(
        f"[Calibration] Generated {len(calibration_data)} masked samples "
        f"({len(trainloader)} texts × {len(mask_ratios)} mask ratios)"
    )
    return calibration_data


def prepare_calibration(
    model_path: str,
    dataset: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 2048,
    mask_ratios: List[float] = [0.3, 0.5, 0.7],
    mask_token_id: int = 126336,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], AutoTokenizer]:
    """
    End-to-end 캘리브레이션 데이터 준비.
    Returns: (calibration_data, tokenizer)
    """
    tokenizer = get_tokenizer(model_path)

    trainloader, testenc = get_loaders(dataset, n_samples, seed, seq_len, model_path)

    calibration_data = apply_multimask_to_loader(
        trainloader, mask_ratios, mask_token_id, seed
    )

    return calibration_data, tokenizer
