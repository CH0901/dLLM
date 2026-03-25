"""
dLLM Quantization Experiment Runner

사용법:
    # dLLM/ 디렉토리에서 실행
    cd dLLM
    python -m dllm_quant.run_experiment --method gptaq --weight_bits 4 --act_bits 16

    # 또는 dllm_quant/ 안에서 직접:
    cd dLLM/dllm_quant
    python run_experiment.py --method gptaq --weight_bits 4 --act_bits 16
"""
import os
import sys
import json
import time
import argparse
import torch
import numpy as np

# 프로젝트 루트를 path에 추가 (dLLM/ 디렉토리)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import ExperimentConfig, ModelConfig, QuantConfig, CalibrationConfig, DecodingConfig
from calibration import prepare_calibration
from quantization import get_quantizer
from evaluation.comparator import FPvsQuantComparator


def load_llada_model(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """
    LLaDA 모델 로드.
    QDLM의 LMClass와 동일한 방식: CPU에 로드 → GPU로 이동.
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)

    print(f"[Loader] Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, legacy=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="cpu",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Loader] Loaded. Parameters: {n_params:.2f}B")
    return model, tokenizer


def save_quantized_model(model, tokenizer, save_path: str):
    """양자화된 모델 저장 (나중에 별도 eval 스크립트에서 로드용)."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Saver] Model saved to {save_path}")


def run_experiment(
    method: str = "gptaq",
    weight_bits: int = 4,
    act_bits: int = 16,
    model_path: str = "models/LLaDA-8B-Instruct",
    n_cal_samples: int = 128,
    n_eval_samples: int = 10,
    output_dir: str = "results",
    device: str = "cuda",
    seed: int = 42,
    save_model: bool = True,
    skip_eval: bool = False,
):
    """
    실험 메인 함수.

    Phase 1: 양자화 → 모델 저장
    Phase 2 (optional): FP vs Quant 분석 메트릭 수집
    Phase 3 (별도): GSM8K, HumanEval 등 task 평가 (eval_benchmarks.py)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = ExperimentConfig(
        model=ModelConfig(model_path=model_path),
        quant=QuantConfig(
            method=method, weight_bits=weight_bits, act_bits=act_bits,
            rotate=(method in ["quarot", "quarot+gptaq"]),
        ),
        calibration=CalibrationConfig(n_samples=n_cal_samples),
        output_dir=output_dir,
        experiment_name=f"{method}_W{weight_bits}A{act_bits}",
        device=device, seed=seed,
    )

    exp_dir = os.path.join(output_dir, config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 설정 저장
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({
            "method": method, "weight_bits": weight_bits, "act_bits": act_bits,
            "n_cal_samples": n_cal_samples, "n_eval_samples": n_eval_samples,
            "model_path": model_path, "seed": seed,
        }, f, indent=2)

    print("=" * 70)
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Method: {method} | W{weight_bits}A{act_bits}")
    print(f"  Model: {model_path}")
    print("=" * 70)

    # ── Step 1: 양자화용 모델 로드 ──
    print("\n[Step 1] Loading model for quantization...")
    t0 = time.time()
    model_q, tokenizer = load_llada_model(model_path, device="cpu")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Step 2: 캘리브레이션 ──
    print("\n[Step 2] Preparing calibration data...")
    t0 = time.time()
    calibration_data, _ = prepare_calibration(
        model_path=model_path,
        dataset=config.calibration.dataset,
        n_samples=config.calibration.n_samples,
        seq_len=config.calibration.seq_len,
        mask_ratios=config.calibration.mask_ratios,
        mask_token_id=config.model.mask_token_id,
        seed=seed,
    )
    print(f"  Prepared {len(calibration_data)} samples in {time.time()-t0:.1f}s")

    # ── Step 3: 양자화 ──
    print(f"\n[Step 3] Quantizing with {method}...")
    t0 = time.time()

    # 양자화 시 GPU 사용
    quantizer = get_quantizer(
        method=method, model=model_q,
        weight_bits=weight_bits, act_bits=act_bits,
        group_size=config.quant.group_size,
        blocksize=config.quant.blocksize,
        percdamp=config.quant.percdamp,
        device=device,
    )
    quantizer.calibrate(calibration_data)
    model_q = quantizer.quantize()
    quant_time = time.time() - t0
    print(f"  Quantized in {quant_time:.1f}s")

    del calibration_data
    torch.cuda.empty_cache()

    # ── Step 4: 양자화 모델 저장 ──
    if save_model:
        save_path = os.path.join(exp_dir, "quantized_model")
        print(f"\n[Step 4] Saving quantized model to {save_path}...")
        save_quantized_model(model_q, tokenizer, save_path)

    # ── Step 5: FP vs Quant 분석 (optional) ──
    if not skip_eval:
        print(f"\n[Step 5] FP vs Quant analysis (n={n_eval_samples})...")

        # FP 모델 로드
        model_fp, _ = load_llada_model(model_path, device="cpu")
        model_fp = model_fp.to(device)
        model_q = model_q.to(device)

        comparator = FPvsQuantComparator(
            model_fp, model_q,
            mask_id=config.model.mask_token_id,
            device=device,
        )

        eval_prompts = _generate_eval_prompts(tokenizer, n_eval_samples)
        all_results = []

        for i, prompt in enumerate(eval_prompts):
            print(f"\n  Sample {i+1}/{n_eval_samples}")
            result = comparator.compare_shared_trajectory(
                prompt=prompt,
                steps=config.decoding.steps,
                gen_length=config.decoding.gen_length,
                block_length=config.decoding.block_length,
                temperature=config.decoding.temperature,
                remasking=config.decoding.remasking,
            )
            result.config = {"sample_idx": i, "mode": "shared"}
            all_results.append(result)

            tm = result.trajectory_metrics
            print(f"    Flip rate: {tm.avg_token_flip_rate:.4f}  "
                  f"KL: {tm.avg_kl_divergence:.4f}  "
                  f"Match: {tm.final_sequence_match:.4f}")

            comparator.save_result(result, os.path.join(exp_dir, f"sample_{i:03d}.json"))

        # 집계
        agg = _aggregate_results(all_results)
        agg["quant_time_sec"] = quant_time
        agg["config"] = {"method": method, "weight_bits": weight_bits, "act_bits": act_bits}

        with open(os.path.join(exp_dir, "summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

        _print_summary(method, weight_bits, act_bits, agg, quant_time)

        # FP 모델 해제
        del model_fp
        torch.cuda.empty_cache()

    else:
        print("\n[Step 5] Skipped (--skip_eval)")
        agg = {"quant_time_sec": quant_time}

    print(f"\n[Done] Results saved to {exp_dir}/")
    print(f"  양자화 모델: {exp_dir}/quantized_model/")
    print(f"  이 모델로 GSM8K/HumanEval 평가:")
    print(f"    python eval_benchmarks.py --model_path {exp_dir}/quantized_model")

    return agg


def _generate_eval_prompts(tokenizer, n_samples):
    prompts_text = [
        "The theory of general relativity describes",
        "In recent years, machine learning has made advances in",
        "Climate change is one of the most pressing issues because",
        "The development of quantum computing promises to",
        "Neural networks have become the foundation of",
        "The human genome project was completed in 2003 and",
        "Blockchain technology was originally designed for",
        "The study of dark matter suggests that",
        "Renewable energy sources such as solar power have",
        "Natural language processing has evolved from",
    ]
    while len(prompts_text) < n_samples:
        prompts_text.extend(prompts_text)
    prompts_text = prompts_text[:n_samples]
    return [tokenizer(t, return_tensors="pt").input_ids for t in prompts_text]


def _aggregate_results(results):
    flip_rates = [r.trajectory_metrics.avg_token_flip_rate for r in results]
    kl_divs = [r.trajectory_metrics.avg_kl_divergence for r in results]
    matches = [r.trajectory_metrics.final_sequence_match for r in results]
    return {
        "avg_token_flip_rate": float(np.mean(flip_rates)),
        "std_token_flip_rate": float(np.std(flip_rates)),
        "avg_kl_divergence": float(np.mean(kl_divs)),
        "avg_final_match": float(np.mean(matches)),
        "n_samples": len(results),
    }


def _print_summary(method, w_bits, a_bits, agg, quant_time):
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Method:         {method}")
    print(f"  Bits:           W{w_bits}A{a_bits}")
    print(f"  Avg Flip Rate:  {agg.get('avg_token_flip_rate', 0):.4f}")
    print(f"  Avg KL Div:     {agg.get('avg_kl_divergence', 0):.4f}")
    print(f"  Avg Match:      {agg.get('avg_final_match', 0):.4f}")
    print(f"  Quant Time:     {quant_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dLLM Quantization Experiment")
    parser.add_argument("--method", type=str, default="gptaq",
                        choices=["gptaq", "quarot+gptaq"])
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--act_bits", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="models/LLaDA-8B-Instruct")
    parser.add_argument("--n_cal_samples", type=int, default=128)
    parser.add_argument("--n_eval_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    args = parser.parse_args()

    run_experiment(**vars(args))
