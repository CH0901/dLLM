# dLLM Quantization Experiment Pipeline

## 설치

```bash
# dLLM/ 디렉토리에 이 폴더를 복사
# 구조: dLLM/dllm_quant/

# 필요 패키지 (이미 설치되어 있을 가능성 높음)
pip install datasets transformers torch numpy
```

## 디렉토리 구조

```
dLLM/
├── models/LLaDA-8B-Instruct/   ← 이미 있음
├── QDLM/                       ← 이미 있음
├── GPTAQ/                      ← 이미 있음
├── Fast-dLLM/                  ← 이미 있음
└── dllm_quant/                 ← 이 폴더
    ├── config.py               # 실험 설정
    ├── llada_utils.py          # LLaDA 모델 구조 어댑터
    ├── calibration.py          # 캘리브레이션 데이터
    ├── run_experiment.py       # 메인 실행
    ├── analyze_results.py      # 결과 분석
    ├── quantization/
    │   ├── base.py             # 양자화 베이스
    │   ├── gptaq.py            # GPTAQ (dLLM 어댑트)
    │   ├── quarot.py           # QuaRot rotation
    │   └── quarot_gptaq.py     # QuaRot + GPTAQ
    └── evaluation/
        ├── decoding.py         # masked decoding loop
        ├── metrics.py          # 분석 메트릭
        └── comparator.py       # FP vs Quant 비교기
```

## 실행 방법

### 1. GPTAQ W4A16 (Phase 1 — 가장 먼저)

```bash
cd dLLM/dllm_quant
python run_experiment.py \
    --method gptaq \
    --weight_bits 4 \
    --act_bits 16 \
    --model_path ../models/LLaDA-8B-Instruct \
    --n_cal_samples 128 \
    --n_eval_samples 5
```

### 2. GPTAQ W3A16

```bash
python run_experiment.py --method gptaq --weight_bits 3 --act_bits 16
```

### 3. QuaRot + GPTAQ W4A16 (Phase 2)

```bash
python run_experiment.py --method quarot+gptaq --weight_bits 4 --act_bits 16
```

### 4. 양자화만 (eval 스킵, 빠르게 모델만 저장)

```bash
python run_experiment.py --method gptaq --weight_bits 4 --act_bits 16 --skip_eval
```

### 5. 결과 분석

```bash
python analyze_results.py --results_dir ../results
```

## 출력

```
results/
├── gptaq_W4A16/
│   ├── config.json            # 실험 설정
│   ├── quantized_model/       # 저장된 양자화 모델
│   ├── summary.json           # 집계 결과
│   └── sample_000.json        # 개별 샘플 결과
└── quarot+gptaq_W4A16/
    └── ...
```

## 다음 단계: GSM8K / HumanEval 평가

양자화 모델이 저장되면 별도 eval 스크립트로 task 평가:

```bash
# 예시 (eval_benchmarks.py 구현 후)
python eval_benchmarks.py \
    --model_path results/gptaq_W4A16/quantized_model \
    --tasks gsm8k humaneval
```
