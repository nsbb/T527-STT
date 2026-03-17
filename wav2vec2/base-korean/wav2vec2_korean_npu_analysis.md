# Wav2Vec2 한국어 T527 NPU 양자화 분석 보고서

## 요약

영어 Wav2Vec2(base-960h)는 T527 NPU uint8에서 CER 17.52%로 정상 동작하나, 동일 아키텍처의 한국어 모델(Kkonjeong/wav2vec2-base-korean)은 60종+ 양자화 시도에도 불구하고 전부 실패. 원인은 한국어 fine-tuning 과정에서 발생한 activation 동적 범위 확대이며, T527 NPU의 uint8 전용 HW 가속 제한과 결합되어 해결 불가능한 상태.

---

## 1. 현상 분석

### 1.1 영어 모델 — 정상 동작

| 항목 | 값 |
|------|-----|
| 모델 | facebook/wav2vec2-base-960h |
| 파라미터 | 94.4M, 12 Transformer layers |
| 양자화 | uint8 asymmetric_affine + moving_average (51 calibration samples) |
| NB 크기 | 87MB |
| ONNX FP32 CER | 9.74% (LibriSpeech test-clean 50샘플) |
| **NPU uint8 CER** | **17.52%** (동일 50샘플) |
| 양자화 열화 | +7.78%p (허용 범위) |
| 추론 시간 | 715ms / 5초 오디오 |

### 1.2 한국어 모델 — 전부 실패

| 항목 | 값 |
|------|-----|
| 모델 | Kkonjeong/wav2vec2-base-korean |
| 파라미터 | 94.4M, 12 Transformer layers (영어와 **완전 동일 아키텍처**) |
| 학습 데이터 | Zeroth-Korean 51시간 (뉴스/책 낭독체) |
| ONNX FP32 CER | 9.53% (Zeroth-Korean test 100샘플) |
| **NPU uint8 출력** | **ALL PAD 또는 ㅇ 토큰 쓰레기** |
| 시도 횟수 | 60종+ 양자화 조합, 21종 실기기 테스트 |
| 결과 | **전부 실패** |

### 1.3 시도한 전체 양자화 조합

#### T527 NPU 지원 데이터 타입과 결과

| 데이터 타입 | Pegasus quantizer | NB 크기 | T527 HW 가속 | 이 모델 결과 |
|------------|-------------------|---------|:-----------:|------------|
| **uint8** | `asymmetric_affine` | ~72MB | **O** (425ms) | 양자화 열화 → 쓰레기 출력 |
| **int8 (PCQ)** | `perchannel_symmetric_affine` | ~72MB | O (추정) | NB 생성 실패 (Acuity segfault) |
| **int16 (DFP)** | `dynamic_fixed_point` | ~153MB | **X** (status=-1) | NPU가 실행 거부 |
| **bf16** | `qbfloat16` | ~181MB | X | NB 생성 실패 (0 bytes) |
| **fp16** | `export --dtype float` | ~182MB | **X** (CPU fallback) | 실행되나 17.7초 (42배 느림) |
| **fp32** | — | ~362MB | X | SRAM 부족 |

> **T527 Vivante VIP9000 NPU는 uint8 연산만 HW 가속.** int16/bf16/fp16은 실행 거부 또는 CPU fallback으로, 실시간 추론 불가.

#### uint8 calibration/알고리즘 조합 (12종, 전부 garbled)

| # | calibration | 알고리즘 | ONNX 변형 | NB 크기 | 출력 |
|---|-------------|---------|----------|---------|------|
| 1 | 기본 (1 sample) | default | 원본 | 72MB | ALL PAD |
| 2 | 50 samples | moving_average (w=0.004) | 원본 | 73MB | ㅇ 토큰 과다 |
| 3 | 50 samples | moving_average v2 | 원본 | 73MB | garbled |
| 4 | 50 samples | KL divergence | 원본 | 73MB | garbled |
| 5 | 50 samples | entropy | 원본 | 73MB | garbled |
| 6 | 50 samples | moving_average | nopad10 | 73MB | garbled |
| 7 | 50 samples | KL divergence | nopad10 | 73MB | garbled |
| 8 | 50 samples | moving_average | opset12+sim | 72MB | garbled |
| 9 | 50 samples | default | opset12+sim | 72MB | garbled |
| 10 | 50 samples | moving_average | 2L pruned | 15MB | status=-1 |
| 11 | 50 samples | moving_average | 6L pruned | 39MB | status=-1 |
| 12 | 50 samples | moving_average | 6L+ReLU | 39MB | status=-1 |

#### Post-training 최적화 시도 (10종, 전부 실패)

| 방법 | FP32 영향 | uint8 개선 | 결론 |
|------|----------|-----------|------|
| SmoothQuant v1 (α=0.3-0.9) | **파괴** | — | 잔차 연결 구조 비호환 |
| **SmoothQuant v3** (Div 노드) | **100% 보존** | **없음** | FP32 보존 최초 성공, uint8은 미미 |
| SQv3 + AttnClip30 | 81.9% | 3.4% | 결합해도 개선 없음 |
| temperature scaling (T=4, T=16) | 파괴 | — | activation range 줄지만 FP32 손상 |
| weight clipping (99.99~99.5%) | 파괴 | — | 0.02%만 clip해도 FP32 파괴 |
| k_proj bias 이식/스케일 | 없음 | — | softmax shift-invariance |
| Pegasus AttnClip100 | — | 40.9% | 원본(46.3%)보다 **악화** |
| Pegasus AttnClip50 | — | 34.9% | 더 악화 |
| Pegasus RangeClip200 | — | 11.4% | 완전 파괴 |
| Q/K vector clamping | 열화 | — | FP32 보존 안 됨 |

#### 구조 변경 시도 (4종, 전부 실패)

| 방법 | 결과 |
|------|------|
| CNN-only (feature extractor만 분리) | NPU status=-1 |
| split: CNN(uint8) → Transformer(int16) | Part B NPU HANG |
| hybrid `--hybrid` (CNN uint8 + Transformer int16) | NPU HANG |
| combo (ReLU 치환 + 6L pruning + nopad) | NPU status=-1 |

---

## 2. 원인 분석

### 2.1 핵심 원인: Activation 동적 범위

동일 아키텍처(94.4M, 12L)인데 영어는 되고 한국어는 안 되는 이유:

**한국어 모델의 내부 activation 값 범위가 영어 대비 2~27배 넓음.**

uint8은 0~255 (256단계)로 실수를 표현. 범위가 넓을수록 한 단계의 간격(scale)이 커져서 정밀도가 떨어짐.

| 지표 | 한국어 | 영어 | 배율 | 의미 |
|------|--------|------|:----:|------|
| Attention MatMul avg range | 37.55 | 19.48 | **1.9x** | Q@K^T 결과 범위 |
| Attention MatMul max range | 276.7 | 49.1 | **5.6x** | 최대 범위 |
| **Softmax avg range** | **0.82** | **0.03** | **27x** | 핵심 병목 |
| FFN output range | 136-254 | 30-69 | **3-4x** | FFN 레이어 출력 |
| CNN feature range | 454-960 | 129-192 | **3-5x** | CNN 특징 추출기 |
| uint8 scale > 0.5인 텐서 수 | **25개** | 3개 | **8x** | 양자화 오류 큰 텐서 수 |
| Pegasus uint8 argmax agreement | **46.3%** | **~85%** | — | FP32 대비 결과 일치율 |

### 2.2 왜 한국어가 더 넓은가?

**Attention 패턴의 근본적 차이:**

- **영어**: 특정 위치에 sharp attention (near one-hot) → softmax 출력이 [0.01, 0.01, 0.95, 0.02, ...] 형태 → 범위 좁음
- **한국어**: distributed attention (여러 위치에 분산) → softmax 출력이 [0.15, 0.20, 0.25, 0.18, ...] 형태 → 범위 넓음

이 차이는 fine-tuning 데이터(Zeroth-Korean 51시간)의 특성에서 비롯된 것으로 추정. 한국어의 교착어 특성(조사, 어미 변화)으로 인해 attention이 더 넓게 분산되는 패턴이 학습됨.

### 2.3 uint8 양자화 실패 메커니즘

```
FP32 logits (예시):
  텍스트 토큰: [25.3, 24.8, 26.1, 23.9, ...]    ← 정답
  PAD 토큰:    [23.5]                              ← gap = 2.6

uint8 양자화 후 (scale=0.15, 256단계):
  텍스트 토큰: [169, 165, 174, 159, ...]
  PAD 토큰:    [157]                               ← gap = 2~17단계

  → 양자화 노이즈(±2~3단계)가 gap(2~17단계)과 비슷
  → 12 layer 누적 시 argmax 결과가 46.3%만 일치
  → CTC 디코딩 결과 완전 파괴
```

영어 모델은 scale이 작아서(범위가 좁으므로) 같은 256단계로도 충분한 정밀도 유지.

### 2.4 fp16 테스트 — CPU fallback 확인

fp16 NB(182MB)를 T527에서 실행한 결과:

| 항목 | 값 |
|------|-----|
| NB 생성 | 성공 (182MB) |
| NPU 실행 | 성공 (vpm run ret=0) |
| 추론 시간 | **17,740ms** (17.7초) |
| uint8 대비 | **42배 느림** (425ms → 17,740ms) |
| ONNX FP32 대비 argmax agreement | **98.0%** |
| vpm_run 로그 | `quant_format=0, none-quant` |

> 17.7초는 NPU HW 가속이 아니라 **CPU fallback**. T527 NPU는 uint8 이외의 데이터 타입을 HW로 처리하지 않음.

### 2.5 도메인 미스매치 (별도 문제)

양자화 문제와 독립적으로, 이 모델은 월패드 음성도 인식 못함:

| 테스트셋 | ONNX FP32 CER | 비고 |
|---------|:------------:|------|
| **Zeroth-Korean** (학습 도메인) | **9.53%** | 정상 |
| modelhouse_2m_noheater | 128.5% | 도메인 미스매치 |
| 7F_KSK | 137.1% | 도메인 미스매치 |
| 7F_HJY | 148.5% | 도메인 미스매치 |
| modelhouse_2m | 174.5% | 도메인 미스매치 |
| modelhouse_3m | 196.7% | 도메인 미스매치 |

→ 양자화가 해결되더라도 월패드 데이터로 fine-tuning 필수.

---

## 3. 결론

### 3.1 T527 NPU 한계 — 변경 불가능한 제약

| 제약 | 내용 |
|------|------|
| HW 가속 타입 | **uint8만** (int8 PCQ 추정 가능하나 Acuity 도구 비호환) |
| int16/bf16 | NPU가 실행 거부 (status=-1) |
| fp16 | 실행은 되나 CPU fallback (17.7초, 실시간 불가) |
| SRAM | fp32(362MB) 수용 불가 |

### 3.2 한국어 Wav2Vec2 양자화 실패 — 모델 특성

| 영어 (성공) | 한국어 (실패) |
|------------|-------------|
| sharp attention (one-hot에 가까움) | distributed attention (분산) |
| softmax range 0.03 | softmax range 0.82 (**27배**) |
| uint8 argmax agreement ~85% | uint8 argmax agreement 46.3% |
| CER 열화 +7.78%p (허용) | CER 열화 → **출력 파괴** |

### 3.3 해결 방안

#### [방안 A] QAT Fine-tuning (권장)

facebook/wav2vec2-base (영어 uint8 성공한 동일 체크포인트)에서 시작하여, 한국어 데이터로 **Quantization-Aware Training** 수행.

```
facebook/wav2vec2-base (pretrained, 영어 uint8 성공)
    ↓ QAT fine-tuning (한국어 데이터)
    ↓ 학습 중 uint8 양자화 시뮬레이션 → activation range 자연 억제
    ↓ ONNX export
    ↓ Acuity uint8 NB 변환
    ↓ T527 NPU 배포
```

**기대 근거:**
- 동일 base 모델이 영어에서 uint8 성공 → 가중치 초기값이 uint8 친화적
- QAT가 학습 중 양자화 오류를 역전파 → activation range 자동 최적화
- HuggingFace `optimum` 라이브러리 QAT 지원

**필요 리소스:**
- GPU 서버 (V100/A100, 수 시간)
- 학습 데이터: Zeroth-Korean 51시간 + 월패드 데이터
- HuggingFace Transformers + Optimum

#### [방안 B] 다른 한국어 모델 탐색

activation range가 좁은 한국어 STT 모델을 찾아서 uint8 양자화 테스트.

- CNN 기반 모델 (Conformer, Citrinet 등)은 Transformer보다 양자화 친화적
- KoCitrinet이 T527 uint8에서 성공한 것이 증거

#### [방안 C] 현재 KoCitrinet 개선

이미 T527에서 동작하는 KoCitrinet (CER 44.44%, 120ms)을 개선:
- 월패드 데이터로 추가 fine-tuning
- 더 큰 모델 (500프레임 등) 시도

### 3.4 증거 자료

| 자료 | 위치 |
|------|------|
| 영어 uint8 NPU 테스트 결과 (50샘플) | `base-960h-en/test_results_librispeech.csv` |
| 한국어 ONNX FP32 Zeroth-Korean (100샘플) | `base-korean/test_results_zeroth_korean_pytorch_fp32.csv` |
| 한국어 ONNX FP32 월패드 (5개 테스트셋) | `base-korean/test_results_*_onnx_fp32.csv` |
| fp16 NPU 테스트 결과 | `base-korean/test_results_fp16_npu.csv` |
| KoCitrinet NPU 테스트 결과 (330샘플) | `ko_citrinet/test_results_sample30.csv` |
| 양자화 시도 전체 기록 | `base-korean/README.md` |
| 모든 CSV는 GitHub에 공개 | https://github.com/nsbb/T527-STT |

---

*작성일: 2026-03-17*
*T527 NPU: Vivante VIP9000NANOSI_PLUS (PID 0x10000016), driver v0x00010d00*
*Acuity Toolkit: v6.12.0 / v6.21.0*
