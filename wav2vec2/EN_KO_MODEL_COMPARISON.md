# Wav2Vec2 영어 vs 한국어 모델 비교 — 양자화 성공/실패 원인 분석

## 결론

**영어(base-960h)와 한국어(base-korean) 모델은 동일한 아키텍처**(12-layer Transformer, 94M params)이지만, 한국어 모델의 **출력 logit margin이 극도로 작아** uint8 양자화 시 argmax 결과가 완전히 뒤집힌다. 이것이 영어는 양자화 성공(CER 17.52%), 한국어는 실패(CER 100%)하는 근본 원인이다.

---

## 1. 아키텍처 비교

두 모델은 **완전히 동일한 아키텍처**를 사용한다:

| 항목 | 영어 (base-960h) | 한국어 (base-korean) |
|------|-----------------|---------------------|
| 원본 | facebook/wav2vec2-base-960h | Kkonjeong/wav2vec2-base-korean |
| Pretrain | LibriSpeech 960h (영어) | XLS-R (53개 언어) → 한국어 fine-tune |
| Encoder Layers | 12 | 12 |
| Hidden Dim | 768 | 768 |
| Attention Heads | 12 | 12 |
| FFN Dim | 3072 | 3072 |
| CNN Layers | 7 (1D Conv) | 7 (1D Conv) |
| CNN Kernel | [10, 3, 3, 3, 3, 2, 2] | [10, 3, 3, 3, 3, 2, 2] |
| Parameters | 94.4M | 94.4M |
| ONNX 크기 | 361MB | 361MB |
| Initializers | 211 | 211 |

**아키텍처 레벨에서 두 모델은 차이가 없다.** 동일한 layer 수, 동일한 hidden dimension, 동일한 attention head 수. Weight tensor의 shape도 모두 동일하다 (lm_head 제외).

---

## 2. ONNX 그래프 차이

| 항목 | 영어 | 한국어 (원본) | 한국어 (opset12 변환) |
|------|------|-----------|-------------------|
| Opset | **12** | **14** | **12** |
| Nodes | 957 | 1306 | 667 |
| Op types | 18종 | 21종 | 16종 |
| Input | `[1, 80000]` (5초) | `[1, 48000]` (3초) | `[1, 48000]` (3초) |
| Output | `[1, 249, 32]` | `[1, 149, 56]` | `[1, 149, 56]` |

### Opset 차이의 영향

한국어 원본은 opset 14로 export되어 349개 추가 노드(Shape, Gather, Cast, Concat 등 동적 shape 연산)가 포함된다. 이를 opset 12로 re-export + onnxsim 적용하면 667개로 줄어들어 영어(957)보다 오히려 적어진다.

**opset 12 변환 후에도 양자화 실패** → opset/노드 수는 양자화 실패의 원인이 아님.

### Op type 비교 (opset 12 변환 후)

| Op | 영어 | 한국어 (op12) | 차이 |
|----|:---:|:---:|------|
| Add | 172 | 172 | 동일 |
| MatMul | 98 | 98 | 동일 |
| Mul | 79 | 79 | 동일 |
| Transpose | 63 | 51 | 영어 +12 |
| ReduceMean | 52 | 52 | 동일 |
| Reshape | 98 | 48 | 영어 +50 |
| Div | 46 | 46 | 동일 |
| Softmax | 12 | 12 | 동일 |
| Conv | 8 | 8 | 동일 |

Reshape/Transpose 차이는 onnxsim 최적화 수준 차이일 뿐, 연산 의미는 동일.

---

## 3. Weight 분포 비교

두 모델의 weight tensor 통계:

| 항목 | 영어 | 한국어 |
|------|------|--------|
| Weight tensors | 210 | 212 |
| abs_max 평균 | 1.1988 | 0.8831 |
| abs_max 최대 | **18.0781** | **5.7580** |
| std 평균 | 0.2164 | 0.0856 |
| std 최대 | **15.8573** | **0.5737** |

### 영어 모델의 특이한 weight

```
layers.11.attention.k_proj.bias: abs_max=18.08, std=15.86
layers.10.attention.k_proj.bias: abs_max=13.45, std=6.75
layers.8.attention.k_proj.bias:  abs_max=6.77,  std=1.07
```

영어 모델은 후반부 attention key bias에 **극단적으로 큰 값**이 있다. 이는 강한 attention 패턴을 형성하여 양자화 노이즈에 대한 내성을 높인다.

### 한국어 모델의 특징

```
max abs_max: 5.76 (MatMul weight)
max std: 0.57 (Mul constant)
```

한국어 모델은 모든 weight가 **비교적 균일하고 작다**. 강한 attention bias가 없어 양자화 노이즈에 취약하다.

---

## 4. 추론 시 활성값(Activation) 비교 — 핵심 원인

동일한 전처리(wav2vec2 normalize: zero-mean, unit-variance)를 적용한 뒤 ONNX FP32 추론:

| 항목 | 영어 | 한국어 |
|------|------|--------|
| **입력 범위 (normalized)** | [-11.81, 9.12] | [-6.01, 6.90] |
| 입력 std | 1.0 | 1.0 |
| **출력 logit 범위** | **[-36.06, 17.34]** | **[-10.33, 11.97]** |
| **출력 logit std** | **8.39** | **1.95** |
| uint8 입력 step size | 0.0821 | 0.0506 |
| uint8 입력 SNR | 34.1 dB | 37.7 dB |

### CTC Argmax Margin — 양자화 실패의 직접 원인

| 항목 | 영어 | 한국어 |
|------|------|--------|
| **Top1-Top2 margin 평균** | **10.85** | **3.08** |
| **Top1-Top2 margin 최소** | **0.34** | **0.005** |
| Blank 비율 | 0/249 (0%) | 0/149 (0%) |

**CTC argmax margin** = 각 시간 프레임에서 가장 높은 logit과 두 번째 logit의 차이.

- **영어**: margin 최소 0.34 → uint8 step size(~0.08)보다 4배 이상 큼 → **양자화해도 argmax 안정**
- **한국어**: margin 최소 **0.005** → uint8 step size(~0.05)보다 **10배 작음** → **양자화하면 argmax 뒤집힘**

### 시각화

```
영어 logit (특정 프레임):
  token A: 15.23  ███████████████▎     ← argmax
  token B:  4.38  ████▍                 margin = 10.85
  token C:  2.11  ██▏
  → uint8 양자화 후에도 A가 최고값 유지

한국어 logit (특정 프레임):
  token ㅇ: 3.012  ███                  ← argmax
  token ㅏ: 3.007  ███                   margin = 0.005
  token ㄴ: 2.998  ██▉
  → uint8 step size 0.05로 양자화하면 세 토큰이 같은 값 → argmax 무작위
```

---

## 5. 왜 한국어 모델의 margin이 작은가?

### 5.1 Fine-tuning 데이터 차이

- **영어 base-960h**: LibriSpeech 960시간으로 처음부터 학습 (pretrain + fine-tune 통합)
- **한국어 base-korean**: XLS-R (53개 언어 사전학습) → 한국어 데이터로 fine-tune

XLS-R에서 한국어로 fine-tune할 때, CTC head의 학습이 불충분하면 logit 분포가 **soft** (불확실)해진다. 영어 모델은 단일 언어로 충분히 학습되어 logit이 **sharp** (확신)하다.

### 5.2 토큰 체계 차이

| | 영어 | 한국어 |
|---|---|---|
| 토큰 수 | 32 (A-Z + 특수) | 56 (자모 + 특수) |
| 토큰 유형 | 알파벳 | 초성 19 + 중성 21 + 종성 11 + 특수 5 |
| 혼동 가능성 | 낮음 (A≠B) | **높음** (ㅇ≈ㄴ≈ㅁ 유사 음가) |

한국어 자모는 음성적으로 유사한 토큰이 많아 (예: ㅂ/ㅃ/ㅍ, ㅈ/ㅉ/ㅊ) 모델이 **토큰 간 구분에 확신을 갖기 어렵다**. 이것이 logit margin을 줄인다.

### 5.3 출력 logit 동적 범위

- 영어: logit range = 53.4 (= 17.34 - (-36.06)), std = 8.39
- 한국어: logit range = 22.3 (= 11.97 - (-10.33)), std = 1.95

한국어 모델의 logit 동적 범위가 영어의 **42%**에 불과하다. 같은 uint8 (256단계)로 양자화하면 한국어의 유효 정밀도는 영어의 절반 이하.

---

## 6. T527 NPU 양자화 결과 비교

| 항목 | 영어 uint8 | 한국어 uint8 |
|------|-----------|-------------|
| NB 크기 | 88MB | 72MB |
| T527 NPU 실행 | 정상 동작 | 정상 동작 |
| 추론 시간 | 715ms | 415ms |
| CER/WER | **CER 17.52%** | **CER 100%+** |
| Logit margin | 평균 10.85, 최소 0.34 | 평균 3.08, **최소 0.005** |
| 양자화 결과 | argmax 유지 | **argmax 전부 뒤집힘** |

---

## 7. 시도한 개선 방법과 결과

| # | 방법 | 결과 | 이유 |
|---|------|------|------|
| 1 | uint8 MA (1/100/300/1000 calib) | CER 100% | margin < step size |
| 2 | uint8 KL divergence | CER 100% (ALL PAD) | 동일 |
| 3 | uint8 min_max | CER 100% (ALL PAD) | 동일 |
| 4 | uint8 + amplitude norm 5.0 | CER 100% | 입력 범위 변경해도 logit margin 불변 |
| 5 | uint8 KL + amp norm | CER 100% | 동일 |
| 6 | int16 DFP | status=-1 (153MB) | NB 크기 초과 |
| 7 | fp16 | 17,740ms | CPU fallback, HW 미가속 |
| 8 | 6L pruned uint8 | FP32에서도 garbage | fine-tuning 필요 |
| 9 | 3-part split (CNN+L05+L611) 각 uint8 | CER 100% | CNN 파트가 고정값 출력 |
| 10 | PCQ int8 (perchannel) | 미시도 (Zipformer에서 악화 확인) | |
| 11 | int16 DFP (KoCitrinet 테스트) | CER 330% | DFP가 AA보다 나쁨 |

---

## 8. 해결 가능한 방향

### 8.1 QAT (Quantization-Aware Training)

학습 과정에서 양자화를 시뮬레이션하여 logit margin을 키우는 방법. 가장 확실하지만 학습 인프라 필요.

### 8.2 Language Model + Beam Search

CTC greedy (argmax) 대신 beam search + 한국어 n-gram LM 사용. 양자화된 logit이 불확실해도 LM이 올바른 토큰 시퀀스를 선택. 하지만 T527 NPU가 아닌 CPU에서 처리.

### 8.3 Knowledge Distillation

영어 모델처럼 sharp한 logit을 출력하도록 teacher-student 학습. 한국어 teacher (FP32) → student (uint8 친화적) 구조.

### 8.4 RK3588 방식 (Split INT8+FP16)

rknn-stt에서 검증된 방법: 전반부 INT8 + 후반부 FP16. **T527에서는 FP16이 HW 미지원**(CPU fallback 25배 느림)이라 실용적이지 않음.

---

## 9. 참고: RK3588과의 비교

동일한 한국어 모델(XLS-R-300M, 24L)을 RK3588 NPU에서 Split INT8+FP16으로 CER 11.78% 달성:

| 항목 | T527 (Acuity) | RK3588 (RKNN) |
|------|--------------|---------------|
| 양자화 | uint8 전체 | Split (INT8 + **FP16**) |
| FP16 HW 가속 | **미지원** | **지원** |
| 한국어 CER | 100% (실패) | **11.78%** |
| 핵심 차이 | 후반 레이어도 uint8 → margin 파괴 | 후반 레이어 **FP16 → margin 보존** |

---

## 10. 요약

```
영어 모델                              한국어 모델
┌─────────────┐                       ┌─────────────┐
│ Logit       │                       │ Logit       │
│ margin:     │                       │ margin:     │
│ 최소 0.34   │                       │ 최소 0.005  │
│ (uint8     │                       │ (uint8     │
│  step 0.08) │                       │  step 0.05) │
│             │                       │             │
│ 0.34 > 0.08 │ ← argmax 안정         │ 0.005<0.05  │ ← argmax 뒤집힘
│ → CER 17.5% │                       │ → CER 100%  │
└─────────────┘                       └─────────────┘
```

**동일 아키텍처, 동일 양자화 방법**이지만 한국어 모델의 logit margin이 uint8 step size보다 작아서 양자화 후 argmax 결과가 완전히 달라진다. 이는 모델의 학습 데이터와 토큰 체계의 차이에서 비롯되며, 양자화 도구(Acuity)의 문제가 아니다.
