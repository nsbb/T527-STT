# Split Model 접근법: Encoder(NPU uint8) + lm_head(CPU float32)

**날짜:** 2026-03-24
**상태:** 제안 (미구현)
**목적:** T527 NPU의 W8A8 강제 제약을 우회하여, 정밀도가 필요한 마지막 레이어만 CPU float32로 실행

---

## 1. 배경: 왜 이 방법이 필요한가

### 1.1 현재 문제

T527 NPU (Vivante VIP9000NANOSI_PLUS)는 **weight와 activation 모두 uint8 강제** (W8A8).
마지막 레이어(lm_head)의 출력(logits)도 uint8로 양자화되어, **1등과 2등 logit의 차이(margin)가 양자화 step보다 작으면 argmax가 뒤집힌다.**

```
현재 (전부 NPU uint8):
  audio → [CNN + Encoder L0-11 + lm_head] → logits (uint8, 256단계) → argmax ✗
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          전부 NPU uint8 (W8A8)

문제: logits이 uint8이라 margin < step → argmax 뒤집힘 → CER 100%
```

### 1.2 OpenVINO가 같은 문제를 해결한 방법

OpenVINO NNCF는 wav2vec2를 INT8 양자화할 때 94개 연산 중 **3개를 FP32로 되돌려서** WER 0.5 → 0.94로 복구. (conv_layers 1,2,3을 FP32로 복원)

```
OpenVINO: Intel CPU/GPU → mixed INT8/FP32 지원 → 문제 레이어만 FP32 → 해결
T527 NPU: Vivante NPU → mixed precision 미지원 → Acuity에 W8A16 옵션 없음
```

**T527에서는 Acuity/NPU 레벨에서 mixed precision이 불가능.** 하지만 **모델을 물리적으로 분리**하면 가능하다.

### 1.3 핵심 아이디어

모델을 encoder(NPU)와 lm_head(CPU)로 분리하여, 수동 mixed precision을 구현한다.

```
Split Model:
  audio → [CNN + Encoder L0-11] → hidden states (uint8, NPU)
          ^^^^^^^^^^^^^^^^^^^^^^
          NPU uint8 (무거운 연산, ~400ms)
                                      ↓ dequantize (uint8 → float32)
                                 [lm_head] → logits (float32, CPU) → argmax ✓
                                 ^^^^^^^^^^
                                 CPU float32 (가벼운 연산, ~5ms)
```

---

## 2. 기술 상세

### 2.1 모델 구조와 분리 지점

```
wav2vec2-base 전체 구조:

[CNN Feature Extractor]     7 conv layers
  ↓
[Feature Projection]        Linear(512→768)
  ↓
[Encoder Layer 0]           Self-Attention + FFN
  ↓
[Encoder Layer 1~10]        ...
  ↓
[Encoder Layer 11]          Self-Attention + FFN
  ↓                         ← ★ 여기서 분리 ★
[Dropout]
  ↓
[lm_head]                   Linear(768→vocab_size) + bias
  ↓
logits [1, T, vocab_size]
```

**분리 지점:** Encoder Layer 11 출력 (= dropout 이전 또는 이후)

### 2.2 각 파트의 역할

**Part A: Encoder (NPU uint8)**

| 항목 | 값 |
|------|-----|
| 구성 | CNN (7 conv) + Feature Projection + Encoder L0-11 |
| 입력 | `[1, 48000]` float32 raw audio → uint8 양자화 |
| 출력 | `[1, 149, 768]` uint8 (hidden states) |
| 연산량 | ~94M multiply-adds (무거움) |
| 추론 시간 | ~400ms (NPU) |
| NB 크기 | ~70MB (기존과 유사, lm_head 제거로 약간 감소) |

**Part B: lm_head (CPU float32)**

| 항목 | 값 |
|------|-----|
| 구성 | Linear(768→56) + bias |
| 입력 | `[149, 768]` float32 (dequantized hidden states) |
| 출력 | `[149, 56]` float32 (logits) |
| 연산량 | 149 × 768 × 56 = 6.4M multiply-adds (가벼움) |
| 추론 시간 | **~5ms** (ARM CPU) |
| 파라미터 크기 | weight 768×56×4 + bias 56×4 = **172KB** |

### 2.3 용어 정리

```
hidden states: Encoder의 출력. shape [1, T, 768]. 로짓이 아님.
logits:        lm_head의 출력. shape [1, T, vocab_size]. CTC argmax의 입력.
lm_head:       마지막 Linear layer. hidden_dim(768) → vocab_size(56). 행렬곱 한 번.
```

### 2.4 왜 이게 margin 문제를 해결하는가

```
현재 (전부 uint8):
  hidden states (uint8) → lm_head (uint8 weight × uint8 activation)
  → logits (uint8, 256단계) → margin 0.005 < step 0.05 → argmax 뒤집힘 ✗

Split (encoder uint8 + lm_head float32):
  hidden states (uint8 → dequant → float32) → lm_head (float32 weight × float32 activation)
  → logits (float32, ~10^7단계) → margin 그대로 보존 → argmax 정확 ✓
```

hidden states에 uint8 양자화 오차가 있지만, 그 오차가 lm_head의 float32 matmul을 통과하면서 **증폭되지 않고 원래 비율대로 logits에 반영**된다. float32 logits에서의 margin은 충분히 크다.

### 2.5 이 방법의 LLM 양자화와의 유사성

```
LLM W4A16 (GPTQ/AWQ):
  weight(4-bit) × activation(FP16) → 하드웨어가 자동 mixed precision

우리 Split Model:
  Encoder: weight(uint8) × activation(uint8) → NPU 실행
  lm_head: weight(float32) × activation(float32) → CPU 실행
  → 수동으로 모델을 쪼개서 mixed precision 구현

효과는 동일: 정밀도가 필요한 부분만 높은 precision으로 실행.
```

---

## 3. 구현 계획

### 3.1 Step 1: ONNX 분리 (5분)

```python
import onnx

model = onnx.load("wav2vec2_ko_3s.onnx")

# lm_head (마지막 Linear) 제거
# Encoder 마지막 출력을 모델의 output으로 변경
# → encoder_only.onnx 생성 (입력: [1,48000], 출력: [1,149,768])

# lm_head weight와 bias를 별도 numpy 파일로 저장
# → lm_head_weight.npy (shape: [56, 768], float32)
# → lm_head_bias.npy (shape: [56], float32)
```

**주의:** ONNX에서 lm_head를 찾아 제거하고, 바로 앞 노드의 출력을 graph output으로 설정해야 함. dropout도 포함할지 제외할지 결정 필요 (inference에서는 dropout=0이므로 제외 가능).

### 3.2 Step 2: Acuity NB 변환 (10분)

```bash
# encoder_only.onnx → import → quantize → export → NB
pegasus import onnx --model encoder_only.onnx \
  --output-model encoder_only.json --output-data encoder_only.data

pegasus quantize --model encoder_only.json --model-data encoder_only.data \
  --with-input-meta inputmeta.yml --rebuild-all \
  --quantizer asymmetric_affine --qtype uint8 \
  --algorithm kl_divergence \
  --model-quantize encoder_only_uint8.quantize

pegasus export ovxlib --model encoder_only.json --model-data encoder_only.data \
  --model-quantize encoder_only_uint8.quantize \
  --dtype quantized --pack-nbg-unify \
  --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk $VSIM --output-path encoder_only_nbg/
```

**출력 NB:** 입력 `[1, 48000]` uint8, 출력 `[1, 149, 768]` uint8
**nbg_meta.json에서 출력 scale/zp 확인 필수** → dequantize에 사용

### 3.3 Step 3: 디바이스 코드 구현 — vpm_run 테스트용 (Python)

```python
import numpy as np

# 1. NPU output 로드 (vpm_run 결과)
hidden_uint8 = np.fromfile("output_0.dat", dtype=np.uint8).reshape(149, 768)

# 2. Dequantize (nbg_meta.json에서 scale/zp 읽기)
hidden_float = (hidden_uint8.astype(np.float32) - zero_point) * scale  # [149, 768]

# 3. lm_head matmul (float32)
weight = np.load("lm_head_weight.npy")  # [56, 768]
bias = np.load("lm_head_bias.npy")      # [56]
logits = hidden_float @ weight.T + bias  # [149, 56] float32

# 4. CTC greedy decode
token_ids = np.argmax(logits, axis=-1)   # argmax on float32 → 정확!
# ... CTC decode (blank 제거, 중복 제거)
```

### 3.4 Step 4: 디바이스 코드 구현 — Android JNI (C)

```c
// awwav2vecsdk.c 수정

// NPU 추론 (encoder only)
awnn_run(context);
float **results = awnn_get_output_buffers(context);

// NPU 출력: uint8 [149, 768] → dequantize → float32
float hidden[149 * 768];
uint8_t *npu_output = (uint8_t*)results[0];
for (int i = 0; i < 149 * 768; i++) {
    hidden[i] = (npu_output[i] - OUTPUT_ZP) * OUTPUT_SCALE;
}

// lm_head: float32 matmul
// weight [56, 768], bias [56] — 앱 assets에서 로드
float logits[149 * 56];
for (int t = 0; t < 149; t++) {
    for (int v = 0; v < 56; v++) {
        float sum = lm_head_bias[v];
        for (int h = 0; h < 768; h++) {
            sum += hidden[t * 768 + h] * lm_head_weight[v * 768 + h];
        }
        logits[t * 56 + v] = sum;
    }
}

// argmax on float32 logits → CTC decode
for (int t = 0; t < 149; t++) {
    int best_id = 0;
    float best_val = logits[t * 56];
    for (int v = 1; v < 56; v++) {
        if (logits[t * 56 + v] > best_val) {
            best_val = logits[t * 56 + v];
            best_id = v;
        }
    }
    token_ids[t] = best_id;
}
```

### 3.5 Step 5: 성능 검증

```bash
# vpm_run으로 encoder NB 테스트
$WIN_ADB push encoder_only_nbg_unify/network_binary.nb /data/local/tmp/split_test/
$WIN_ADB push input_0.dat /data/local/tmp/split_test/
$WIN_ADB shell "cd /data/local/tmp/split_test && \
  LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt"
$WIN_ADB pull /data/local/tmp/split_test/output_0.dat .

# Python에서 lm_head 실행 + CER 측정
python3 eval_split_model.py --output output_0.dat --gt ground_truth.txt
```

---

## 4. 실험 결과 (2026-03-24)

### 4.0 영어 모델 Split 실험 (검증용)

**모델:** facebook/wav2vec2-base-960h (vocab 32, 영어)
**방법:** ONNX에서 lm_head (MatMul+Add) 제거 → encoder_only.onnx → Acuity uint8 NB → T527 디바이스 → lm_head fp32 CPU

**ONNX 분리 검증:**
- 원본 모델 vs (encoder + 수동 lm_head): max diff = 0.000009 (사실상 동일)

**양자화 & NB 변환:**
- 51 calibration samples, moving_average(0.004), reverse_channel=false
- encoder NB: 92MB, 입력 scale=0.00286004 zp=137, 출력 scale=0.01440469 zp=134
- 디바이스 추론: ~715ms (기존 full model과 동일)

**CER 결과 (20 samples, LibriSpeech test-clean):**

| 방식 | CER | 비고 |
|------|-----|------|
| ONNX FP32 (서버) | 9.74% | 최상 (50 samples) |
| **Full uint8 (기존 NB)** | **17.52%** | 전체 NPU (50 samples) |
| **Split (encoder NPU uint8 + lm_head CPU fp32)** | **20.68%** | 새 방식 (20 samples) |

**분석:**
- Split이 기존 full uint8 (17.52%)보다 약간 나쁨 (+3.16%p)
- 원인: encoder 출력을 uint8→fp32로 dequantize하면서 정밀도 손실
- full uint8 NB는 Acuity가 encoder+lm_head를 통째로 최적화한 것이라 내부적으로 더 효율적
- **영어에서는 full uint8이 이미 잘 되므로 Split의 이점 없음**
- **Split의 진짜 가치: 한국어에서 full uint8 CER 100% → Split으로 개선되는지**

**샘플 출력 (Split):**

| # | Split 출력 | GT | CER |
|---|---|---|---|
| 0 | CORCORD RETURNED TO ITS PLACE AWIDST THE TENTS | CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS | 5.1% |
| 7 | I AW CONVINCED OF MHAT I SAY SAID THE COUNT | I AM CONVINCED OF WHAT I SAY SAID THE COUNT | 5.9% |
| 4 | CAN YOU IWAGINE WY BUCKINGHAW HAS BEEN SO VIOLENT | CAN YOU IMAGINE WHY BUCKINGHAM HAS BEEN SO VIOLENT | 6.0% |

→ 의미 있는 영어 텍스트 출력. M/W 혼동 패턴 (uint8 양자화 특성).

### 4.0.1 첫 시도 실패 기록

첫 번째 encoder NB는 KL divergence + 1 sample calibration으로 양자화 → CER 94.35% (거의 blank).
원인: **입력 scale/zp 불일치** (NB가 zp=119를 기대하는데 테스트 입력은 zp=137로 양자화됨).
수정: 기존 working model과 동일한 calibration (51 samples, MA, reverse_channel=false) → CER 20.68%.

---

### 4.1 한국어 모델 Split 실험 (핵심 테스트)

**모델:** wav2vec2-base-korean (Kkonjeong, vocab 56, 자모)
**방법:** 영어와 동일 — encoder_only.onnx → uint8 NB → T527 → lm_head fp32 CPU

**NB 변환:**
- 50 calibration tensors, KL divergence + fqb16, reverse_channel=false
- encoder NB: 75MB, 입력 scale=0.00072033 zp=121, 출력 scale=0.02796588 zp=140
- 디바이스 추론: ~415ms

**CER 결과 (20 samples, Zeroth-Korean):**

| 방식 | CER | 비고 |
|------|-----|------|
| ONNX FP32 (서버) | 33.74% | 최상 |
| **Full uint8 (기존)** | **100.86%** | 완전 실패 |
| **Split (encoder NPU uint8 + lm_head CPU fp32)** | **99.70%** | **개선 없음** |

**출력 예시:**

| # | Split 출력 | GT | CER |
|---|---|---|---|
| 0 | ㅗㄴ토기느  ㅇ 잉 ㄷ앟ㄴㅇ앙다 | 삼 분기에만 매출 일 조 이 | 120% |
| 3 | ㅡ리고  나 ㅇ | 당초 지난 십 칠 일 열린 회의에서 결 | 100% |
| 19 | 지금으 대통령이 ㅏ | 머니투데이 편집자주 소위 강남 삼 | 100% |

### 4.2 결론: Split model의 한계

**lm_head를 fp32로 빼도 한국어에서는 효과 없다.**

원인: 문제는 lm_head의 uint8 양자화가 아니라 **encoder 자체의 uint8 양자화 오차**.

```
영어 encoder uint8:  cos(NPU, FP32) ≈ 0.97  → hidden states 정보 충분 → lm_head fp32 효과 있음
한국어 encoder uint8: cos(NPU, FP32) ≈ 0.66  → hidden states 이미 망가짐 → lm_head fp32해도 무의미
```

**Split model이 효과적인 조건:**
1. encoder의 uint8 양자화 품질이 충분히 좋아야 함 (cos > 0.9)
2. margin 문제가 lm_head 단에서만 발생하는 경우

한국어 base-korean은 **encoder L8-11에서 양자화 오차가 누적**되어 hidden states 자체가 쓰레기 → lm_head를 아무리 정밀하게 해도 복구 불가.

**남은 경로:**
- QAT + margin loss → encoder 양자화 내성 강화 (진행 중)
- 영어→한국어 fine-tune → encoder activation 분포가 양자화에 유리 (attempt5 WER 40.6%)
- aihub 4356시간 대규모 학습 → 모델 자체 성능 향상

---

### 4.3 L0-7(NPU) + L8-11+lm_head(CPU fp32) 실험

**가설:** L8-11이 가장 양자화에 취약하니까, L0-7만 NPU uint8로 돌리고 L8-11+lm_head는 CPU fp32로 실행하면?

**구현:**
- ONNX를 Part A(CNN+L0-7, 264MB, 675노드)와 Part B(L8-11+lm_head, 114MB, 282노드)로 분리
- Part A: Acuity uint8 NB (52MB) → T527 NPU, ~320ms
- Part B: ONNX Runtime fp32 → 서버 Python에서 실행 (디바이스에서는 ONNX Runtime CPU)
- 분리 검증: Part A→Part B vs 원본 = max diff 0.0 (완벽 일치)

**CER 결과 (20 samples, Zeroth-Korean):**

| 방식 | CER | NPU 시간 |
|------|-----|---------|
| Full uint8 | 100.86% | 415ms |
| Split lm_head만 (L0-11 NPU + lm fp32) | 99.70% | 415ms |
| **Split L7 (L0-7 NPU + L8-11+lm fp32)** | **99.26%** | **320ms** |
| ONNX FP32 | 33.74% | 서버 |

**결론: L8-11을 fp32로 빼도 CER 개선 없음 (99.26% ≈ 100%).**

출력은 이전보다 한국어 단어가 약간 더 나옴 ("그리고 이 나므", "지금은 대통렬") — L8-11 fp32의 효과가 미미하게 있지만 실용적 개선은 아님.

**근본 원인:** L0-7의 uint8 양자화조차 한국어 모델에서는 정보를 충분히 보존하지 못함. 한국어 모델의 activation 분포가 영어보다 uint8에 취약 — 이건 모델 weight의 문제이지 분리 지점의 문제가 아님.

---

### 4.4 CNN(fp32) + Transformer(NPU uint8) + lm_head(fp32) — OpenVINO 방식

**가설:** OpenVINO NNCF도 conv_layers 1,2,3을 FP32로 복원하여 WER 회복. CNN이 양자화에 가장 민감하니 CNN을 fp32로 빼면?

**구현:**
- Part A: CNN+feature_projection+pos_conv+layer_norm (115 nodes, 37MB) → CPU fp32
- Part B: Transformer L0-11 (840 nodes, 340MB) → Acuity uint8 NB (72MB)
- Part C: lm_head (weight 768×56) → CPU fp32 matmul
- calibration: CNN fp32 출력 50개를 npy로 저장 → Transformer 양자화에 사용

**CER 결과 (20 samples):**

| 방식 | CER | NPU 시간 |
|------|-----|---------|
| **CNN(fp32) + TF(NPU uint8) + lm(fp32)** | **100.00% (전부 blank)** | **285ms** |

**오히려 악화.** 이전 split에서는 자모 파편이라도 나왔는데 이번엔 아예 blank.

원인: CNN fp32 출력을 uint8로 양자화해서 Transformer NPU에 넣을 때, **CNN의 정밀한 feature가 uint8 경계에서 손실**. 전체 uint8로 할 때는 Acuity가 CNN→Transformer를 통째로 최적화하지만, 분리하면 그 최적화가 깨짐.

---

## 5. 전체 실험 결과 종합

| 방식 | 영어 CER | 한국어 CER | 효과 |
|------|---------|-----------|------|
| ONNX FP32 | 9.74% | 33.74% | 서버 최상 |
| Full uint8 | **17.52%** | **100.86%** | 영어 OK, 한국어 실패 |
| Split lm_head (L0-11 NPU + lm fp32) | 20.68% | 99.70% | 효과 없음 |
| Split L7 (L0-7 NPU + L8-11+lm fp32) | — | 99.26% | 효과 없음 |
| **CNN+lm fp32 (OpenVINO 방식)** | — | **100.00% (전부 blank)** | **오히려 악화** |

### 속도 실측 (T527 ARM CPU)

T527 디바이스에서 Transformer 4레이어(L8-11) + lm_head를 naive C matmul로 실측:

| 구간 | T527 시간 | 비고 |
|------|----------|------|
| NPU (전체 모델, uint8) | 415ms | 기존 |
| NPU (Transformer only) | 285ms | Split 시 |
| **CPU naive matmul (L8-11 + lm_head)** | **391,854ms (6.5분)** | naive 3중 루프, 최적화 없음 |
| CPU ONNX Runtime (예상) | 4~40초 | NEON SIMD 최적화 시 |

→ L8-11을 CPU에서 돌리면 naive 기준 **6.5분**, 최적화해도 **수 초** — 실시간 사용 불가.
| QAT + margin loss | — | 진행 중 | margin 0.099 (개선 중) |
| fine-tune (attempt5) | — | WER 40.6% | 유일한 성공 |

**Split model 최종 결론:** 한국어 base-korean 모델은 어디서 잘라도 uint8 양자화가 안 됨. L0-7조차 한국어 activation 분포에서는 uint8 정보 손실이 치명적. **모델 weight 자체를 바꿔야 함** (fine-tune, QAT).

---

## 6. 예상 결과 (향후)

### 4.1 정확도

| 방식 | logits precision | margin 보존 | CER 예상 |
|------|-----------------|------------|---------|
| 현재 (전부 uint8) | uint8 (256단계) | ✗ | 100% |
| **Split (encoder uint8 + lm_head fp32)** | **float32** | **✓** | **ONNX float과 유사** |
| QAT (전부 uint8) | uint8 | 부분적 | 개선 중 |

**핵심:** lm_head가 float32로 실행되므로, logits의 margin은 ONNX float 추론과 동일.
encoder의 uint8 양자화 오차만 영향 → 이건 영어 모델에서도 있었고 CER 17.52%로 동작.

### 4.2 속도

| 구간 | 시간 | 디바이스 |
|------|------|---------|
| NPU encoder 추론 | ~400ms | NPU |
| dequantize (149×768) | ~0.1ms | CPU |
| lm_head matmul (149×768×56) | ~5ms | CPU |
| argmax + CTC decode | ~0.1ms | CPU |
| **총합** | **~405ms** | |

기존 ~400ms 대비 **5ms 추가** — 무시할 수 있는 수준.

### 4.3 메모리

| 파일 | 크기 |
|------|------|
| encoder NB | ~68MB (lm_head 제거로 기존 72MB에서 감소) |
| lm_head weight | 768 × 56 × 4 = 172KB |
| lm_head bias | 56 × 4 = 224B |
| **총합** | **~68MB** (기존과 거의 동일) |

---

## 6. 장단점

### 5.1 장점

1. **margin 문제 완전 해결** — float32 logits이므로 argmax 정확
2. **QAT 불필요** — encoder uint8 + lm_head float32면 QAT 없이도 동작 가능
3. **vocab 크기 제약 완화** — float32 logits이면 vocab 1900도 가능할 수 있음
4. **속도 거의 동일** — lm_head는 5ms (전체 400ms 대비 1.2%)
5. **구현 간단** — ONNX 분리 + matmul 코드 ~30줄

### 5.2 단점

1. **encoder 출력의 uint8 오차는 여전히 존재** — hidden states에 양자화 노이즈
2. **모델 2개 관리** — NB + lm_head weight 파일
3. **코드 수정 필요** — JNI에서 matmul 직접 구현 (또는 라이브러리 사용)
4. **encoder 출력 shape이 커짐** — 기존 output [149,56]=8.3KB → [149,768]=114KB

### 5.3 리스크

**가장 큰 리스크:** encoder 출력의 uint8 양자화 오차가 lm_head를 거쳐 logits에 전파될 때, 여전히 argmax를 뒤집을 수 있는가?

이전 레이어 분석에서:
- Encoder L11 final_layer_norm 출력: ko_cos=0.616 (FP32 대비)
- 영어 모델도 encoder 출력에 양자화 오차가 있지만 CER 17.52%로 동작

→ **encoder 출력 오차는 lm_head float32 matmul로 상쇄 가능할 것으로 예상.** 하지만 실측 필요.

---

## 7. vocab 1900에 대한 재평가

Split model 방식이면 **vocab 1900도 재고해볼 수 있다:**

```
Split + vocab 56:
  encoder (NPU uint8) → [149,768] → lm_head(CPU fp32, 768→56) → argmax ✓

Split + vocab 1900:
  encoder (NPU uint8) → [149,768] → lm_head(CPU fp32, 768→1900) → argmax ?
  lm_head weight: 768×1900×4 = 5.8MB (여전히 작음)
  matmul: 149×768×1900 = 217M ops → ARM CPU ~200ms (추가 시간)
```

| | Split + vocab 56 | Split + vocab 1900 |
|---|---|---|
| lm_head 시간 | ~5ms | ~200ms |
| 총 추론 시간 | ~405ms | ~600ms |
| lm_head weight | 172KB | 5.8MB |
| CTC 시퀀스 길이 | 길다 (자모) | 짧다 (음절) |
| 후처리 | 자모→음절 조합 필요 | 바로 음절 출력 |
| float CER | 떨어짐 | 좋음 |

vocab 1900 + split이면:
- float CER이 좋고 (음절 단위의 장점)
- margin 문제 없고 (float32 logits)
- 대신 200ms 추가 (총 600ms, 여전히 실시간 이하)

**이게 될지는 실측해봐야 한다** — encoder 출력의 uint8 오차가 1900개 클래스 구분에 충분한 정보를 유지하는지가 관건.

---

## 8. 다른 접근법과의 비교

| 방법 | 구현 난이도 | 추가 시간 | margin 해결 | vocab 제약 |
|------|-----------|----------|------------|-----------|
| PTQ 21종 시도 | 높음 | 0ms | ✗ | 56만 간신히 |
| QAT + margin loss | 중간 (GPU 필요) | 0ms | 부분적 | 56만 |
| **Split model** | **낮음** | **5~200ms** | **✓ 완전 해결** | **56 또는 1900** |
| W8A16 (Acuity 미지원) | 불가 | — | ✓ | — |
| OpenVINO mixed (T527 아님) | 불가 | — | ✓ | — |

---

## 9. 구현 우선순위

```
[1단계] encoder-only ONNX 분리 + NB 변환 (30분)
   ↓
[2단계] vpm_run으로 encoder NB 디바이스 테스트 — 동작 확인 (10분)
   ↓
[3단계] Python에서 lm_head float32 실행 + CER 측정 (20분)
   ↓ CER 확인 — ONNX float과 비슷하면 성공
   ↓
[4단계] Android JNI에 matmul 코드 구현 (1시간)
   ↓
[5단계] vocab 1900으로도 테스트 (추가 2시간)
```

**예상 소요:** encoder 분리 + 테스트까지 1시간. JNI 구현까지 2시간.

---

## 10. FAQ

### 9.1 lm_head만 빼면 되나? L8-11도 빼야 하지 않나?

**lm_head만 빼면 된다.**

```
문제의 본질:
  L8-11 encoder → uint8 오차 누적 → hidden states에 노이즈
  lm_head(uint8) → 노이즈 낀 hidden states를 uint8 logits로 또 깎음 → margin 사망

Split으로 lm_head만 fp32로 빼면:
  L8-11 encoder → uint8 오차 누적 → hidden states에 노이즈 (여전히 있음)
  lm_head(fp32) → 노이즈 있어도 float32 logits → margin 보존 → argmax 정확
```

L8-11을 CPU로 빼면 정확도는 더 좋겠지만 속도가 치명적:

| CPU에서 실행하는 범위 | 연산량 | ARM CPU 시간 | 총 추론 시간 |
|---|---|---|---|
| **lm_head만** | 6.4M ops | **~5ms** | **~405ms** |
| L11 + lm_head | ~10B ops | ~800ms | ~1200ms |
| L8-11 + lm_head | ~40B ops | ~3000ms | ~3400ms |

lm_head만 빼도 margin 문제는 해결되고, 속도 페널티가 5ms뿐이라 최적.

### 9.2 lm_head가 "activation"인가?

아니다. 용어 정리:

```
weight:     모델 파라미터 (학습된 값, 고정). lm_head.weight shape=[56,768]
activation: 입력 데이터가 레이어를 통과하면서 나오는 중간 값 (매번 다름)
lm_head:    마지막 Linear 레이어. weight를 가지고 있고, activation(hidden states)을
            받아서 logits를 출력.

T527의 W8A8 문제:
  lm_head의 weight(uint8) × activation(uint8) → logits(uint8) → margin 사망

Split의 해결:
  lm_head의 weight(fp32) × activation(fp32) → logits(fp32) → margin 보존
```

### 9.3 검증 순서: 왜 영어 모델부터?

```
1번: 영어 vocab 32 (이미 전체 uint8에서 CER 17.52%)
  → Split하면 CER ≈ 9.74% (ONNX float)에 근접 예상
  → 비교 기준이 이미 있어서 Split 효과를 깨끗하게 검증 가능

2번: 한국어 vocab 56
  → 영어에서 검증되면 바로 적용

3번: 한국어 vocab 1900
  → 56에서 되면 1900도 시도 (float32 logits이면 vocab 제약 완화)
```

영어로 먼저 하면 "Split이 정말 margin 문제를 해결하는지" 깨끗하게 증명 가능.
한국어로 바로 가면 encoder 양자화 오차 문제와 섞여서 원인 구분이 어려움.

---

## 11. 관련 선행 사례

- **OpenVINO NNCF**: 94개 연산 중 3개를 FP32로 되돌려 WER 복구 — 같은 원리
- **LLM W4A16 (GPTQ/AWQ)**: weight만 양자화, activation FP16 유지 — 효과 동일
- **Split inference**: 모바일 AI에서 무거운 부분은 NPU, 가벼운 부분은 CPU로 분담하는 패턴 — 일반적 기법
- **Amazon Accelerator-Aware Training**: NPU 특성에 맞춘 학습 — 비슷한 동기

---

## 10. 핵심 파일 (구현 시)

| 파일 | 역할 | 상태 |
|------|------|------|
| `wav2vec2_ko_3s.onnx` | 원본 ONNX | 있음 |
| `split_onnx.py` | ONNX 분리 스크립트 | **작성 필요** |
| `encoder_only.onnx` | encoder-only ONNX | 생성 예정 |
| `lm_head_weight.npy` | lm_head weight (172KB) | 생성 예정 |
| `lm_head_bias.npy` | lm_head bias (224B) | 생성 예정 |
| `eval_split_model.py` | split model CER 측정 | **작성 필요** |
| `awwav2vecsdk.c` | Android JNI (matmul 추가) | **수정 필요** |
