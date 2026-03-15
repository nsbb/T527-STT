# Wav2Vec2 T527 NPU 변환 및 성능 평가 보고서

## 목차

1. [개요](#1-개요)
2. [Wav2Vec2 모델 구조](#2-wav2vec2-모델-구조)
3. [영어 모델 변환 과정](#3-영어-모델-변환-과정)
4. [발견된 버그 4건과 해결 과정](#4-발견된-버그-4건과-해결-과정)
5. [양자화 방식별 시도 및 실패 기록](#5-양자화-방식별-시도-및-실패-기록)
6. [영어 모델 CER 평가 결과](#6-영어-모델-cer-평가-결과)
7. [한국어 모델(XLS-R-300M) 변환 시도](#7-한국어-모델xls-r-300m-변환-시도)
8. [한국어 대안 모델(wav2vec2-base-korean)](#8-한국어-대안-모델wav2vec2-base-korean)
9. [T527 NPU STT 모델 종합 비교](#9-t527-npu-stt-모델-종합-비교)
10. [재현 방법](#10-재현-방법)

---

## 1. 개요

본 문서는 Meta의 wav2vec 2.0 음성 인식 모델을 Allwinner T527 SoC 탑재 Vivante NPU에서 구동하기 위해 수행한 전체 작업을 기술한다.

**핵심 결과:**

| 항목 | 영어 (base-960h) | 한국어 (XLS-R-300M) |
|------|-------------------|---------------------|
| 원본 모델 | facebook/wav2vec2-base-960h | facebook/wav2vec2-xls-r-300m | Kkonjeong/wav2vec2-base-korean |
| 파라미터 수 | 94.4M (12 layers) | 300M (24 layers) | 94.4M (12 layers) |
| ONNX 크기 | 362MB | 1.27GB | 378MB |
| Vocab | 32 (영어 알파벳) | 2,617 (한글 음절) | 56 (한글 자모) |
| NB 크기 (uint8) | 87MB | 249MB | 75.5MB |
| NPU CER | **17.52%** | ALL PAD (완전 실패) | ALL PAD / 가비지 (실패) |
| NPU 추론 시간 | **715ms** / 5초 | 1,098ms / 3초 | **425ms** / 3초 |
| RTF | **0.143** | 0.366 | **0.142** |
| 양자화 시도 | 2종 (uint8 성공) | 1종 (uint8 실패) | **43종+ (Phase 1-2 실패, Phase 3-4 미테스트)** |

> **RTF (Real-Time Factor)** = 처리시간 ÷ 오디오길이. 1 미만이면 실시간보다 빠른 것.
> T527 RTF 0.143은 RK3588 fp16 (0.15)과 유사하며, RTX A6000 fp16 (0.007)의 약 20배 느림.

---

## 2. Wav2Vec2 모델 구조

Wav2Vec 2.0은 Meta(구 Facebook)가 2020년 발표한 자기지도학습 기반 음성 인식 모델이다 (arXiv:2006.11477).

### 아키텍처

```
[Raw Audio] → [CNN Feature Extractor] → [Transformer Encoder] → [Linear Head] → [CTC Decode]

입력: 16kHz mono PCM waveform [1, T]
  ↓
CNN 7층: Conv1D stride=[5,2,2,2,2,2,2] → 총 stride 320
  - 16kHz 기준 20ms당 1 frame 출력
  - 5초(80,000 samples) → 249 frames, 3초(48,000) → 149 frames
  ↓
Transformer Encoder:
  - base: 12 layers, hidden 768, attention heads 12 → 94.4M params
  - large/XLS-R-300M: 24 layers, hidden 1024, heads 16 → 300M params
  ↓
Linear Head: hidden → vocab_size
  - 영어 base-960h: 32 (A-Z + space + 특수토큰)
  - 한국어 XLS-R-300M: 2,617 (가~힝 한글 음절)
  ↓
CTC Greedy Decode: argmax → 연속 중복 제거 → blank 제거 → 텍스트
```

### 전처리

Wav2Vec2는 **별도의 mel-spectrogram 변환이 필요 없다**. 원시 PCM waveform을 그대로 입력한다.
CNN Feature Extractor가 모델 내부에서 특징 추출을 수행하므로, 입력 전처리는 다음 한 단계뿐이다:

```
audio = (audio - mean) / std   # 정규화 (feature_extractor가 수행)
```

이는 mel-spectrogram → 양자화 → NPU 전달이 필요한 CitriNet과 대비되는 장점이다.

---

## 3. 영어 모델 변환 과정

### 3.1 전체 파이프라인

```
HuggingFace 모델 (PyTorch)
  → torch.onnx.export() → ONNX (362MB, dynamic shape)
  → ONNX shape 고정 → [1, 80000] → [1, 249, 32]
  → Acuity Toolkit import → FP32 내부 표현
  → Acuity quantize (uint8, 51 calibration samples)
  → Docker에서 export → network_binary.nb (87MB)
  → T527 NPU에서 추론
```

### 3.2 ONNX 변환

```python
# PyTorch → ONNX (opset 14)
torch.onnx.export(model, dummy_input,
    input_names=["input_values"], output_names=["logits"],
    opset_version=14, do_constant_folding=True)

# 동적 shape → 고정 shape (Acuity가 고정 shape만 지원)
# 입력: [1, 80000], 출력: [1, 249, 32]
```

### 3.3 Acuity 양자화 및 NB 변환

```bash
# 1. Import (ONNX → Acuity 내부 표현)
pegasus import onnx --model wav2vec2_base_960h_5s.onnx --output-model model.json --output-data model.data

# 2. Quantize (FP32 → uint8)
pegasus quantize --quantizer asymmetric_affine --qtype uint8 \
  --algorithm moving_average --moving-average-weight 0.004 \
  --with-input-meta inputmeta.yml   # reverse_channel=false 필수!

# 3. Export (NB 생성, Docker 환경에서)
pegasus export ovxlib --dtype quantized --pack-nbg-unify \
  --optimize VIP9000NANOSI_PLUS_PID0X10000016   # T527 NPU target
```

### 3.4 양자화 파라미터

| 항목 | Scale | Zero Point | 범위 |
|------|-------|------------|------|
| **입력** (uint8) | 0.002860 | 137 | [-0.393, 0.336] |
| **출력** (uint8) | 0.150270 | 186 | [-27.88, 10.44] |

이 파라미터들은 `nbg_meta.json`에 기록되어 있으며, JNI 코드에서 입력 양자화 및 출력 역양자화에 사용된다.

---

## 4. 발견된 버그 4건과 해결 과정

최초 NB를 T527에서 실행했을 때 **출력이 전부 공백(blank)이거나 의미 없는 문자열**이었다.
원인은 NPU나 모델 자체가 아니라, **변환·통합 과정의 코드 버그 4건**이었다.
각 버그를 하나씩 수정하면서 단계적으로 정상 출력을 얻어냈다.

### 버그 1: float32 포인터를 uint8로 캐스팅 (JNI 핵심 버그)

**파일:** `awaiasr_2/.../jni/wav2vec/awwav2vecsdk.c` 라인 156

**증상:** NPU 출력이 완전히 쓰레기 값 — 텍스트를 전혀 인식하지 못함.

**원인:** NPU는 uint8(0~255) 정수 입력을 기대하는데, float32 원시 오디오 데이터의 바이트를 그대로 전달하고 있었다.

```c
// ❌ 버그 코드 (수정 전)
float *processedAudio = ...;  // [-1.0 ~ +1.0] 범위의 오디오 데이터
unsigned char *input_buffers[1] = {(unsigned char*)processedAudio};
// → float32의 4바이트가 그대로 uint8 4개로 해석됨
// → 예: float 0.5 = 0x3F000000 → uint8로는 [63, 0, 0, 0] = 완전 엉뚱한 값

// ✅ 수정 후
unsigned char *quantized_input = malloc(MODEL_INPUT_LENGTH);
for (int i = 0; i < MODEL_INPUT_LENGTH; i++) {
    float val = processedAudio[i];
    int q = (int)roundf(val / INPUT_SCALE) + INPUT_ZP;  // 양자화 공식
    quantized_input[i] = (unsigned char)(q < 0 ? 0 : q > 255 ? 255 : q);
}
unsigned char *input_buffers[1] = {quantized_input};  // 양자화된 버퍼 사용
```

**설명:** `quantized_input` 버퍼는 이미 코드에 올바르게 구현되어 있었다 (라인 139~148에서 양자화 수행). 하지만 정작 NPU에 전달하는 라인 156에서 양자화된 버퍼 대신 원래의 float32 포인터를 그대로 넘기고 있었다. **만들어놓고 사용하지 않는** 전형적인 실수였다.

**비유:** 외국어 번역문을 준비해놓고, 정작 발표할 때 원본 한국어 원고를 읽는 것과 같다. 청중(NPU)은 번역문(uint8)을 기대하는데 원본(float32)을 듣게 되니 전혀 이해하지 못한다.

---

### 버그 2: reverse_channel=true 설정 오류 (양자화 캘리브레이션 오염)

**파일:** `wav2vec2_base_960h_5s_inputmeta.yml`

**증상:** 코드 흐름은 정상인데 NPU 출력이 모두 blank.

**원인:** Acuity Toolkit의 `inputmeta.yml`에서 `reverse_channel: true`로 설정되어 있으면, 캘리브레이션 데이터의 채널 순서를 뒤집는다. 이미지(BGR↔RGB) 변환 모델에서는 필요하지만, **음성 모델에서는 1차원 waveform이므로 뒤집으면 데이터가 완전히 망가진다**.

```yaml
# ❌ 수정 전
preprocess:
  reverse_channel: true   # 이미지 모델에서 복사해온 기본값

# ✅ 수정 후
preprocess:
  reverse_channel: false  # 음성은 채널 반전 불필요
```

**설명:** 이 설정은 양자화 과정에서 캘리브레이션 데이터를 읽어올 때 적용된다. `reverse_channel: true`이면 입력 데이터의 값 분포가 왜곡되어 양자화 범위(scale/zero_point)가 잘못 계산된다. 결과적으로 NPU가 정상 데이터를 받아도 내부 연산이 틀어져 blank만 출력하게 된다.

**비유:** 악보를 거꾸로 복사해서 연주자에게 주는 것과 같다. 연주자(NPU)는 자기가 받은 악보대로 성실히 연주하지만, 관객은 엉뚱한 음악을 듣게 된다.

---

### 버그 3: 출력 텐서 레이아웃 오류 ([seq,vocab] vs [vocab,seq])

**파일:** `awaiasr_2/.../jni/wav2vec/wav2vec_postprocess.cpp`

**증상:** NPU가 값을 출력하긴 하지만, 디코딩 결과가 랜덤한 문자열.

**원인:** NPU 출력 텐서의 차원 순서를 반대로 읽고 있었다.

```
NPU 출력: [1, 249, 32] = [batch, 시간프레임, vocab]
  - 올바른 해석: 249개 시간 프레임 × 32개 토큰 확률
  - 잘못된 해석: 32개 × 249개 → 완전히 다른 위치의 값을 읽게 됨
```

```cpp
// ❌ 수정 전: [vocab, seq] 순서로 접근
for (int t = 0; t < SEQ_LEN; t++) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        logits[t][v] = output[v * SEQ_LEN + t];  // 전치된 인덱싱
    }
}

// ✅ 수정 후: [seq, vocab] 순서로 접근
for (int t = 0; t < SEQ_LEN; t++) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        logits[t][v] = output[t * VOCAB_SIZE + v];  // 올바른 인덱싱
    }
}
```

**설명:** 2D 배열에서 행(row)과 열(column)을 바꿔 읽으면 완전히 다른 값을 참조하게 된다. 시간 프레임 0에서의 각 토큰 확률을 읽어야 하는데, 토큰 0에서의 각 시간 프레임 값을 읽게 되면 argmax 결과가 완전히 달라진다.

**비유:** Excel에서 행과 열을 바꿔서 읽는 것과 같다. A열의 1~249행을 읽어야 하는데, 1행의 A~IQ열을 읽으면 전혀 다른 데이터가 나온다.

---

### 버그 4: 모델 입출력 차원 불일치

**파일:** `awaiasr_2/.../jni/wav2vec/awwav2vecsdk.c` 라인 16~17

**증상:** 배열 범위 초과(buffer overflow) 또는 불완전한 출력.

**원인:** 코드의 입출력 크기 상수가 실제 NB 모델의 사양과 맞지 않았다.

```c
// ❌ 수정 전 (다른 모델의 값이 남아있었음)
#define MODEL_INPUT_LENGTH 320000   // 20초 모델용
#define MODEL_OUTPUT_SEQ_LEN 999    // 20초 모델 출력 길이

// ✅ 수정 후 (5초 uint8 모델에 맞춤)
#define MODEL_INPUT_LENGTH 80000    // 5초 × 16kHz = 80,000 samples
#define MODEL_OUTPUT_SEQ_LEN 249    // CNN stride 320 → 80000/320 = 250 → 실제 249
```

**설명:** 입력 길이가 잘못되면 NPU에 전달하는 버퍼 크기가 맞지 않아 쓰레기 데이터가 포함되거나, 출력을 읽을 때 잘못된 범위를 참조하게 된다.

---

### 버그 수정 요약

| # | 버그 | 영향 | 수정 내용 |
|---|------|------|-----------|
| 1 | float32→uint8 캐스팅 | 입력 데이터 완전 손상 | `quantized_input` 사용 |
| 2 | reverse_channel=true | 양자화 범위 오염 | `false`로 변경 |
| 3 | 텐서 레이아웃 반전 | 출력 디코딩 랜덤화 | [seq,vocab] 순서로 수정 |
| 4 | 차원 상수 불일치 | 버퍼 오버플로우 | 80000/249로 수정 |

4개 버그를 모두 수정한 후, **CER 17.52%**의 정상적인 영어 음성 인식 결과를 얻었다.

---

## 5. 양자화 방식별 시도 및 실패 기록

T527 NPU는 FP32 연산을 직접 지원하지 않으므로, 모델을 양자화(quantization)하여 정수 연산으로 변환해야 한다. 다양한 양자화 방식을 시도하였으며, **uint8 asymmetric_affine만이 유일하게 성공**했다.

| # | 양자화 방식 | 결과 | 실패 원인 |
|---|------------|------|-----------|
| 1 | **uint8 asymmetric_affine** | **성공 (CER 17.52%)** | — |
| 2 | bf16 (bfloat16) | 실패 | `gen_nbg` 바이너리에서 segfault. NB 파일 0바이트 생성. NB 크기 ~181MB로 NPU SRAM 한도 초과 가능성 |
| 3 | PCQ int8 (Per-Channel) | 실패 | `gen_nbg`에서 "Reshape tensor error" segfault. Per-channel 양자화는 T527 NPU 드라이버 호환성 문제 |
| 4 | int16 | 실패 | NB 생성은 되나, T527 NPU에서 실행 시 hang (응답 없음). 드라이버가 int16 완전 지원하지 않음 |
| 5 | fp32 | 실패 | SRAM overflow — 362MB 모델이 NPU 내부 메모리 한도 초과 |
| 6 | hybrid (혼합 정밀도) | 실패 | `--hybrid` 플래그가 실제로 정밀도를 변경하지 않음. Acuity 버전 한계 |
| 7 | MLE (Maximum Likelihood) | 실패 | Acuity 내부 `AttributeError` 크래시. 이 알고리즘은 현재 Transformer 모델 미지원 |
| 8 | KL divergence | 부분 성공 | 양자화 자체는 완료되나 NB 미생성. 시뮬레이션에서 moving_average와 유사한 품질 |

**결론:** T527 NPU (Vivante VIP9000NANOSI_PLUS)에서 gen_nbg가 지원하는 양자화:
- **asymmetric_affine uint8/int8** — NB 생성 성공, 추론 동작
- **dynamic_fixed_point int16** — opset12+sim 모델에서 NB 생성 성공 (opset14는 segfault)
- ❌ bf16, PCQ (perchannel_symmetric), symmetric_affine — 모두 gen_nbg 실패

---

## 6. 영어 모델 CER 평가 결과

### 6.1 평가 환경

- **모델:** facebook/wav2vec2-base-960h (영어, 94.4M params)
- **NB:** uint8, 87MB, 5초 입력 (80,000 samples @ 16kHz)
- **테스트셋:** LibriSpeech test-clean, speaker 6930, 50개 샘플
- **오디오 길이:** 1.8초 ~ 7.4초 (5초로 패딩/절단)
- **평가 방식:** vpm_run (T527 NPU 직접 실행) → output_0.dat → CTC greedy decode → CER/WER

### 6.2 정량적 결과

| 지표 | ONNX FP32 | NPU uint8 | 양자화 열화 |
|------|-----------|-----------|------------|
| **CER** | **9.74%** | **17.52%** | **+7.78%p** |
| **WER** | — | **27.38%** | — |
| **정답 일치** | 25/50 (50%) | 6/50 (12%) | — |
| **평균 추론 시간** | — | **715ms** | — |
| **RTF** | — | **0.143** | — |

> - **CER (Character Error Rate):** 문자 단위 오류율. 낮을수록 좋음.
> - **WER (Word Error Rate):** 단어 단위 오류율.
> - **양자화 열화 +7.78%p:** FP32 모델의 고유 오류(9.74%)에 양자화로 인한 추가 오류(7.78%p)가 더해져 최종 17.52%.

### 6.3 샘플별 결과 (발췌)

| # | 정답 (Ground Truth) | NPU 출력 | CER |
|---|---------------------|----------|-----|
| 7 | I AM CONVINCED OF WHAT I SAY SAID THE COUNT | I AM CONVINCED OF WHAT I SAY SAID THE COUNT | **0.0%** |
| 25 | THIS THOUGHT HOWEVER DID NOT ENTER THE HEADS... | THIS THOUGHT HOWEVER DID NOT ENTER THE HEADS... | **0.0%** |
| 37 | NO SOUND BROKE THE STILLNESS OF THE NIGHT | NO SOUND BROKE THE STILLNESS OF THE NIGHT | **0.0%** |
| 28 | NOW LET'S DUST THE FURNITURE AND PICTURES | ALLITS DOS TH FIRNTURN PICTURE | 40.0% |
| 31 | THEN SHE SUDDENLY REMARKED | BER ESMAK | 73.9% |
| 8 | IT IS ANNOYANCE THEN | S N | 88.2% |

### 6.4 오류 분석

- **짧은 발화(2초 미만)에서 CER 급등:** 5초 입력에 맞추기 위해 나머지를 0으로 채우면(zero-padding), 모델이 무음 구간의 컨텍스트에 영향받아 인식률이 크게 떨어진다. 샘플 8번("IT IS ANNOYANCE THEN", 1.8초)의 CER 88.2%가 대표적.
- **3초 이상 발화에서 안정적:** 대부분 CER 0~20% 범위.
- **양자화 열화의 주 원인:** Transformer의 attention score 분포가 uint8의 256단계 해상도로 충분히 표현되지 않아, softmax 이후 확률 분포가 왜곡됨.

---

## 7. 한국어 모델(XLS-R-300M) 변환 시도

### 7.1 모델 정보

| 항목 | 값 |
|------|-----|
| 원본 | facebook/wav2vec2-xls-r-300m (한국어 CTC fine-tune) |
| 파라미터 | 300M (영어 base의 3.2배) |
| Transformer | 24 layers, hidden 1024, heads 16 |
| Vocab | 2,617개 (한글 음절: 가~힝) |
| ONNX 크기 | 1.27GB |
| 입력 | 3초 (48,000 samples) → 149 frames 출력 |

### 7.2 변환 파이프라인

```
기존 ONNX (1.2GB, dynamic shape)
  → make_fixed_onnx.py: shape 고정 [1,48000] → [1,149,2617]
  → Acuity import: 성공 (FP32 시뮬레이션 max diff 0.0026 vs ONNX — 거의 완벽)
  → Acuity quantize (uint8, 200 calibration samples): 완료
  → Docker export: NB 생성 성공 (249MB)
  → T527 NPU 실행: 1,098ms에 실행 완료
  → 출력 디코딩: ❌ ALL PAD (모든 프레임이 pad 토큰)
```

### 7.3 실패 원인 분석

**최종 진단: 24-layer Transformer + 2,617 vocab에서 uint8 양자화 오류 누적**

NPU 출력이 모두 `<pad>` 토큰인 이유를 단계별로 추적한 결과:

1. **Acuity FP32 시뮬레이션은 정상** — ONNX와 max diff 0.0026, argmax 149/149 일치.
   → 모델 import는 문제 없음.

2. **Acuity uint8 시뮬레이션도 ALL PAD** — NPU가 아닌 CPU에서 양자화된 모델을 실행해도 동일 결과.
   → NPU 하드웨어 문제가 아니라, **양자화 자체의 문제**.

3. **로짓(logit) 값 비교:**

   ```
   ONNX FP32 (정상):
     텍스트 위치 → 올바른 토큰 logit ~26, pad logit ~5  → argmax = 올바른 토큰 ✅

   uint8 양자화 후:
     텍스트 위치 → 올바른 토큰 logit ~20, pad logit ~23 → argmax = pad ❌
   ```

4. **원인:** uint8의 해상도(scale=0.187, 즉 한 단계당 0.187)로는 26 vs 5의 차이(21)를 보존할 수 있지만, 24개 Transformer layer를 거치면서 오류가 누적되어 실질적 차이가 줄어든다. 2,617개 vocab 중 pad 토큰의 bias가 상대적으로 강화되어, 결국 **모든 프레임에서 pad가 이긴다**.

### 7.4 추가 시도 결과 (총 8종)

**기본 양자화:**

| 시도 | Calibration | 결과 |
|------|-------------|------|
| uint8 (50 samples) | 한국어 WAV 50개 | ALL PAD |
| uint8 (200 samples, moving_avg) | 한국어+영어+증강 200개 | ALL PAD |
| PCQ int8 (200 samples) | 동일 200개 | ALL PAD |

**Hybrid 양자화 (uint8 + int16/float32 혼합):**

| 전략 | int16/f32 대상 | 변경 레이어 수 | 결과 |
|------|----------------|---------------|------|
| A (softmax+head i16) | softmax, lm_head | 25 | ALL PAD |
| C (softmax+LN+head i16) | softmax, LayerNorm, lm_head | 82 | ALL PAD |
| D (softmax+head f32) | softmax, lm_head | 25 | ALL PAD |

**PAD bias 조정 (nopad 트릭):**

| 전략 | PAD bias 조정 | 양자화 | non-PAD | FP32 일치 | 결과 |
|------|-------------|--------|---------|----------|------|
| E (nopad30) | bias[2616]-=30 | uint8 | 149/149 | 13% | "네요" 반복 |
| F (nopad+hybrid) | nopad + FC12-23 i16 | hybrid | 149/149 | 10% | "네요" 반복 |
| H (nopad+대규모hybrid) | nopad + 모든FC+SM+LN | hybrid | 149/149 | 9.4% | "네요" 반복 |

**int16 전체 양자화:**

| 전략 | 결과 | FP32 일치 | 비고 |
|------|------|----------|------|
| int16 (nopad) | **149/149 FP32 일치 (100%)** | **100%** | T527 NPU에서 hang (실행 불가) |

**12-layer pruning:** ONNX GraphSurgeon으로 24→12 layer 축소 시도 → FP32에서도 3/149만 non-PAD (모델 품질 파괴됨).

3번의 기본 양자화 모두 ALL PAD → **캘리브레이션 양이 아니라 모델 크기와 Transformer 깊이가 근본 원인**.

### 7.5 결론

| 모델 규모 | T527 uint8 양자화 |
|-----------|-------------------|
| base (94M, 12 layers) | **성공** (CER 17.52%) |
| large/XLS-R (300M, 24 layers) | **실패** (ALL PAD) |

**교훈:** T527 NPU의 uint8 양자화는 **~100M 파라미터, ~12 Transformer layers**가 실질적 상한선이다. 이를 초과하면 양자화 오류 누적으로 출력이 무의미해진다.

> 한국어 base-korean 모델에서 레이어 pruning 시도 결과: 6L, 8L, 10L 모두 FP32에서도 쓰레기 출력. 한국어 wav2vec2-base-korean은 **12L 전부 필요** (영어와 달리 레이어 축소 불가).

또한 vocab 크기도 영향을 미친다:
- 영어 base-960h: vocab 32개 → 각 토큰의 logit 차이가 충분히 큼
- 한국어 XLS-R-300M: vocab 2,617개 → 토큰 간 logit 차이가 작아 양자화에 취약

---

## 8. 한국어 대안 모델(wav2vec2-base-korean)

XLS-R-300M의 실패를 바탕으로, 동일한 base 아키텍처(12 layers, 94M params)의 한국어 모델을 대안으로 선정했다.

### 8.1 모델 비교

| 항목 | XLS-R-300M (실패) | base-korean (대안) |
|------|--------------------|--------------------|
| HuggingFace | (한국어 fine-tune) | Kkonjeong/wav2vec2-base-korean |
| 기반 모델 | facebook/wav2vec2-xls-r-300m | facebook/wav2vec2-base |
| 파라미터 | 300M | **94.4M** |
| Layers | 24 | **12** |
| Hidden | 1024 | **768** |
| Vocab | 2,617 (음절: 가~힝) | **56 (자모: ㄱ~ㅎ, ㅏ~ㅣ)** |
| 학습 데이터 | — | Zeroth-Korean |
| 보고된 CER | — | **7.3%** (FP32) |
| 예상 NB 크기 | 249MB | **~87MB** |

### 8.2 성공 가능성 근거

1. **동일 아키텍처(base, 12L)의 영어 모델이 uint8 CER 17.52%로 성공** — 같은 구조이므로 양자화 특성이 유사할 것으로 예상.
2. **Vocab 56개(자모)** — 2,617개(음절)보다 토큰 간 logit 구분이 훨씬 용이하여 양자화에 유리.
3. **NB 크기 ~87MB** — T527 NPU가 이미 처리한 영어 모델과 동일 수준.

### 8.3 자모 vs 음절 방식

| 방식 | Vocab 크기 | 예시 ("안녕") | 양자화 영향 |
|------|-----------|--------------|------------|
| 음절 | 2,617 | 안 녕 (2토큰) | 불리 — 토큰 간 logit 차이 작음 |
| 자모 | 56 | ㅇ ㅏ ㄴ ㄴ ㅕ ㅇ (6토큰) | **유리** — 토큰 간 logit 차이 큼 |

자모 방식은 출력 토큰 수가 많아지지만, 각 프레임에서 56개 중 하나를 고르는 것이 2,617개 중 하나를 고르는 것보다 양자화 후에도 정확하다.

### 8.4 변환 결과

모델 다운로드 및 ONNX 변환, Acuity import, 다양한 양자화 시도, NB export까지 **모두 성공**:

```
model.safetensors (378MB) 다운로드
  → download_and_convert.py: ONNX export [1,48000] → [1,149,56]
  → Acuity import: 494 layers, input: input_values_568
  → 양자화: 14종 시도 (uint8/PCQ/bf16/int16/hybrid/nopad/MA/KL)
  → Docker export: NB 생성 성공 (75.5MB, T527 optimize flag 필수)
  → T527 NPU 실행: 425ms (RTF 0.142)
  → 출력 디코딩: ❌ ALL PAD (기본) / garbled (nopad 적용 시)
```

**중요 발견:** Acuity 시뮬레이션에서 100% FP32 일치하는 양자화도, 실제 T527 NPU에서는 ALL PAD 또는 garbled 출력. **시뮬레이션과 NPU 간 불일치가 근본 원인.**

### 8.5 양자화 방식별 시도 (총 26종+)

기본 uint8부터 moving_average, KL divergence, PCQ, hybrid, nopad, 모델 수술(GELU→ReLU, 레이어 pruning), 온도 스케일링 등 총 26가지 이상의 양자화/모델 변형 전략을 시도했다.

**FP32 기준 출력:** `ㄸㅓㄴㅌㅐㅇㅣㅂㅇㅡㄹ` (테스트 오디오에 대한 정답)

**시뮬레이션 결과 (Acuity CPU 추론):**

| # | 전략 | 캘리브레이션 | non-PAD | FP32 일치율 | 비고 |
|---|------|-------------|---------|------------|------|
| 1 | uint8 (기본, normal) | 1 sample | 0/149 | ~90% | ALL PAD |
| 2 | PCQ int8 (normal) | 1 sample | 1/149 | 90.6% | ALL PAD |
| 3 | Hybrid (softmax i16) | 1 sample | 0/149 | 91.3% | ALL PAD |
| 4 | nopad5 uint8 | 1 sample | 77/149 | 49.0% | ㅇㄱㅏㅇ... |
| 5 | nopad5 PCQ | 1 sample | 53/149 | 65.8% | ㅇㅇㅔㅇㅣ... |
| 6 | nopad5 Hybrid | 1 sample | 77/149 | 50.3% | ㅇㅏㅇ ㅇㄱㅔㅇ... |
| 7 | nopad8 PCQ | 1 sample | 149/149 | 1.3% | 가비지 |
| 8 | nopad15 uint8 | 1 sample | 149/149 | 38.3% | ㅇ ㅇ ㅇ... |
| **9** | **int16** | 1 sample | **14/149** | **98.7%** | **ㄸㅓㄴㅌㅔ ㅇㅣㅂㅇㅡㄹ** |
| 10 | uint8 MA | 200 samples | 13/149 | **~100%** | **시뮬레이션에서 FP32 완벽 일치** |
| 11 | uint8 KL | 200 samples | 13/149 | **~100%** | **시뮬레이션에서 FP32 완벽 일치** |
| 12 | bf16 | — | 13/149 | **~100%** | gen_nbg segfault (NB 생성 불가) |
| 13 | nopad5 MA | 200 samples | 14/149 | ~100% | sim 정확 |
| 14 | nopad10 MA | 200 samples | 13/149 | ~100% | sim 정확 |

**핵심 발견: 시뮬레이션 vs NPU 불일치**

MA/KL 200샘플 캘리브레이션 결과, Acuity 시뮬레이션에서는 모든 양자화 방식이 FP32와 100% 일치하는 출력을 냈다. **그러나 실제 T527 NPU에서는 전혀 다른 결과가 나온다.** 시뮬레이션 결과를 신뢰할 수 없다는 것이 핵심 교훈.

**T527 NPU 실측 결과 (Phase 1):**

| # | 전략 | non-PAD | FP32 상관계수 | NPU 출력 |
|---|------|---------|--------------|----------|
| 1 | uint8 MA | 0/149 | — | ALL PAD |
| 2 | uint8 KL | 0/149 | — | ALL PAD |
| 3 | nopad5 MA | 64/149 | r≈0.2 | garbled |
| **4** | **nopad10 MA** | **149/149** | **r=0.560** | **81% ㅇ 토큰** |
| 5 | nopad10 KL | 149/149 | r=0.561 | 50% ㅇ 토큰 |
| 6 | nopad15 MA | 149/149 | r=0.292 | 82% ㅇ 토큰 |
| 7 | nopad20 MA | 149/149 | r=0.050 | garbled |
| 8 | nopad30 MA | 149/149 | r=-0.258 | garbled (역상관) |

**nopad 강도와 상관계수의 관계:** nopad10이 최적점(r=0.56). 이후 증가할수록 악화, nopad30에서는 역상관(-0.26).

**T527 NPU 실측 결과 (Phase 2: 추가 전략):**

| # | 전략 | NB 크기 | NPU 결과 |
|---|------|---------|----------|
| 9 | temp3×nopad10 (logit 3배 증폭) | 72MB | PAD 4/149, vocab 순환 패턴 (garbled) |
| 10 | dynamic_fixed_point (DFP) | 54MB | **status=-1** (NPU 미지원 포맷) |
| 11 | symmetric_affine int8 | — | gen_nbg Fatal error 65280 |
| 12 | MLE (Minimum Layer Error) | — | Acuity 내부 오류 |
| 13 | hybrid (uint8+int16 attention) | 72MB | NB 생성 성공 (디바이스 미연결로 미테스트) |
| 14 | algorithm=normal (min/max) | 72MB | NB 생성 성공 (미테스트) |
| 15 | GELU→ReLU 치환 | 71MB | NB 생성 성공 (미테스트) |
| 16 | 12→6 레이어 pruning | **38MB** | NB 생성 성공 (미테스트) |
| 17 | CNN-only (Transformer 제거) | **3MB** | NB 생성 성공 (미테스트) |
| 18 | temp5×nopad10 (5배 증폭) | 72MB | NB 생성 성공 (미테스트) |

> 13-18번은 NB 생성까지 성공했으나, NPU 테스트 중 DFP NB 실행 실패로 디바이스가 FEL 모드에 진입하여 물리적 리셋 필요. 추후 테스트 예정.

nopad10_ma는 모든 프레임이 non-PAD이나, 81% (121/149)가 ㅇ 토큰으로 의미 있는 텍스트 불가. FP32 시뮬레이션과의 상관계수 r=0.56 (약한 양의 상관)으로, NPU가 "뭔가를" 계산하고 있지만 정확도가 극도로 낮음.

**"nopad"란?** ONNX 모델의 lm_head bias에서 PAD 토큰(id=53)의 값을 일정량(5, 10, 15 등) 감소시켜, 양자화 후에도 PAD가 과도하게 우세하지 않도록 하는 트릭. FP32에서는 speech 프레임에서 PAD logit ~0-3 vs text logit ~10이므로 영향 없으나, uint8에서는 양자화 오류로 PAD logit이 인위적으로 높아져 ALL PAD가 되는 문제를 완화한다.

### 8.6 PAD 우세 현상 분석

FP32 vs uint8 로짓 비교:

```
                    FP32 (정확)                uint8 (왜곡)
                    PAD     text               PAD     text
무음 프레임(0):    12.26   0.37  → PAD ✅     9.52    2.48  → PAD ✅
무음 프레임(50):   12.29   0.50  → PAD ✅     9.31    2.90  → PAD ✅
음성 프레임(100):   3.03  10.28  → text ✅    5.59    2.28  → PAD ❌
음성 프레임(148):   0.05  11.17  → text ✅    5.66    2.14  → PAD ❌
```

FP32에서는 무음/음성 프레임의 PAD-text 차이가 명확(11~12 vs -7~-11)하지만, uint8 양자화 후에는 12 Transformer layer의 오류 누적으로 로짓 분포가 왜곡되어, **모든 프레임에서 PAD가 우세**해진다.

### 8.7 int16 양자화 결과

int16(dynamic_fixed_point, 16bit)은 **98.7% FP32 일치**로 사실상 완벽한 양자화 정확도를 보여주나:

- **NB 생성**: vsimulator에서 int16 GPU shader 컴파일 오류 발생 (instance_norm_I16, gemm_I16I16 등). VIV_VX_ENABLE_SHADER_PATH 설정으로 해결 후 gen_nbg 실행 시도 중이나, **CPU 시뮬레이터 속도가 극도로 느림** (20분+ 실행 중).
- **T527 NPU 호환성**: XLS-R-300M int16 NB의 경우 T527에서 NPU hang 발생. base-korean(12L)은 규모가 작아 동작 가능성이 있으나 미확인.

### 8.8 T527 NPU 실측 결과

총 20종의 NB를 생성하고, 그 중 10종을 T527 NPU에서 실제 추론했다. 모든 NB는 Docker 환경에서 `--optimize VIP9000NANOSI_PLUS_PID0X10000016` (T527 chip ID 0x10000016) 플래그로 생성.

**NPU 테스트 완료:**

| NB 종류 | NB 크기 | 추론 시간 | 출력 | FP32 상관계수 |
|---------|---------|----------|------|--------------|
| uint8 MA | 72MB | 425ms | ALL PAD | — |
| uint8 KL | 72MB | 425ms | ALL PAD | — |
| nopad5 MA | 72MB | 425ms | 64/149 non-PAD (garbled) | r≈0.2 |
| **nopad10 MA** | **72MB** | **425ms** | **149/149 non-PAD (ㅇ 81%)** | **r=0.56** |
| nopad10 KL | 72MB | 425ms | 149/149 non-PAD (ㅇ 50%) | r=0.56 |
| nopad15 MA | 72MB | 425ms | 149/149 non-PAD (ㅇ 82%) | r=0.29 |
| nopad20 MA | 72MB | 425ms | 149/149 garbled | r=0.05 |
| nopad30 MA | 72MB | 425ms | 149/149 garbled | r=-0.26 |
| temp3×nopad10 | 72MB | 425ms | PAD 4/149, vocab 순환 | — |
| DFP int8 | 54MB | — | **NPU status=-1** (미지원) | — |

**NB 생성 성공, NPU 테스트 대기 (디바이스 FEL 모드):**

| NB 종류 | NB 크기 | 특징 |
|---------|---------|------|
| hybrid (uint8+int16) | 72MB | attention 레이어 int16 |
| normal algorithm | 72MB | min/max 범위 calibration |
| GELU→ReLU | 71MB | Erf 연산 20개 제거 |
| 6-layer pruning | **38MB** | encoder 12→6 레이어 |
| CNN-only | **3MB** | Transformer 완전 제거 |
| temp5×nopad10 | 72MB | logit 5배 증폭 |

**NB 생성 실패:**

| 시도 | 실패 원인 |
|------|----------|
| bf16 (opset14) | gen_nbg segfault (NB 0 bytes) |
| bf16 (opset12+sim) | gen_nbg Fatal error 35584 |
| int16 (opset14) | gen_nbg segfault |
| PCQ (per-channel) | gen_nbg segfault |
| symmetric_affine int8 | gen_nbg Fatal error 65280 |
| MLE | Acuity 내부 오류 (AttributeError) |

**nopad10_ma 상세 분석 (한국어 음성 입력):**

```
FP32 시뮬레이션:  "ㄸㅓㄴㅌㅐㅇㅣㅂㅇㅡㄹ" (의미 있는 한국어)
NPU 출력:         "ㅇ ㅇ ㅇㅣㅇㅏㅣㅇㄴㅏㄴㅣㅇ..." (81% ㅇ 토큰)

Frame 0:  FP32 PAD=12.26 > ㅇ=0.37  →  NPU PAD=-1.87 < ㅇ=2.21  (역전)
Frame 75: FP32 ㄸ=10.28 > PAD=3.03  →  NPU ㅇ=2.21 > ㄸ=-1.02  (완전 다름)
```

PAD logit이 NPU에서 과도하게 억제(FP32 12.26 → NPU -1.87)되는 반면, ㅇ(0x11)이 비정상적으로 높아지는 현상. 이는 nopad 트릭과 NPU 양자화 오류가 결합된 결과.

### 8.9 ONNX 구조 분석 — 동적 연산 차이 발견 (Phase 3)

영어 모델(base-960h)은 NPU에서 정상 동작하고 한국어 모델(base-korean)은 실패하는 근본 원인을 찾기 위해 ONNX 그래프를 비교 분석했다.

#### ONNX 연산 비교

| 연산(Op Type) | 영어 (base-960h) | 한국어 (base-korean) | 차이 |
|--------------|------------------|---------------------|------|
| **Shape** | 1 | 37 | **+36** |
| **Gather** | 0 | 24 | **+24** |
| **Concat** | 0 | 48 | **+48** |
| **Unsqueeze** | 1 | 73 | **+72** |
| **Cast** | 0 | 24 | **+24** |
| **Slice** | 1 | 13 | **+12** |
| Reshape | 98 | 50 | -48 |
| Transpose | 63 | 51 | -12 |
| **총 노드** | **957** | **1306** | **+349** |

#### 핵심 발견

**한국어 모델은 동적 shape 연산 패턴을 사용한다:**

```
Shape → Gather → Unsqueeze → Concat → Reshape  (한국어: 24세트, 영어: 0세트)
```

- 영어 모델: attention에서 **static Reshape** (96개) 사용 — 컴파일 시 shape 확정
- 한국어 모델: attention에서 **dynamic shape 계산** (Shape→Gather→Concat→Reshape) 사용 — 런타임 shape 계산

이는 모델 ONNX export 시 사용된 transformers 라이브러리 버전 차이에서 비롯된 것으로 추정. 같은 wav2vec2 아키텍처이나 export 방식이 달라 ONNX 그래프 구조가 다르다.

#### onnxsim(ONNX Simplifier) 적용 결과

`onnx-simplifier`로 입력 크기 `[1, 48000]`을 고정하여 동적 연산을 정적으로 변환:

| 연산 | 원본 | 단순화 후 | 제거 |
|------|------|-----------|------|
| Shape | 37 | 0 | **-37** |
| Gather | 24 | 0 | **-24** |
| Concat | 48 | 0 | **-48** |
| Unsqueeze | 73 | 1 | **-72** |
| Cast | 24 | 0 | **-24** |
| Constant | 360 | 0 | **-360** (initializer로 fold) |
| **총 노드** | **1306** | **679** | **-627** |

단순화 후 한국어 모델의 연산 구성이 영어 모델과 거의 동일해짐.

#### Pegasus import 비교

Pegasus import 후 내부 그래프 구조도 차이를 보인다:

| 항목 | 원본 ONNX (동적ops) | onnxsim 후 |
|------|---------------------|-----------|
| Pegasus 총 레이어 | 494 | **465** (-29) |
| **variable 레이어** | **24** | **1** (-23) |
| permute 레이어 | 53 | 49 (-4) |
| reshape 레이어 | 199 | 197 (-2) |

**`variable` 레이어가 핵심 차이.** Pegasus는 ONNX의 Shape→Gather→Concat 패턴을 `variable` 레이어로 변환한다. 이 `variable` 레이어는 런타임에 텐서 shape을 동적으로 계산하는 역할을 하며, T527 NPU의 `gen_nbg` 컴파일러가 이를 올바르게 처리하지 못할 가능성이 높다.

Acuity CPU 시뮬레이션에서는 `variable` 레이어를 소프트웨어로 정확히 실행하므로 FP32와 100% 일치하지만, NPU 하드웨어에서는 다른 실행 경로를 타면서 출력이 달라진다.

#### ONNX 정확도 검증

onnxsim 적용 전후 ONNX 모델의 출력을 비교:
- **Non-PAD 토큰 max diff: 0.0** (완벽 보존)
- PAD 토큰은 nopad10 적용으로 정확히 -10.0 차이

onnxsim은 연산을 제거하는 것이 아니라, 동적 shape 연산을 **상수로 대체(constant folding)**하여 정확도를 완벽히 보존한다.

#### 가설

T527 NPU 컴파일러(gen_nbg)가 `variable` 레이어(동적 shape 연산)를 포함한 모델에서 올바른 NPU 코드를 생성하지 못한다. onnxsim으로 `variable` 레이어를 24→1로 감소시킨 NB (`nopad10_sim_ma`)가 NPU에서 정상 동작한다면 이 가설이 입증된다.

#### 양자화 파라미터 비교

| 항목 | 영어 (base-960h) | 한국어 (base-korean) |
|------|------------------|---------------------|
| 입력 범위 | [-0.39, 0.34] | [-8.70, 8.28] |
| 입력 scale | 0.002860 | 0.066622 |
| 출력 범위 | [-27.9, 10.4] | [-10.0, 11.7] |
| L11 Add_1 범위 | 51.8 | 6.6 |
| 텐서 수 | 694 | 670 |

### 8.10 Phase 3 NB 변형 목록

Phase 3에서 생성한 추가 NB 변형:

| # | 변형 | NB 크기 | 설명 | NPU 테스트 |
|---|------|---------|------|-----------|
| 27 | nopad10_sim_ma | 72MB | **onnxsim 단순화** + nopad10 + MA | 대기중 |
| 28 | combo_relu_6l_nopad10_ma | 38MB | ReLU + 6-layer + nopad10 (동적ops 포함) | 대기중 |
| 29 | combo_sim_relu_6l_nopad10_ma | ~38MB | **onnxsim** + ReLU + 6-layer + nopad10 | 대기중 |
| 30 | XLS-R nopad10_ma | 249MB | XLS-R-300M + nopad10 + MA | 대기중 |
| 31 | XLS-R 12layers_nopad10_ma | 128MB | XLS-R-300M 12-layer + nopad10 | 대기중 |

**최우선 테스트 대상: #27 (nopad10_sim_ma)** — onnxsim으로 동적 연산을 제거한 유일한 변형.

### 8.11 Phase 4: Opset 12 Re-export — 근본적 해결 시도

#### 핵심 발견: Opset 차이가 동적 연산의 원인

분석 결과 **영어 모델은 opset 12**, **한국어 모델은 opset 14**로 export되어 있었다:

| 속성 | 영어 (base-960h) | 한국어 (base-korean) |
|------|-------------------|---------------------|
| ONNX Opset | **12** | **14** |
| ONNX 노드 수 | 957 | 1,306 |
| 동적 Shape ops | 1 | 37 |
| Pegasus variable layers | **12** | **24** |
| NPU 동작 | **성공** | **실패** |

Opset 14에서 추가된 `aten::scaled_dot_product_attention` (SDPA)이 export 시 동적 Shape→Gather→Concat→Reshape 패턴을 생성한다. Opset 12에서는 수동 attention 구현이 사용되어 정적 Reshape만 생성된다.

#### 해결: 수동 Attention으로 Opset 12 Re-export

HuggingFace transformers의 `Wav2Vec2SdpaAttention.forward()`를 수동 attention 구현으로 monkey-patch하여 opset 12 export에 성공:

```python
# SDPA → Manual attention (opset 12 호환)
def manual_attention_forward(self, hidden_states, ...):
    bsz, tgt_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    w = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
    w = F.softmax(w, dim=-1)
    o = torch.matmul(w, v).transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
    return self.out_proj(o), None, None
```

결과 비교:

| 속성 | 원본 (opset 14) | Opset 12 Re-export | Opset 12 + onnxsim |
|------|----------------|--------------------|--------------------|
| ONNX 노드 수 | 1,306 | **849** | **667** |
| Shape ops | 37 | **1** | **0** |
| Gather ops | 24 | **0** | **0** |
| Concat ops | 48 | **0** | **0** |
| Pegasus variable layers | 24 | **12** | **1** |
| 정확도 (vs 원본) | 기준 | max diff 0.000067 | 동일 |

#### Phase 4 NB 변형 목록

| # | 변형 | NB 크기 | 변수 레이어 | 설명 |
|---|------|---------|-----------|------|
| 32 | nopad10_opset12_ma | 72MB | 12 | opset12 re-export + nopad10 + MA |
| 33 | nopad10_opset12_sim_ma | 72MB | **1** | opset12 + onnxsim + nopad10 + MA |
| 34 | 6L_nopad10_opset12_sim_ma | **39MB** | 1 | 6-layer pruning + opset12 + sim |
| 35 | relu_nopad10_opset12_sim_ma | 72MB | 1 | GELU→ReLU + opset12 + sim |
| 36 | 6L_relu_nopad10_opset12_sim_ma | **39MB** | 1 | 6L + ReLU + opset12 + sim (최소 모델) |
| 37 | nopad10_opset12_sim_normal | 72MB | 1 | opset12+sim + normal quantization |
| 38 | nopad10_opset12_sim_kl | 72MB | 1 | opset12+sim + KL divergence |
| 39 | nopad10_opset12_sim_hybrid_ma | 72MB | 1 | opset12+sim + hybrid (attention bf16) |
| 40 | nopad10_opset12_sim_int16 | **153MB** | 1 | **int16 DFP — 이전 opset14에서 segfault났던 것이 opset12+sim에서 성공!** |

| 41 | nopad10_opset12_sim_int8 | 72MB | 1 | opset12+sim + signed int8 |
| 42 | nopad10_opset12_sim_ma96 | 72MB | 1 | opset12+sim + MA + 96 calibration samples |
| 43 | clip3s_nopad10_opset12_sim_ma | **98MB** | 1 | weight clipping ±3σ + opset12+sim + MA |
| 44 | clip3s_nopad10_opset12_sim_int16 | **175MB** | 1 | weight clipping + int16 DFP |
| 45 | nopad10_opset12_sim_ma01 | 72MB | 1 | opset12+sim + MA weight 0.01 |
| 46 | nopad10_opset12_sim_ma001 | 72MB | 1 | opset12+sim + MA weight 0.001 |

**총 47종+ 변형 생성 완료 (base-korean 40종 + XLS-R 5종 + clip3s/MA변형 2종). 디바이스 FEL 모드로 전체 NPU 테스트 대기중.**

#### Pegasus CPU 시뮬레이션 비교 (opset12+sim 모델, 동일 입력)

| 양자화 | PAD/149 | 상위3 토큰 | 디코드 텍스트 (처음 40자) | 평가 |
|--------|---------|-----------|------------------------|------|
| **FP32** (기준) | **34** | [PAD]:34, 공백:32, ㅇ:17 | ㅇㅣㅣㄱㅡㄴㅇㅡㄴ ㄱㅙㄴ ㅊㅏㄹㄴㅇㅡㄴ ㅈㅓㅇㄱ | 기준 |
| **int16 DFP** | **32** | 공백:35, [PAD]:32, ㅇ:15 | ㅇㅣㅣㄱㅡㄴㅇㅡㄴ ㄱㅙㄴ ㅊㅏㄹㄴㅇㅡㄴ ㅈㅓㄱ | **FP32와 거의 동일** |
| uint8 MA (24cal) | 0 | 공백:47, ㅇ:35, ㅏ:17 | ㅇㄱㅡㄴㅇㅡㄴ ㅇ ㅇㄱㅐㅇㄹ ㅊㅏㄹㄴㅇㅡ | 열화 있으나 유의미 |
| int8 | 0 | 공백:47, ㅇ:35, ㅏ:17 | ㅇㄱㅡㄴㅇㅡㄴ ㅇ ㅇㄱㅐㅇㄹ ㅊㅏㄹㄴㅇㅡ | uint8 MA와 동일 |
| Hybrid MA | 0 | 공백:47, ㅇ:35, ㅏ:17 | ㅇㄱㅡㄴㅇㅡㄴ ㅇ ㅇㄱㅐㅇㄹ ㅊㅏㄹㄴㅇㅡ | uint8 MA와 동일 |
| KL divergence | 0 | 공백:50, ㅇ:37, ㄱ:13 | ㅇㄱㅇㄱ ㄱㅡㄴㅇㅡㅇ ㅇ ㄱㅚㅕㄴ | MA보다 약간 나쁨 |
| Normal | 0 | ㅇ:68, 공백:39, ㅏ:8 | ㅇ ㅇㄱㅡㄹㅇㅐ ㅇ ㅇㅡ ㄹㄱㅇㅐ | 가장 나쁨 |
| MA (96cal aug) | 0 | ㅇ:60, 공백:48, ㅏ:16 | ㅇㅡㅇㄱㅡㄹㅇㅐ ㅇ ㅇ ㅇㅔ | 증강 데이터로 오히려 악화 |
| 6L uint8 MA | 0 | ㅏ:20, ㅡ:14, ㅣ:11 | ㅣㅂㅟㅣㅅㅡㅐㅜㅂㅕ... | 6-layer: 완전 쓰레기 |
| ReLU uint8 MA | 0 | ㅇ:122, ㄴ:26, ㅅ:1 | ㅇㄴㅇㄴㅇㄴㅇㅅㅇㄴ... | GELU→ReLU: 완전 실패 |
| clip3s uint8 MA | 0 | 공백:67, ㅇ:49, ㅏ:8 | ㅇ ㅇ ㅇㄱㅡㄴ ㅇㅡㄴ ㅇㅔ | weight clip: 약간 악화 |
| clip3s int16 DFP | 0 | 공백:48, ㅇ:46, ㄴ:13 | ㅇㄱㅡㄴㅇㅡㄴ ㅇ ㄱㅐㄴㅇ ㅊㅏㄴ | **FP32 거의 동일** |
| MA01 (weight 0.01) | 0 | ㅇ:60, 공백:48, ㅏ:16 | ㅇㅡㅇㄱㅡㄹㅇㅐ ㅇ ㅇ | MA과적합: 악화 |
| symmetric int8 | 0 | ㅇ:77, 공백:34, ㅏ:18 | ㅇㅣㄹ ㄹㅇㅐ ㅇ ㅇ ㅇ | 최악, NB 생성도 실패 |

**핵심 결론:**
- **int16 DFP가 압도적** — FP32와 거의 동일한 출력, PAD 토큰도 정상 분포
- uint8 MA/int8/Hybrid는 열화 있지만 일관된 한국어 텍스트 출력
- 6-layer, ReLU 변형은 시뮬레이션에서도 실패 → NPU 테스트 불필요
- 추가 calibration (96샘플 증강)은 오히려 품질 저하

#### 중요 발견: int16 NB 생성 성공

원본 opset14 모델에서는 int16 NB 생성 시 gen_nbg segfault로 실패했으나, **opset12+sim 모델에서는 153MB int16 NB 생성에 성공**했다. 이는 동적 연산 제거가 NB 컴파일 안정성에도 영향을 미친다는 증거다. int16은 uint8보다 Transformer 모델에 훨씬 적합하므로, 이 NB가 NPU에서 동작한다면 가장 높은 정확도를 기대할 수 있다.

#### 테스트 우선순위

1. **#33 (nopad10_opset12_sim_ma)** — 1 variable layer, 가장 깨끗한 12-layer 모델
2. **#32 (nopad10_opset12_ma)** — 12 variable layers (영어와 동일 수), 정확도 보존
3. **#27 (nopad10_sim_ma)** — 원본 opset14에서 onnxsim 적용
4. **#34 (6L_nopad10_opset12_sim_ma)** — 39MB, 속도 우선
5. **#36 (6L_relu_nopad10_opset12_sim_ma)** — 최소 모델, 양자화 내성 최대화

### 8.12 XLS-R-300M Opset 12 Re-export

base-korean 모델에서 성공한 opset 12 re-export 기법을 XLS-R-300M에도 적용:

```
kresnik/wav2vec2-large-xlsr-korean
→ 316.7M params, 24 layers, 1024 hidden, 16 heads
→ Vocab: 1205 (한국어 음절 단위, base-korean의 56 자모와 다름)
→ 12-layer pruning + manual attention + opset 12 + onnxsim
→ 729 nodes, 0 Shape ops → 121MB uint8 NB
```

| # | 변형 | NB 크기 | 설명 |
|---|------|---------|------|
| 43 | XLS-R 12L nopad10_opset12_sim_ma | **121MB** | 12-layer + opset12 + sim + MA |
| (기존) | XLS-R nopad10_ma | 249MB | opset14 원본, ALL PAD |
| (기존) | XLS-R 12L nopad10_ma | 128MB | opset14, 12-layer, ALL PAD |

**Pegasus CPU 시뮬레이션 (XLS-R 12L opset12+sim):**

| 양자화 | PAD/149 | 상위 토큰 | 디코드 텍스트 |
|--------|---------|-----------|-------------|
| FP32 | 0 | \|:83, 내:5, 그:5 | 미인 육몇역 그 내 려용 약낮악날학 직기치이 들서 |
| uint8 MA | 0 | \|:72, 더:5, 인:5 | 인 인 인 런 르숙여더석갖 실 그 내대내거 이용유 |

> \| 토큰은 단어 경계(space) 역할. FP32에서 83회, uint8에서 72회 — 합리적인 분포.
> 음절 단위 출력으로 base-korean (자모 단위)보다 직관적이나, 12-layer 축소로 정확도 저하.

### 8.13 추가 시도 (실패)

| 접근 | 결과 | 이유 |
|------|------|------|
| SmoothQuant (alpha=0.1~0.5) | FP32에서도 빈 출력 | LayerNorm 스케일링이 wav2vec2에 부적합 |
| Temperature scaling (T=2.0) | FP32 텍스트 변경, uint8 품질 악화 | attention smoothing이 역효과 |
| PCQ (per-channel int8) on opset12+sim | gen_nbg segfault | T527 gen_nbg가 perchannel_symmetric_affine 미지원 |
| vsimulator NB 직접 실행 | double free crash | x86 소프트웨어 시뮬레이터 한계 |
| 추가 calibration (96샘플 증강) | uint8 품질 오히려 악화 | 증강 데이터가 원본 분포 왜곡 |
| Weight clipping (±3σ) | 시뮬레이션 약간 악화 (ㅇ:49→67 space) | 0.89% weights clipped, 정밀도 손실이 이득보다 큼 |
| 8-layer 모델 | FP32에서도 쓰레기 | 한국어 모델은 12L 전부 필요 |
| 10-layer 모델 | FP32에서도 쓰레기 | 10L도 부족, 최소 11-12L 필요 |
| MA weight 0.01 | 시뮬레이션 악화 (ㅇ:60) | 과도한 MA weight → 개별 샘플에 과적합 |
| MA weight 0.001 | MA01과 동일 결과 | shared .data file 문제 (신뢰 불가) |
| symmetric_affine int8 | gen_nbg Fatal error 65280 | T527 gen_nbg가 symmetric quantizer 미지원 |
| bf16 on opset12+sim | gen_nbg Fatal error 35584 | opset12에서도 bf16 NB 생성 불가 |
| clip3s + int16 DFP | 시뮬레이션 FP32 동일 | NB 175MB, 동작 가능하나 clip3s 이점 미미 |

### 8.14 결론 및 교훈

1. **base 아키텍처(94M, 12L)도 한국어 wav2vec2는 uint8 양자화 실패** — 영어 base-960h(CER 17.52%)와 동일 구조임에도 실패. vocab이 작아(56 vs 32) 양자화에 유리할 것으로 예상했으나, Transformer의 양자화 오류 누적은 vocab 크기와 무관하게 발생.

2. **Acuity 시뮬레이션 결과는 NPU 결과와 일치하지 않음** — MA/KL 200샘플 캘리브레이션으로 시뮬레이션에서 100% FP32 일치를 달성했으나, 실제 NPU에서는 ALL PAD 또는 garbled 출력. 시뮬레이션과 NPU 실행의 불일치가 **근본적 문제**이며, 시뮬레이션 기반 양자화 튜닝의 한계를 보여줌.

3. **nopad 트릭의 한계** — nopad10으로 모든 프레임 non-PAD 달성, 그러나 FP32와의 상관계수 r=0.56 (약한 양의 상관)에 불과. 출력의 81%가 단일 토큰(ㅇ)으로, 의미 있는 한국어 텍스트 불가.

4. **int16만이 유효** — 시뮬레이션에서 98.7% FP32 일치. 그러나 T527 NPU의 int16 지원 제한(shader 컴파일 문제, NPU hang 가능성)으로 실용화에 장벽.

5. **시도한 47종+ 양자화/모델 변형 전략 요약:**
   - ✅ 성공: 없음 (디바이스 FEL 모드로 Phase 3-4 NPU 미테스트)
   - ⚠️ 시뮬레이션 유망: int16 DFP (FP32와 거의 동일), uint8 MA/int8/Hybrid (열화 있지만 유의미)
   - ⚠️ 부분 성공: nopad10_ma (NPU non-PAD 출력, 81% ㅇ 토큰)
   - ❌ 시뮬레이션 실패: 6L, 8L, 10L, ReLU, SmoothQuant, Temperature scaling, MA 96-cal 증강, weight clip(uint8), symmetric int8
   - ❌ NB 생성 불가: bf16(opset12/14), int16(opset14), PCQ, symmetric_affine (gen_nbg segfault/error)
   - 🔍 **미테스트 (Phase 3):** onnxsim으로 동적 연산 제거한 NB 4종
   - 🔍 **미테스트 (Phase 4, 최우선):** opset 12 re-export NB 14종 + XLS-R 12L opset12 NB 2종

6. **ONNX 동적 연산 차이 발견** — 한국어 모델은 **opset 14** (SDPA)로 export되어 attention에서 Shape→Gather→Concat→Reshape 동적 패턴(349개 추가 노드)을 생성. 영어 모델은 **opset 12** (수동 attention)로 export되어 static Reshape만 사용. `onnxsim`으로 동적 연산 제거 시 627노드 감소(1306→679). **opset 12로 re-export하면 근본적으로 동적 연산이 발생하지 않음** (849 nodes, Shape 1).

7. **Wav2Vec2 Transformer는 T527 uint8 양자화에 근본적으로 부적합할 가능성** — attention score의 softmax, layer normalization 등 정밀도에 민감한 연산이 uint8의 256단계 해상도로 표현 불가. 단, opset12 re-export로 동적 연산을 제거한 모델은 미테스트 — 동적 연산이 원인이라면 해결 가능. CNN 기반 KoCitrinet(CER 44.44%)이 같은 NPU에서 정상 동작하는 것과 대조적.

---

## 9. T527 NPU STT 모델 종합 비교

### 9.1 성능 비교

| 모델 | 언어 | CER | WER | 추론시간 | RTF | NB 크기 | 입력 | 비고 |
|------|------|-----|-----|---------|-----|---------|------|------|
| **KoCitrinet 300f int8** | 한국어 | **44.44%** | — | **120ms** | 0.040 | ~5MB | 3초 | CNN 기반, 양자화 적합 |
| **Wav2Vec2 uint8** | 영어 | **17.52%** | 27.38% | 715ms | 0.143 | 87MB | 5초 | Transformer, uint8 성공 |
| XLS-R-300M uint8 | 한국어 | ALL PAD | — | 1,098ms | 0.366 | 249MB | 3초 | 300M params, 양자화 완전 실패 |
| base-korean uint8 | 한국어 | ALL PAD | — | 425ms | 0.142 | 75.5MB | 3초 | 94M, sim 100% 일치 but NPU 실패 |
| base-korean nopad10_ma | 한국어 | garbled | — | 425ms | 0.142 | 72MB | 3초 | NPU non-PAD 출력 but 81% ㅇ |
| base-korean 6-layer | 한국어 | (미테스트) | — | — | — | **38MB** | 3초 | encoder 절반 제거 |
| base-korean CNN-only | 한국어 | (미테스트) | — | — | — | **3MB** | 3초 | Transformer 완전 제거 |
| base-korean **onnxsim** nopad10_ma | 한국어 | **(미테스트)** | — | — | — | 72MB | 3초 | **동적ops 제거 — 최우선 테스트** |
| base-korean combo_sim | 한국어 | (미테스트) | — | — | — | ~38MB | 3초 | onnxsim + ReLU + 6L |
| base-korean **opset12** nopad10_ma | 한국어 | **(미테스트)** | — | — | — | 72MB | 3초 | **opset12 re-export, var 12** |
| base-korean **opset12+sim** nopad10_ma | 한국어 | **(미테스트)** | — | — | — | 72MB | 3초 | **opset12+onnxsim, var 1 — 최우선** |
| base-korean opset12+sim 6L | 한국어 | (미테스트) | — | — | — | **39MB** | 3초 | 6-layer + opset12 + sim |
| base-korean opset12+sim ReLU | 한국어 | (미테스트) | — | — | — | 72MB | 3초 | ReLU + opset12 + sim |
| base-korean opset12+sim 6L ReLU | 한국어 | (미테스트) | — | — | — | **39MB** | 3초 | 최소 모델 (6L+ReLU) |
| XLS-R nopad10_ma | 한국어 | (미테스트) | — | — | — | 249MB | 3초 | 300M, nopad10 + MA |
| XLS-R 12L nopad10_ma | 한국어 | (미테스트) | — | — | — | **128MB** | 3초 | 12/24 layers, nopad10 |
| **XLS-R 12L opset12+sim MA** | 한국어 | **(미테스트)** | — | — | — | **121MB** | 3초 | **opset12 re-export, 729nodes, 0 Shape** |
| base-korean **opset12+sim int16** | 한국어 | **(미테스트)** | — | — | — | **153MB** | 3초 | **int16 DFP — opset12에서 NB 생성 성공!** |
| base-korean int16 (opset14) | 한국어 | ~0%* | — | (미확인) | — | NB 생성 실패 | 3초 | *시뮬레이션 98.7% FP32 일치, gen_nbg segfault |

### 9.2 분석

1. **CER 직접 비교는 부적절:** 영어(17.52%)와 한국어(44.44%)는 언어 특성이 다르다. 한국어는 교착어(조사·어미 변화)이고 문자 수가 많아 CER이 구조적으로 높다.

2. **속도:** KoCitrinet(120ms)이 Wav2Vec2(715ms)보다 **6배 빠르다**. KoCitrinet은 1D CNN 기반(SE block 포함)이고, Wav2Vec2는 Transformer 기반이라 연산량 차이가 크다. 한국어 base-korean(425ms)은 3초 입력이라 5초 영어 모델보다 빠르다.

3. **모델 크기:** KoCitrinet NB(~5MB)가 Wav2Vec2 NB(87MB)보다 **17배 작다**. 임베디드 환경에서는 KoCitrinet이 압도적으로 유리하다.

4. **실시간 처리:** 모든 모델이 실시간보다 빠르다.
   - KoCitrinet: 3초 오디오를 0.12초에 처리 → **25배 실시간**
   - Wav2Vec2 (영어): 5초 오디오를 0.72초에 처리 → **7배 실시간**
   - Wav2Vec2 (한국어): 3초 오디오를 0.43초에 처리 → **7배 실시간**

5. **핵심 발견: Transformer 모델의 uint8 양자화 한계**
   - CNN 기반(KoCitrinet): int8 양자화로 FP32 대비 CER +0.33%p 열화 → **양자화 적합**
   - Transformer 기반(Wav2Vec2): 영어(12L, vocab 32)만 uint8 성공, 한국어(12L 또는 24L)는 **ALL PAD로 완전 실패**
   - int16 양자화만이 Transformer에서 정상 동작(98.7~100% 일치)하나, T527 NPU의 int16 지원 제한(shader 컴파일 문제, NPU hang)으로 **실용화 불가**
   - **잠정 결론: Phase 3-4 (opset12 re-export) NPU 테스트가 최종 판단 근거.** int16 DFP가 시뮬레이션에서 FP32 수준이므로, NPU에서도 동작한다면 한국어 STT 가능. 실패 시 CNN 기반(KoCitrinet)이 유일한 선택지

### 9.3 NPU 성능 비교 (타 플랫폼)

| 플랫폼 | 양자화 | RTF | 비교 |
|--------|--------|-----|------|
| **T527 NPU** (Vivante VIP9000) | uint8 | **0.143** | 기준 |
| RK3588 NPU | fp16 | 0.15 | T527과 유사 |
| RTX A6000 GPU | fp16 | 0.007 | T527의 ~20배 빠름 |

---

## 10. 재현 방법

### 10.1 vpm_run으로 NPU 직접 테스트

```bash
# Windows adb를 통해 T527 보드에 파일 전송
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"

# NB 모델 및 테스트 데이터 전송
$WIN_ADB push network_binary.nb /data/local/tmp/w2v_test/
$WIN_ADB push input_0000.dat /data/local/tmp/w2v_test/
$WIN_ADB push sample_0000.txt /data/local/tmp/w2v_test/

# NPU 추론 실행
$WIN_ADB shell "cd /data/local/tmp/w2v_test && \
  LD_LIBRARY_PATH=/vendor/lib64 \
  /data/local/tmp/vpm_run_aarch64 -s sample_0000.txt -b 0"

# 출력 파일 회수 및 CER 평가
$WIN_ADB pull /data/local/tmp/w2v_test/output_0000.dat .
python3 scripts/eval_wav2vec_cer.py --output-dir . --gt ground_truth.txt
```

### 10.2 Android 앱 테스트

```bash
# 단일 파일 테스트
adb shell am start -n com.t527.awaiasr_2/com.t527.wav2vecdemo.Wav2VecTestActivity \
  --es auto_test test.wav

# 배치 테스트 (ground_truth.txt 기반)
adb shell am start -n com.t527.awaiasr_2/com.t527.wav2vecdemo.Wav2VecTestActivity \
  --es auto_en_batch ground_truth.txt

# 로그 확인
adb logcat -s Wav2VecTestActivity | grep BATCH_
```

### 10.3 출력 디코딩 (Python)

```python
import numpy as np

# NPU 출력 로드 및 역양자화
output = np.fromfile("output_0.dat", dtype=np.uint8).reshape(249, 32)
logits = (output.astype(np.float32) - 186) * 0.150270  # zp=186, scale=0.150270

# CTC greedy decode
tokens = np.argmax(logits, axis=1)
VOCAB = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # index 4=space, 5~31=문자
deduped = [tokens[0]] + [t for i, t in enumerate(tokens[1:]) if t != tokens[i]]
text = ''.join(VOCAB[t-4] for t in deduped if 4 <= t <= 31)
print(text)
```

---

## 부록: 파일 구조

```
t527-stt/wav2vec2/
├── RESULTS.md                    # 본 문서
├── network_binary.nb             # T527 NPU용 uint8 NB (87MB)
├── nbg_meta.json                 # 양자화 파라미터 (scale, zero_point)
├── ground_truth.txt              # 50개 테스트 샘플 정답
├── wav2vec2_base_960h_5s_inputmeta_fixed.yml  # Acuity 입력 메타 (수정 완료)
├── wav2vec_postprocess.cpp       # 출력 후처리 C++ (텐서 레이아웃 수정 완료)
├── wav2vec_postprocess.h         # 후처리 헤더
└── scripts/
    ├── eval_wav2vec_cer.py       # CER/WER 평가 스크립트
    ├── compare_onnx_npu.py       # ONNX vs NPU 출력 비교
    └── compare_onnx_npu_50.py    # 50샘플 일괄 비교
```
