# RKNN vs Acuity 양자화 비교 — RK3588 기법의 T527 적용 방안

RK3588 NPU (RKNN-Toolkit2)에서 wav2vec2 CER 35.96% → 11.78%를 달성한 핵심 기법을 T527 NPU (Acuity 6.12.0)에 적용하는 방안.

> 원본: [nsbb/rknn-stt](https://github.com/nsbb/rknn-stt) — RK3588 NPU 한국어 STT 프로젝트

---

## rknn-stt 핵심 성과 요약

RK3588 NPU에서 wav2vec2-xls-r-300m (24L, 300M params) 한국어 STT:

| 단계 | 기법 | CER | 개선 |
|------|------|-----|------|
| Baseline | FP16 단일 모델 | 35.96% | — |
| +양자화 | KL divergence INT8 (Split) | 35.25% | -0.71pp |
| +정규화 | amplitude norm 0.95 | 18.25% | -16.99pp |
| +최적화 | amplitude norm 5.0 | **11.78%** | -6.47pp |

### Split INT8+FP16 아키텍처 (4파트 분할)

```
Part1 (FP16, 12.7MB)   — CNN Feature Extractor
Part2A (INT8-KL, 167MB) — Encoder Layer 0~11 (양자화에 강함)
Part2B (FP16, 295MB)    — Encoder Layer 12~23 (양자화에 민감)
Part3 (FP16, 5.2MB)     — LM Head
총 480MB, 추론 427ms/5초
```

핵심 발견: Transformer 전반부(Layer 0~11)는 INT8 양자화에 내성이 있지만, 후반부(Layer 12~23)는 LayerNorm + Softmax + GELU 조합이 INT8 정밀도에 극도로 민감.

---

## T527 적용 가능한 기법

### 1. Amplitude Normalization (즉시 적용 가능)

**난이도: ★☆☆ | 기대 효과: 매우 높음**

rknn-stt에서 가장 큰 CER 개선을 가져온 기법. INT8 양자화 모델에서 입력 음성의 peak를 목표값으로 정규화하면 dynamic range 활용도가 극대화됨.

```python
peak = np.max(np.abs(audio))
audio = audio / peak * 5.0  # peak를 5.0으로 정규화
```

**T527 적용 대상:**
- Wav2Vec2 base-960h (영어, uint8, CER 17.52%) → amplitude norm으로 CER 개선 가능
- Wav2Vec2 base-korean (한국어, uint8, CER 100.86%) → 이것만으로 개선될 가능성

**rknn-stt 실험 결과 (RK3588):**

| 목표값 | CER | 빈 출력 |
|:---:|:---:|:---:|
| 정규화 없음 | 35.25% | 23건 |
| 0.95 | 18.25% | 1건 |
| **5.0** | **11.78%** | **0건** |
| 10.0 | 12.19% | 0건 |

> T527은 Acuity 기반이므로 최적 목표값이 다를 수 있음. 0.5~10.0 범위에서 탐색 필요.

**구현:**
- 전처리 파이프라인(Python 또는 JNI)에서 waveform normalize 추가
- 기존 테스트 스크립트에 `--amp-norm` 옵션 추가

### 2. KL Divergence 양자화 (즉시 적용 가능)

**난이도: ★☆☆ | 기대 효과: 중간**

Acuity 6.12.0에 이미 `--algorithm kl_divergence` 옵션 존재. 현재 사용 중인 `moving_average` 대비 이상치에 강함.

```bash
pegasus quantize \
  --quantizer asymmetric_affine --qtype uint8 \
  --algorithm kl_divergence \
  ...
```

**rknn-stt 효과:** CER 43.90% → 35.25% (-8.65pp)

**T527 적용 대상:**
- Wav2Vec2 base-korean uint8 재양자화
- KoCitrinet int8 재양자화 (현재 moving_average 사용 중)

### 3. Split Model — 전반부 uint8 + 후반부 int16 (중기 과제)

**난이도: ★★★ | 기대 효과: 높음 (한국어 Wav2Vec2 해결 가능성)**

T527에서 FP16은 25배 느려서 사용 불가. 대신 **int16 DFP**를 후반부에 사용.

#### T527 base-korean (12L) Split 전략

```
Part A: CNN Feature Extractor (uint8)
  ONNX: ~36MB → NB: ~4MB

Part B前: Encoder Layer 0~5 (uint8)
  ONNX: ~180MB → NB: ~36MB

Part B後: Encoder Layer 6~11 (int16 DFP)
  ONNX: ~180MB → NB: ~72MB

Part C: LM Head (uint8)
  ONNX: ~5MB → NB: ~2MB
```

각 파트 NB ≤ 128MB → T527 NPU 크기 제한 이내.

**전제 조건:**
- ONNX 모델을 4파트로 분할하는 스크립트 필요 (rknn-stt의 `export_onnx.py` 참고)
- 파트 간 I/O 직렬 실행 파이프라인 구현
- int16 DFP가 T527에서 동작하는 것은 Zipformer 118MB로 확인 완료

**rknn-stt에서 검증된 Split 지점 효과:**

| Split | INT8 범위 | CER (RK3588) |
|:---:|:---:|:---:|
| Split11 | Layer 0~11 | 11.78% |
| Split15 | Layer 0~15 | 11.74% |
| Split17 | Layer 0~17 | 11.96% |

> 12L 모델에서는 Split6 (전반 6L INT8, 후반 6L INT16)이 대응됨.

#### 대안: KoCitrinet int16 (단기 과제)

KoCitrinet은 CNN 기반이라 양자화에 강하지만, int16으로 정밀도를 높이면 CER 개선 가능:
- 현재 uint8 NB: 62MB → int16 NB: **~120MB** (추정, 128MB 이내)
- uint8 CER 44.44% → int16에서 개선 기대
- Split 불필요 (단일 NB로 동작 가능)

### 4. 큰 모델 활용 — XLS-R-300M Split (장기 과제)

**난이도: ★★★★ | 기대 효과: 매우 높음**

rknn-stt에서 CER 11.78%를 달성한 XLS-R-300M (24L, 300M params)을 T527에서 Split 실행.

```
Part1 (uint8, ~4MB)    — CNN
Part2A (uint8, ~60MB)  — Encoder Layer 0~11
Part2B (int16, ~120MB) — Encoder Layer 12~23  ← NB 크기 제한 경계
Part3 (uint8, ~2MB)    — LM Head
```

**문제:** Part2B int16이 ~120MB로 128MB 제한 경계선. Layer 12~17만 int16으로 하고 18~23은 별도 파트로 분리하면 해결 가능하나 복잡도 증가.

---

## 적용 우선순위

| 순위 | 기법 | 대상 모델 | 난이도 | 기대 효과 | 소요시간 |
|:---:|------|-----------|:---:|:---:|:---:|
| 1 | **Amplitude norm** | Wav2Vec2 base-korean uint8 | ★☆☆ | CER 100% → ? (대폭 개선 가능) | 수 시간 |
| 2 | **KL divergence 양자화** | Wav2Vec2 base-korean uint8 | ★☆☆ | CER 추가 개선 | 수 시간 |
| 3 | **KoCitrinet int16** | KoCitrinet 300f | ★★☆ | CER 44.44% → ? | 1일 |
| 4 | **Split uint8+int16** | Wav2Vec2 base-korean 12L | ★★★ | 한국어 STT 해결 가능 | 수 일 |
| 5 | **XLS-R-300M Split** | XLS-R-300M 24L | ★★★★ | CER ~12% 목표 | 1주+ |

> **권장: 1+2를 먼저 적용하여 기존 uint8 NB의 CER 개선 확인 후, 3 또는 4 진행.**

---

## 참고: T527 vs RK3588 NPU 비교

| 항목 | T527 (Vivante VIP9000) | RK3588 (RKNN) |
|------|----------------------|---------------|
| NPU 성능 | 2 TOPS | 6 TOPS |
| 코어 수 | 1 | 3 |
| INT8 지원 | O | O |
| INT16 지원 | O (NB ≤128MB) | X (FP16 대신 사용) |
| FP16 지원 | SW 에뮬레이션 (25배 느림) | HW 가속 |
| 양자화 도구 | Acuity Toolkit 6.12 | RKNN-Toolkit2 |
| KL divergence | O (`--algorithm kl_divergence`) | O (RKNN 내장) |
| NB 크기 제한 | ~128MB | 제한 없음 (파트당) |
| 모델 분할 | 수동 ONNX 분할 필요 | 수동 ONNX 분할 필요 |

**T527의 장점:** int16 DFP HW 가속 지원 (RK3588은 미지원). Transformer 후반부를 int16으로 처리하면 FP16 없이도 정밀도 유지 가능.

**T527의 단점:** NB 크기 제한 (~128MB), FP16 HW 미지원, NPU 성능 1/3.

---

## 추가 실험 결과 (2026-03-19)

### Amplitude Normalization — T527에서 효과 없음

rknn-stt에서 CER 24pp 개선한 amp norm을 T527에서 시도:

| 구성 | CER |
|------|-----|
| 기존 uint8 (amp norm 없음) | 100% |
| amp norm 5.0 + 기존 NB | 100% (캘리브레이션 불일치) |
| amp norm 5.0 + KL divergence 재양자화 | 100% (전 프레임 non-blank) |

**차이점**: rknn-stt는 **Split INT8+FP16** 구조에서 amp norm 적용. FP16 파트가 정밀도를 보존하기 때문에 INT8 파트의 dynamic range 개선이 효과 있음. T527은 **전체 uint8**이라 후반 레이어 정밀도 손실이 근본 원인 → amp norm으로 해결 불가.

### 3-Part Split uint8 — Part A(CNN)가 고정값 출력

모델을 CNN + Encoder前半 + Encoder後半으로 분할, 각각 uint8 NPU:

```
Part A (CNN, 3.7MB)     → 고정값 출력 (입력 무시)
Part B (L0-5, 35MB)     → Part A가 고정이니 무의미
Part C (L6-11, 34MB)    → 무의미
```

Part A(CNN)의 uint8 양자화가 7-layer 1D Conv의 출력을 완전히 붕괴시킴.

### int16 DFP 결론

KoCitrinet int16 DFP (150MB, 216ms) 테스트에서 CER 330% → **int8보다 훨씬 나쁨**.

Acuity 6.12의 양자화 조합:
- `asymmetric_affine` + `int16` → **미지원** (uint8/int8만)
- `dynamic_fixed_point` + `int16` → 실행되나 결과 부정확
- **int16은 T527에서 실용적으로 사용 불가**
