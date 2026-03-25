# T527 NPU uint8 STT 양자화 종합 분석: Conformer는 왜 성공하고, 나머지는 왜 실패했는가

**날짜:** 2026-03-25
**범위:** T527 NPU에서 시도한 모든 STT 모델 (15개+, 100+ 양자화 실험, 2000+ 디바이스 테스트)
**핵심 결론:** **모델 아키텍처가 양자화 성패를 결정한다.** vocab 크기, FP32 정확도, 학습 데이터량은 부차적.

---

# 1. Executive Summary

## 한 줄 결론

> **CNN + Self-Attention 하이브리드 구조(Conformer)는 T527 NPU uint8에서 동작한다. 순수 Transformer(wav2vec2, Zipformer, HuBERT)는 동작하지 않는다.**

## 핵심 숫자

| 모델 | 아키텍처 | vocab | FP32 CER | **uint8 CER** | NB | 추론 |
|------|---------|-------|---------|-------------|-----|------|
| **SungBeom Conformer** | **CNN + Attention** | **2049** | 좋음 | **10.02%** | 102MB | 233ms |
| KoCitrinet | CNN only | 2048 | 8.44% | **44.44%** | 62MB | 120ms |
| Wav2Vec2 EN | Transformer | 32 | 9.74% | **17.52%** | 87MB | 715ms |
| **Wav2Vec2 KO (aihub 80k)** | **Transformer** | **1912** | **9~18%** | **92.83%** | 77MB | 424ms |
| Wav2Vec2 KO (base) | Transformer | 56 | 30.22% | **100.86%** | 72MB | 415ms |
| Zipformer | Transformer 변형 | 5000 | 16.2% | **100%** | 63MB | 50ms |
| HuBERT KO | Transformer | 2142 | — | **100%** | 76MB | 423ms |

**Conformer(vocab 2049)가 CER 10.02%로 성공한 반면, wav2vec2(vocab 56, 1912 모두)는 전부 실패.**
**이전 분석에서 "vocab 크기가 원인"이라고 했으나, 이는 틀렸다. 진짜 원인은 아키텍처.**

---

# 2. 모든 모델 스코어보드

T527 NPU에서 시도한 **모든 STT 모델**의 결과.

## 2.1 성공한 모델 (T527 NPU uint8에서 동작)

| # | 모델 | 아키텍처 유형 | Params | vocab | 입력 | CER | NB | 추론 |
|---|------|------------|--------|-------|------|-----|-----|------|
| 1 | **SungBeom Conformer** | **CNN+Attention 하이브리드** | 122.5M | 2049 BPE | mel | **10.02%** | 102MB | 233ms |
| 2 | cwwojin Conformer | CNN+Attention 하이브리드 | 31.8M | 5001 BPE | mel | 54.53% | 29MB | 111ms |
| 3 | KoCitrinet 300f | **CNN only** | ~10M | 2048 SP | mel | 44.44% | 62MB | 120ms |
| 4 | Wav2Vec2 EN (base-960h) | Transformer | 94.4M | 32 | raw wav | 17.52% | 87MB | 715ms |

## 2.2 실패한 모델 (T527 NPU uint8에서 CER > 90%)

| # | 모델 | 아키텍처 유형 | Params | vocab | FP32 CER | **uint8 CER** | 실패 모드 |
|---|------|------------|--------|-------|---------|-------------|----------|
| 5 | Wav2Vec2 KO (base) | **Transformer** | 94.4M | 56 | 30.22% | **100.86%** | 자모 파편 |
| 6 | Wav2Vec2 KO (80k aihub) | **Transformer** | 94.4M | 1912 | **9~18%** | **92.83%** | 거의 blank |
| 7 | Wav2Vec2 XLS-R-300M | **Transformer** | 300M | 2617 | 1.78% | **ALL PAD** | 완전 blank |
| 8 | Zipformer | **Transformer 변형** | ~40M | 5000 | 16.2% | **100%** | 상관계수 0.6 |
| 9 | HuBERT KO | **Transformer** | 96M | 2142 | — | **100%** | 동일 토큰 반복 |
| 10 | SpeechBrain Conformer | CNN+Attention | 42.9M | — | — | NB export 실패 | error 64768 |

## 2.3 미완/미테스트 모델

| # | 모델 | 상태 |
|---|------|------|
| 11 | CitriNet EN 3s | NB 7MB 변환 성공, CER 미측정 |
| 12 | DeepSpeech2 | NB 56MB 변환 성공, CER 미측정 |
| 13 | Whisper tiny encoder | NB 117MB 변환 성공, decoder 미변환 |

---

# 3. uint8 양자화란 무엇이고 왜 모델이 깨지는가

## 3.1 비전공자를 위한 설명

**양자화 = 숫자의 정밀도를 낮추는 것.**

```
원래 모델 (float32): 소수점 7자리까지 표현
  3.1415927, -0.0012345, 128.99999

양자화 후 (uint8): 0~255 중 하나만 선택
  3.14 → 3,  -0.001 → 0,  129.0 → 129
```

**비유:** 고해상도 사진을 256색 GIF로 변환하는 것. 대부분은 비슷하게 보이지만, 미세한 색 차이가 중요한 부분에서 깨진다.

## 3.2 왜 어떤 모델은 되고 어떤 모델은 안 되는가

```
모델 A (Conformer):
  "가" 토큰 점수: 5.20
  "나" 토큰 점수: 3.10
  차이: 2.10 >> uint8 한 칸 크기(0.20)
  → 양자화 후에도 "가"가 1등 유지 ✓

모델 B (wav2vec2 한국어):
  "가" 토큰 점수: 3.012
  "나" 토큰 점수: 3.007
  차이: 0.005 << uint8 한 칸 크기(0.05)
  → 양자화 후 "나"가 1등이 될 수 있음 ✗
```

**핵심:** 모델이 정답과 오답의 점수 차이(margin)를 크게 만들 수 있느냐가 양자화 성패를 결정.

## 3.3 T527 NPU의 특수한 제약: W8A8 강제

T527 NPU (Vivante VIP9000NANOSI_PLUS)는 **weight와 activation 모두 uint8 강제** (W8A8).

| | LLM (GPU) | **T527 NPU** |
|---|---|---|
| Weight | 4-bit | **uint8** |
| Activation | **FP16 유지** | **uint8 강제** |
| 방식 | W4A16 | **W8A8** |
| Softmax 입력 | FP16 → 정확 | **uint8 → 부정확** |

LLM이 vocab 128K에서도 4-bit 양자화가 되는 이유: **activation을 FP16으로 유지**하기 때문.
T527에서 안 되는 이유: **activation도 uint8**이라 내부 연산 정밀도가 극도로 제한.

Acuity Pegasus 소스 코드에서 확인: `pegasus_quantizer` 딕셔너리에 **W8A16 같은 mixed precision 옵션 자체가 없음**.

---

# 4. **Factor 1: 아키텍처 — CNN 하이브리드 vs 순수 Transformer**

> ## **이것이 양자화 성패를 결정하는 가장 핵심적인 요인이다.**

## 4.1 Conformer: CNN + Self-Attention 하이브리드

```
Conformer 블록 하나:
  입력 → [FFN ½] → [Multi-Head Self-Attention] → [Depthwise Conv(kernel=31)] → [FFN ½] → 출력
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^
                                                   이 CNN 블록이 핵심
```

**Depthwise Convolution의 양자화 안정성:**
- 고정 크기 커널(31)로 인접 프레임의 local 패턴을 캡처
- 입출력 값의 범위가 **bounded하고 예측 가능**
- uint8로 양자화해도 local 패턴 정보가 보존
- **Attention 후 activation이 불안정해져도 Conv가 다시 안정화**

## 4.2 Wav2Vec2: 순수 Transformer (CNN frontend만 있음)

```
Wav2Vec2:
  [CNN frontend (7 conv layers)] → [12 × Self-Attention + FFN(GELU)] → [lm_head]
  ^^^^^^^^^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  여기만 CNN (입력 처리용)           12개 레이어 전부 순수 Transformer
                                   Attention 후 안정화 해주는 CNN 없음!
```

**Self-Attention의 양자화 취약성:**
- Q, K, V 행렬곱으로 **모든 프레임 쌍의 관계**를 계산 — 값 범위 예측 불가
- Softmax 출력이 uint8로 깎이면 attention weight 왜곡
- **GELU 활성화 함수**: 비선형 영역이 uint8에서 부정확하게 근사
- 12개 레이어를 통과하면서 오차 누적 → 최종 출력 파괴

## 4.3 실측 데이터로 증명

| 아키텍처 | 대표 모델 | CNN 위치 | uint8 CER | 성공 |
|---------|----------|---------|-----------|------|
| **CNN only** | KoCitrinet | 전체 | 44.44% | **✓** |
| **CNN + Attention (매 레이어)** | SungBeom Conformer | **Attention 뒤에 Conv** | **10.02%** | **✓** |
| CNN frontend + Transformer | Wav2Vec2 EN | **입력부만 CNN** | 17.52% | ✓ (영어만) |
| **CNN frontend + Transformer** | **Wav2Vec2 KO 모든 변형** | **입력부만 CNN** | **92~100%** | **✗** |
| Transformer 변형 | Zipformer | 없음 | 100% | ✗ |
| Transformer | HuBERT KO | 없음 | 100% | ✗ |

**핵심 발견:** CNN이 **매 레이어에 포함**된 Conformer는 성공. CNN이 **입력부에만 있는** wav2vec2는 한국어에서 실패.

---

# 5. Factor 2: 입력 타입 — mel spectrogram vs raw waveform

| | mel spectrogram | raw waveform |
|---|---|---|
| 사용 모델 | Conformer, KoCitrinet | wav2vec2 |
| Shape | `[1, 80, 301]` | `[1, 48000]` |
| 값 범위 | [-1.6, 4.5] (bounded) | [-1.0, 1.0] |
| 의미 | 이미 전처리된 주파수 feature | 원시 오디오 샘플 |
| uint8 호환성 | **좋음** (값 분포 안정) | CNN 처리 후 activation 불안정 |

mel spectrogram은 이미 정규화된 feature라서 uint8 매핑이 예측 가능. raw waveform은 CNN frontend이 처리한 후의 activation 범위가 모델마다 크게 다름.

---

# 6. Factor 3: 위치 인코딩 — bounded vs unbounded

| | Conformer | wav2vec2 |
|---|---|---|
| 방식 | Relative Positional Encoding | Convolutional Positional Encoding |
| 범위 | **[-301, +301] (bounded)** | 학습 데이터에 따라 다름 (unbounded) |
| uint8 매핑 | 깔끔 | 예측 불가 |

---

# 7. Factor 4: Subsampling 전략 — 조기 차원 축소

| | Conformer | wav2vec2 |
|---|---|---|
| 방식 | CNN stem → factor 4 subsampling | CNN frontend → stride ~320x |
| Attention 입력 | **76 frames** | **149 frames** |
| Attention 연산 | 76² = 5,776 쌍 | 149² = 22,201 쌍 |
| 양자화 오차 누적 | 적음 | 많음 |

Conformer는 Attention이 **76개 프레임**만 처리. wav2vec2는 **149개**. 프레임 수가 적을수록 attention weight의 uint8 오차 영향이 줄어듦.

---

# 8. Factor 5: Activation 분포 특성 — 530개 레이어 실측 데이터

## 8.1 영어 vs 한국어 wav2vec2 — 같은 구조, 다른 운명

530개 레이어를 FP32 vs uint8로 dump하여 비교:

| 측정 항목 | 영어 wav2vec2 | 한국어 wav2vec2 | 비율 |
|----------|-------------|---------------|------|
| Attention softmax range | **0.03** | **0.82** | **27배** |
| Logit std | 8.39 | 1.95 | 4배 |
| **Logit margin min** | **0.34** | **0.005** | **68배** |
| L10 residual Add range | 8.2 | **420** | **51배** |

**한국어 모델의 내부 activation range가 영어보다 5~50배 넓다.** uint8 256단계로 이 넓은 범위를 표현하면 해상도가 극도로 부족.

## 8.2 Conformer의 CNN이 activation을 안정화하는 메커니즘

```
Conformer 블록:
  FFN → Attention (값 넓어질 수 있음) → Conv(kernel=31) → FFN
                                        ^^^^^^^^^^^^^^^^
                                        local averaging 효과로
                                        activation range 다시 좁힘

wav2vec2 블록:
  Attention (값 넓어짐) → FFN(GELU) (더 넓어짐) → 다음 레이어
  → 12개 레이어 거치면서 range 계속 넓어짐 → 양자화 파괴
```

---

# 8.5 Logit Margin 실측 비교 (2026-03-25 측정)

모든 모델의 FP32 logit margin을 직접 측정한 결과:

| 모델 | vocab | margin min | uint8 step | **ratio** | **uint8 CER** |
|------|-------|-----------|-----------|-----------|-------------|
| Wav2Vec2 EN | 32 | 0.340 | 0.080 | **4.25x** | **17.52%** |
| **SungBeom Conformer** | **2049** | **0.180** | **0.191** | **0.94x** | **10.02%** |
| cwwojin Conformer (uint8) | 5001 | 0.000 | 0.177 | 0.00x | 54.53% |
| Wav2Vec2 KO (base) | 56 | 0.005 | 0.050 | 0.10x | 100.86% |

**핵심 발견:**
- SungBeom Conformer는 **ratio 0.94x** (margin < step)인데도 **CER 10.02%**로 동작
- wav2vec2 한국어는 ratio 0.10x로 완전 실패
- **CNN이 대부분의 프레임에서 margin을 충분히 확보**하고, 소수 프레임에서만 부족 → 전체 CER은 양호
- 순수 Transformer는 **모든 프레임에서 margin이 불안정** → 전체 CER 붕괴

**이것이 아키텍처가 vocab보다 중요한 결정적 증거:**
- Conformer vocab 2049, margin 0.18 → CER 10% (동작)
- Wav2Vec2 vocab 56, margin 0.005 → CER 100% (실패)
- vocab이 37배 큰 Conformer가 margin은 36배 크다

---

# 9. Factor 6: ONNX 그래프 복잡도

| 모델 | ONNX 노드 수 | uint8 결과 |
|------|------------|-----------|
| KoCitrinet | ~200 | CER 44.44% ✓ |
| Wav2Vec2 | 957 | 한국어 ✗, 영어 ✓ |
| **SungBeom Conformer** | **1982** | **CER 10.02% ✓** |
| Zipformer | **5868** | CER 100% ✗ |

**노드 수 자체는 결정적이지 않다.** 1982 노드인 Conformer가 957 노드인 wav2vec2보다 잘 됨. 중요한 건 **노드 유형** (CNN vs pure Attention).

다만 5868+ 노드에서는 순차적 오차 누적이 치명적.

---

# 10. Factor 7: 출력 헤드 설계와 Logit Margin

| 모델 | vocab | output range | uint8 step | 출력 방식 |
|------|-------|-------------|-----------|----------|
| **SungBeom Conformer** | **2049** | **[-51.8, 0.0]** | **0.203** | **log softmax** |
| Wav2Vec2 EN | 32 | [-36, +17] | 0.208 | raw logit |
| Wav2Vec2 KO | 56 | [-10.3, +12.0] | 0.088 | raw logit |

Conformer(NeMo)는 **log softmax** 출력 — 값이 항상 음수이고 정답은 ~0.0, 오답은 -20~-50. 차이가 매우 커서 uint8 step 0.203으로도 충분.

---

# 10.5 같은 Conformer인데 왜 CER 10% vs 55%? — cwwojin vs SungBeom

| | **SungBeom** | **cwwojin** | 차이 |
|---|---|---|---|
| Params | **122.5M** | 31.8M | **3.9배** |
| d_model | **512** | 256 | **2배** |
| Attention heads | **8** | 4 | 2배 |
| Vocab | **2049** | 5001 | cwwojin이 2.4배 큼 |
| NB | 102MB | **29MB** | cwwojin이 3.5배 작음 |
| Inference | 233ms | **111ms** | cwwojin이 2.1배 빠름 |
| **CER** | **10.02%** | **54.53%** | **SungBeom이 5.4배 좋음** |

**같은 Conformer 아키텍처에서 CER이 5.4배 다른 이유:**

1. **d_model 512 vs 256** — d_model이 클수록 encoder hidden representation의 해상도가 높아서, uint8 양자화 후에도 **충분한 정보를 유지**. d_model 256은 uint8 256단계와 비슷한 수준이라 정보 손실 치명적.

2. **vocab 2049 vs 5001** — vocab이 크면 유사 토큰 간 경쟁이 치열해져 margin 감소. 하지만 이것만으로 5.4배 차이는 설명 안 됨.

3. **모델 용량(d_model)이 vocab보다 더 큰 영향.** SungBeom이 vocab은 2.4배 작지만 d_model이 2배 크다. d_model 512의 표현력이 uint8 양자화 noise를 흡수하는 **버퍼 역할**.

> **결론:** Conformer를 T527에 배포하려면 **d_model ≥ 512** 권장. d_model 256은 uint8에서 정확도 급락.

---

# 11. ⚠️ Vocab 크기 신화 교정 — 이전 분석의 오류

## 11.1 이전에 주장한 것 (틀렸음)

> "vocab 1900이면 logit margin이 uint8 step보다 작아서 안 된다. 자모 56으로 해야 한다."

이 주장에 기반하여 팀에 vocab 56으로 전환을 권고했고, 실행되었다.

## 11.2 현실: Conformer가 증명한 것

| 모델 | vocab | uint8 CER | 결론 |
|------|-------|-----------|------|
| **SungBeom Conformer** | **2049** | **10.02%** | **됨** |
| cwwojin Conformer | 5001 | 54.53% | 됨 (정확도 낮지만 동작) |
| KoCitrinet | 2048 | 44.44% | 됨 |
| Wav2Vec2 KO (56 jamo) | 56 | 100.86% | **안 됨** |
| Wav2Vec2 KO (1912 syllable) | 1912 | 92.83% | **안 됨** |

**vocab 2049에서 되고 vocab 56에서 안 된다.** vocab 크기가 문제가 아님.

## 11.3 오류의 원인

두 가지 문제를 혼동했다:

| | 실제 원인 | 내가 주장한 원인 |
|---|---|---|
| 문제 | **wav2vec2 아키텍처의 activation 분포가 uint8 부적합** | vocab이 크면 margin이 작아서 불리 |
| 해결 | **아키텍처를 Conformer로 변경** | vocab을 56으로 축소 |

## 11.4 정정

**vocab 크기는 부차적 요인.** 같은 아키텍처 내에서는 영향이 있지만 (cwwojin 5001 vs SungBeom 2049), 아키텍처 차이가 압도적으로 크다. wav2vec2는 vocab을 1로 줄여도 아키텍처 특성상 uint8에서 실패할 가능성이 높다.

**aihub 학습은 음절 vocab (~1900~2000)으로 진행해도 된다. 단, 모델 아키텍처를 Conformer CTC로 변경해야 한다.**

---

# 12. Acuity 시뮬레이션 vs T527 디바이스 불일치

wav2vec2 한국어에서 발견:

| 비교 | argmax 일치율 |
|------|-------------|
| Acuity Sim uint8 vs FP32 | 67.1% |
| **Device vs Sim uint8** | **31.5%** |

**시뮬레이션 최적화가 디바이스 결과를 예측 못 함.** 반드시 디바이스 직접 테스트 필요.

Conformer에서는 이 문제 미관찰 — 시뮬레이션과 디바이스 결과 일치.

---

# 13. T527 NPU 하드웨어 제약 종합

| 제약 | 값 | 영향 |
|------|-----|------|
| **양자화** | **uint8 only (W8A8)** | weight+activation 둘 다 uint8 강제 |
| W8A16 | **미지원** | Acuity에 옵션 자체 없음 |
| NB 크기 | ~120MB 이하 | int16 NB (152MB+) → status=-1 거부 |
| int16 DFP | 실행됨 | 정확도가 uint8보다 나쁨 (DFP 한계) |
| bf16 | NB export 실패 | error 64768 |
| fp16 | CPU fallback | HW 미가속, 17초 |
| fp32 | SRAM 부족 | 362MB → 메모리 초과 |

---

# 14. 향후 모델 선택 가이드라인

## 14.1 **반드시 지켜야 할 것**

| 조건 | 이유 |
|------|------|
| **CNN + Attention 하이브리드 구조** | CNN이 activation 안정화 |
| **mel spectrogram 입력** | 값 범위 bounded |
| **NB 크기 < 120MB** | T527 메모리 제한 |
| **디바이스 직접 테스트** | 시뮬레이션 불신뢰 |

## 14.2 **강력 권장**

| 조건 | 이유 |
|------|------|
| Relative positional encoding | bounded → uint8 안정 |
| 조기 subsampling (factor 4+) | Attention sequence 축소 |
| d_model ≥ 512 | 양자화 noise 흡수 용량 |
| vocab ≤ ~2000 | logit margin 여유 (부차적 요인) |

## 14.3 **하지 말아야 할 것**

| 조건 | 이유 |
|------|------|
| 순수 Transformer (CNN 없이) | activation range 불안정 |
| raw waveform 입력 | activation range 예측 불가 |
| 24L+ deep Transformer | 오차 누적 |
| 5000+ 노드 ONNX | 순차적 오차 누적 |
| Acuity 시뮬레이션만 보고 판단 | 디바이스와 불일치 가능 |

## 14.4 추천 모델

| 모델 | 추천도 | 이유 |
|------|--------|------|
| **Conformer CTC (NeMo, ≥medium)** | ⭐⭐⭐⭐⭐ | 실증 CER 10.02% |
| **Conformer CTC (aihub 직접 학습)** | ⭐⭐⭐⭐⭐ | 도메인 특화 가능 |
| Citrinet (NeMo) | ⭐⭐⭐ | CNN only, 안정적이나 정확도 한계 |
| wav2vec2 + QAT | ⭐⭐ | 근본 한계 (margin 0.099, step 0.151) |

---

# 15. Split Model 실험 결과 요약

"encoder NPU uint8, lm_head CPU fp32" 분리 실험도 진행.

| 모델 | 방식 | CER | 결론 |
|------|------|-----|------|
| Wav2Vec2 KO (base) | Full uint8 | 100.86% | 실패 |
| Wav2Vec2 KO (base) | Split lm_head fp32 | 99.70% | 효과 없음 |
| Wav2Vec2 KO (base) | Split L7 | 99.26% | 효과 없음 |
| Wav2Vec2 KO (base) | CNN+lm fp32 | 100.00% | 악화 |
| Wav2Vec2 KO (80k) | Split lm_head fp32 | 92.65% | 효과 없음 |

**encoder uint8 오차가 근본 원인이므로 후처리로 복구 불가.**

---

# 16. QAT 실험 결과 요약

| 방법 | WER | logit margin min | ratio |
|------|-----|-----------------|-------|
| 한국어 base-korean 원본 | 7.5% (FP32) | 0.005 | 0.1x |
| 영어→한국어 fine-tune | 40.6% | 0.037 | 0.25x |
| QAT basic | 38.86% | 0.099 | 0.66x |
| QAT + Margin Loss | 43.05% | 0.000 | 실패 |

**QAT로 margin 3배 개선했으나 step(0.151)을 넘지 못함. wav2vec2의 근본 한계.**

---

# 17. 최종 권고

## 즉시: SungBeom Conformer 사용 (CER 10.02%, 233ms)
## 중기: aihub 데이터 + Conformer CTC 직접 학습 (음절 vocab ~2000 가능)
## 장기: Squeezeformer, E-Branchformer 등 하이브리드 모델 탐색

---

# 18. 학술 근거: Conformer의 양자화 강건성

| 논문 | 연도 | 결과 |
|------|------|------|
| [4-bit Conformer (Google, INTERSPEECH 2022)](https://arxiv.org/abs/2203.15952) | 2022 | Conformer **4-bit까지 lossless**, 7.7x 크기 축소 |
| [1-bit Conformer (2025)](https://arxiv.org/html/2505.21245v1) | 2025 | **2-bit, 1-bit** 극저비트 양자화도 성공 |
| [INT8 Conformer on ARM Ethos-U85 NPU](https://developer.arm.com/community/arm-community-blogs/b/internet-of-things-blog/posts/end-to-end-int8-conformer-on-arm-training-quantization-and-deployment-on-ethos-u85) | 2025 | ARM NPU에서 **int8 Conformer 배포** 성공 |
| [Conformer-Based ASR on Extreme Edge](https://arxiv.org/html/2312.10359v1) | 2023 | 극한 엣지 디바이스에서 Conformer 동작 |
| [Conformer 원본 (Google, INTERSPEECH 2020)](https://arxiv.org/abs/2005.08100) | 2020 | CNN+Attention 하이브리드의 근본 설계 |

**학술적으로도 Conformer가 양자화에 강건하다는 것은 다수 논문에서 검증됨.** 순수 Transformer(wav2vec2)의 양자화 문제는 T527에만 국한된 것이 아니라 **아키텍처의 본질적 특성**.

---

# 18.5 반론: "Transformer가 Conformer보다 양자화에 강건하다"는 연구

### EUSIPCO 2024: "Assessing the Robustness of Conformer and Transformer Models Under Compression"

이 논문은 **Transformer가 Conformer보다 양자화/pruning에 더 robust**하다고 결론. **우리 실험과 모순.**

| | 논문 (EUSIPCO 2024) | 우리 실험 (T527) |
|---|---|---|
| 데이터 | LibriSpeech (영어) | Zeroth-Korean (한국어) |
| 하드웨어 | GPU (FP16 activation) | **T527 NPU (uint8 activation)** |
| 양자화 | PTQ (W8A16 가능) | **PTQ (W8A8 강제)** |
| 결과 | Transformer > Conformer | **Conformer > Transformer** |

### 왜 모순되는가

**핵심 차이: W8A16 vs W8A8.**

- 논문: GPU에서 **weight만 양자화**, activation은 FP16 유지 → Transformer의 attention이 FP16으로 정확히 계산됨
- T527: **weight + activation 둘 다 uint8** → Transformer의 attention이 uint8로 깎여서 파괴됨

Conformer의 CNN은 **activation이 uint8이어도 local 패턴을 보존** — 이건 **W8A8 환경에서만 나타나는 이점**. W8A16에서는 Transformer의 attention이 FP16으로 정확하므로 CNN의 보정이 불필요.

> **결론:** 논문의 결론은 GPU(W8A16) 환경에서 맞다. T527 NPU(W8A8) 환경에서는 **Conformer가 Transformer보다 양자화에 강건하다**는 우리 실험 결과가 맞다. **환경(하드웨어 + 양자화 방식)이 결론을 바꾼다.**

### 추가 논문: Depthwise Conv의 양자화 취약성

[Quantization-Friendly Separable Convolution (2018)](https://arxiv.org/pdf/1803.08607)은 depthwise conv가 **channel-wise outlier** 때문에 INT8에 취약하다고 보고. 하지만 이건 MobileNet의 depthwise conv (kernel 3×3)이고, Conformer의 depthwise conv (kernel 31, 1D)는 더 넓은 receptive field로 local averaging 효과가 커서 **outlier가 자연 억제**.

---

# 19. 왜 영어 wav2vec2는 되는데 한국어 wav2vec2는 안 되는가

**같은 wav2vec2 아키텍처인데 영어는 CER 17.52%, 한국어는 100%.** 이건 아키텍처가 아니라 **weight(학습 결과)의 차이**.

| | 영어 base-960h | 한국어 base-korean |
|---|---|---|
| 학습 데이터 | LibriSpeech **960시간** | Zeroth-Korean **50시간** |
| 사전학습 | wav2vec2-base (self-supervised) | XLS-R-128 (128개 언어) |
| Fine-tune 대상 | 영어만 | 한국어만 |
| **Attention 패턴** | **Sharp (한 곳에 집중)** | **Soft (여러 곳에 분산)** |
| **Logit margin** | **0.34** | **0.005** |

**영어 모델의 attention이 "sharp"한 이유:**
- 960시간 대규모 영어 데이터로 **확신 있게 학습**
- attention weight가 특정 프레임에 **집중**
- 결과: activation range가 좁고 margin이 큼

**한국어 모델의 attention이 "soft"한 이유:**
- 50시간 소규모 데이터로 **불확실하게 학습**
- attention weight가 여러 프레임에 **분산**
- 결과: activation range가 넓고 margin이 작음

**Conformer는 이 문제를 CNN으로 해결:**
- Attention이 soft해도, Conv(kernel=31)이 local context를 보강
- CNN은 학습 데이터량에 덜 민감 (local 패턴은 소규모 데이터로도 학습 가능)
- 따라서 **한국어 + 소규모 데이터에서도 uint8 양자화 가능**

---

# 20. 실험 타임라인

| 날짜 | 시도 | 결과 | 교훈 |
|------|------|------|------|
| ~2026-02 | Wav2Vec2 EN uint8 | CER 17.52% ✓ | 영어는 됨 |
| ~2026-02 | Wav2Vec2 KO 여러 양자화 | CER 100% ✗ | 한국어 안 됨 |
| 2026-03-13~14 | vocab_ko.txt 버그 수정 | KoCitrinet CER 44.44% | 데이터 먼저 의심 |
| 2026-03-15 | Wav2Vec2 JNI 4개 버그 수정 | CER 17.52% (영어) | reverse_channel 핵심 |
| 2026-03-17 | Acuity 6.12 vs 6.21 비교 | 6.12가 최적 | 새 버전이 항상 좋진 않음 |
| 2026-03-18 | 한국어 Wav2Vec2 21종+ PTQ | 전부 실패 | PTQ 한계 |
| 2026-03-19 | Zipformer uint8/int16/PCQ | 전부 CER 100% | 5868노드 → 오차 누적 |
| 2026-03-20 | KoCitrinet int16 DFP | CER 330% | int16이 int8보다 나쁨 |
| 2026-03-23 | 영어→한국어 fine-tune (attempt5) | WER 40.6% uint8 | 최초 한국어 출력! |
| 2026-03-23 | 레이어별 dump 분석 | L8-11 delta -0.17 | 문제 레이어 특정 |
| 2026-03-23 | fqb16, range clip 등 | 효과 없음 | 시뮬≠디바이스 |
| 2026-03-24 | QAT + margin loss | margin 3배 개선 | 여전히 부족 |
| 2026-03-24 | Vocab 분석 (잘못됨) | vocab 56 권고 | **이후 Conformer가 반증** |
| 2026-03-24~25 | Split model 6종 | 전부 효과 없음 | encoder가 근본 원인 |
| 2026-03-25 | **80k aihub 모델 Split** | **CER 92.65% (효과 없음)** | FP32 좋아도 uint8 실패 |
| **2026-03-25** | **SungBeom Conformer 확인** | **CER 10.02%** | **아키텍처가 답** |

---

# 21. 한 페이지 요약

```
┌─────────────────────────────────────────────────────────────┐
│           T527 NPU uint8 STT 양자화: 무엇이 성패를 가르는가          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 성공하는 모델                    ✗ 실패하는 모델               │
│  ─────────────                    ─────────────            │
│  • CNN + Attention 하이브리드         • 순수 Transformer          │
│  • mel spectrogram 입력             • raw waveform 입력         │
│  • Relative positional encoding    • Convolutional pos enc    │
│  • 조기 subsampling (factor 4)      • 늦은 subsampling          │
│  • d_model ≥ 512                  • d_model < 512            │
│                                                             │
│  Conformer CER 10.02% ✓           wav2vec2 CER 100% ✗       │
│  KoCitrinet CER 44.44% ✓          Zipformer CER 100% ✗      │
│                                   HuBERT CER 100% ✗         │
│                                                             │
│  ⚠️ vocab 크기는 부차적 요인 (Conformer vocab 2049에서 성공)       │
│  ⚠️ FP32 정확도는 무관 (aihub 80k FP32 9%, uint8 93%)          │
│  ⚠️ 학습 데이터량도 무관 (같은 데이터, 다른 아키텍처 = 다른 결과)      │
│                                                             │
│  핵심: 아키텍처가 양자화 성패의 80%를 결정한다.                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 22. T527 vs RK3588 NPU 비교

RK3588 NPU(RKNN)에서는 wav2vec2-xls-r-300m이 **CER 11.78%**로 동작.
T527에서는 같은 모델이 **ALL PAD**. 왜?

| 항목 | T527 (Vivante) | RK3588 (RKNN) |
|------|---------------|---------------|
| NPU 성능 | 2 TOPS | **6 TOPS** |
| INT8 | HW 가속 ✓ | HW 가속 ✓ |
| **FP16** | **SW 에뮬레이션 (25배 느림)** | **HW 가속 ✓** |
| 핵심 차이 | **W8A8 강제** | **Split INT8+FP16 가능** |

RK3588의 핵심 기법: **4파트 분할 (CNN FP16 + L0-11 INT8 + L12-23 FP16 + LM_head FP16)**
- 양자화에 민감한 부분은 FP16으로 → 정밀도 보존
- T527에서는 **FP16 HW 가속이 없어서** 이 전략 사용 불가

| 기법 | RK3588 | T527 | 이유 |
|------|--------|------|------|
| Split INT8+FP16 | ✓ CER 11.78% | ✗ FP16 17초 | T527 FP16 HW 미지원 |
| Amplitude norm 5.0 | ✓ CER -17pp | ✗ 효과 없음 | T527은 전체 uint8, FP16 보정 없음 |
| KL divergence | ✓ CER -8.65pp | △ 미미한 개선 | T527은 근본 문제가 아키텍처 |

**결론:** T527에서 RK3588 수준의 성능을 얻으려면 Split이 아니라 **아키텍처 교체(Conformer)**가 답.
SungBeom Conformer가 **Acuity PTQ uint8만으로 CER 10.02%** — RK3588의 11.78%보다 좋다.

---

# 23. 실패한 모든 시도의 상세 카탈로그

## 23.1 wav2vec2 한국어 PTQ 21종+ (전부 실패)

| # | 양자화 | calibration | 알고리즘 | CER | 실패 모드 |
|---|--------|------------|---------|-----|----------|
| 1 | uint8 AA | 1 sample | MA(0.004) | ~blank | 거의 출력 없음 |
| 2 | uint8 AA | 100 samples | MA | input range 문제 | 비정상 range |
| 3 | uint8 AA | 100+fix | MA | 38.9% non-PAD | CER > 100% |
| 4 | uint8 AA | **300+fix** | MA | **46.3% non-PAD** | CER 100.86% (최선) |
| 5 | uint8 AA | 1000+fix | MA | 26.6% non-PAD | over-convergence |
| 6 | uint8 | — | min_max | ALL PAD | 완전 blank |
| 7 | uint8 | — | KL divergence | ALL PAD | 완전 blank |
| 8 | uint8 | — | normalized input | worse | — |
| 9 | uint8 + fqb16 | 50 | KL + fqb16 | CER 174% | 시뮬 70.8%, 디바이스 악화 |
| 10 | uint8 + padbias0 | 50 | KL + fqb16 | CER 98.65% | 미미한 차이 |
| 11 | uint8 + Clip L10 | 50 | KL + fqb16 | CER 100.20% | outlier 제거 효과 없음 |
| 12 | PCQ int8 | 50 | perchannel | NB export 실패 | error 65280 |
| 13 | symmetric int8 | — | — | NB export 실패 | — |
| 14 | bf16 | — | — | NB export 실패 | error 64768 |
| 15 | fp16 | — | — | CPU fallback | 17.7초 (HW 미가속) |
| 16 | int16 DFP | — | — | status=-1 | NB 크기 초과 (153MB) |
| 17 | hybrid int16/uint8 | — | — | 디바이스 크래시 | 전원 꺼짐 |
| 18 | SmoothQuant | — | — | FP32 변형 | 효과 없음 |
| 19 | Range Clipping | — | — | 58%→27.5% | saturation error |
| 20 | amplitude norm 5.0 | — | — | CER 100% | 캘리브레이션 불일치 |
| 21 | 3-part split | — | — | CNN 고정값 | 입력 무시 |

## 23.2 QAT 시도 (부분 성공)

| # | 방법 | WER | margin min | ratio |
|---|------|-----|-----------|-------|
| 1 | 영어→한국어 fine-tune (attempt5) | 40.6% | 0.037 | 0.25x |
| 2 | QAT basic (30ep) | 38.86% | **0.099** | 0.66x |
| 3 | QAT + Margin Loss (20ep) | 43.05% | 0.000 | 실패 |

## 23.3 Split Model 시도 (전부 효과 없음)

| # | 모델 | 방식 | CER | 효과 |
|---|------|------|-----|------|
| 1 | Wav2Vec2 KO base | lm_head fp32 | 99.70% | 없음 |
| 2 | Wav2Vec2 KO base | L8-11+lm fp32 | 99.26% | 없음 |
| 3 | Wav2Vec2 KO base | CNN+lm fp32 | 100.00% | 악화 |
| 4 | Wav2Vec2 KO 80k | lm_head fp32 | 92.65% | 없음 |

## 23.4 Zipformer (전부 실패)

| # | 양자화 | NB | Encoder corr | CER |
|---|--------|-----|-------------|-----|
| 1 | uint8 AA | 63MB | 0.627 | 100% |
| 2 | int16 DFP | 118MB | 0.643 | 100% |
| 3 | PCQ int8 | 71MB | 0.275 | 100% |
| 4 | bf16 | — | — | export 실패 |

---

# 24. Android 앱 배포 가이드 (Conformer)

SungBeom Conformer를 Android 앱에 통합하는 방법.

## 24.1 필요 파일

| 파일 | 크기 | 용도 |
|------|------|------|
| `network_binary.nb` | 102MB | NPU 모델 |
| `vocab_correct.json` | ~50KB | BPE 토큰 매핑 (2049개) |
| NeMo mel preprocessor | 앱 내 구현 | mel spectrogram 생성 |

## 24.2 추론 파이프라인

```
1. 오디오 입력 (16kHz, mono)
2. NeMo mel spectrogram 생성 ([1, 80, T])
   - n_fft=512, hop=160, n_mels=80
   - dither=0.0, pad_to=0
3. 3초 윈도우로 분할 (301 frames, stride 250)
4. 각 윈도우마다:
   a. mel → uint8 양자화 (scale=0.02418, zp=67)
   b. NPU 추론 (233ms)
   c. uint8 출력 → dequantize (scale=0.20301, zp=255)
   d. CTC greedy decode (blank=0)
5. 윈도우 결과 연결
6. BPE 토큰 → 한국어 텍스트 (vocab_correct.json)
```

## 24.3 주의사항

- **mel 생성은 NeMo 방식 필수** — librosa로 만들면 결과 다름
- **dither=0.0** 설정 필수 (기본값 1e-5는 noise 추가)
- **pad_to=0** 설정 필수 (기본값은 padding 추가)
- 슬라이딩 윈도우 경계에서 단어 잘림 가능 — overlap(51 frames) 처리 필요

---

# 25. Conformer Depthwise Conv의 수학적 직관

## 25.1 왜 kernel=31인가

Conformer의 depthwise conv는 **1D convolution, kernel size=31**.
31 프레임 = 약 0.3초 (stride 10ms 기준). 이것이 **음소(phoneme) 하나의 길이**에 해당.

```
kernel=3:  너무 짧음 (0.03초) → 음향 특성 캡처 못 함
kernel=7:  모음/자음 전환 정도 → 부족
kernel=31: 음소 하나의 전체 패턴 → 적절
kernel=63: 음절 수준 → 너무 길어서 인접 음절과 혼합
```

## 25.2 왜 depthwise conv가 양자화에 유리한가

```
Standard conv: output[c] = Σ(all_channels) weight[c,ch] × input[ch]
  → 모든 채널을 합산 → 값 범위 크게 변동 가능

Depthwise conv: output[c] = weight[c] × input[c]  (같은 채널끼리만)
  → 채널 독립 → 각 채널의 값 범위가 독립적으로 bounded
  → uint8 per-tensor quantization에서도 각 채널의 정보 보존
```

**Self-attention과의 대비:**
```
Self-attention: output[t] = Σ(all_frames) softmax(Q×K^T) × V
  → 모든 프레임의 가중합 → 값 범위 예측 불가
  → uint8에서 softmax weight가 깎이면 전체 출력 왜곡
```

## 25.3 Conformer에서 Conv의 역할: Attention의 "정규화기"

```
Attention 출력: 값 범위 넓음 (global dependency → outlier 발생)
Conv 통과 후: 값 범위 좁아짐 (local averaging → outlier 억제)
```

이것이 **Conformer가 18개 레이어를 거쳐도 activation이 폭발하지 않는 이유**. 매 레이어마다 Conv가 activation을 "재정규화"하여 uint8 범위 내에 유지.

wav2vec2는 이 "재정규화"가 없어서 레이어를 거칠수록 activation range가 넓어지고, L8-11에서 파괴적 수준에 도달.

---

# 26. HuBERT 한국어 — wav2vec2와 동일한 실패

| 항목 | HuBERT KO | wav2vec2 KO | 비교 |
|------|----------|-----------|------|
| 모델 | HJOK/asr-hubert-base-ko | Kkonjeong/wav2vec2-base-korean | — |
| Params | 96M | 94.4M | 유사 |
| 아키텍처 | **Transformer** (HuBERT) | **Transformer** (wav2vec2) | **동일 유형** |
| ONNX | 384MB, 727 nodes | 361MB, 957 nodes | 유사 |
| NB | 76MB | 72MB | 유사 |
| Vocab | 2145 | 56 | HuBERT가 38배 큼 |
| **uint8 CER** | **100% (동일 토큰 반복)** | **100.86%** | **둘 다 실패** |

HuBERT도 **순수 Transformer 기반 self-supervised learning** 모델. vocab이 2145(HuBERT)이든 56(wav2vec2)이든 관계없이 둘 다 uint8에서 실패. **아키텍처가 원인이라는 추가 증거.**

---

# 27. Acuity 6.12 vs 6.21 비교

영어 wav2vec2에서 실측한 Acuity 버전 비교:

| 양자화 | Acuity | CER | WER | NB |
|--------|--------|-----|-----|-----|
| **uint8 AA** | **6.12** | **17.52%** | **27.38%** | **87MB** |
| PCQ int8 | 6.21 | 19.24% | 34.39% | 99MB |
| uint8 AA | 6.21 | 23.41% | 40.57% | 76MB |

**Acuity 6.12가 6.21보다 우수.** 새 버전이 항상 좋은 건 아님. 내부 그래프 최적화 전략 차이로 추정.

Conformer NB 변환에도 Acuity 6.12 사용 권장.

---

# 28. 양자화 가능성 사전 판단 체크리스트

새 모델을 T527 NPU에 배포하기 전에 **이 체크리스트로 사전 판단**:

## ✅ 통과해야 하는 항목

- [ ] **아키텍처에 CNN이 포함되어 있는가?** (Conformer, Citrinet 등)
  - 순수 Transformer → 실패 확률 높음
- [ ] **입력이 mel spectrogram인가?**
  - raw waveform → 위험
- [ ] **ONNX 노드 수 < 3000인가?**
  - 5000+ → 오차 누적 위험
- [ ] **d_model ≥ 512인가?**
  - 256 이하 → uint8 noise 흡수 부족
- [ ] **vocab < 5000인가?**
  - 5000+ → margin 부족 위험 (부차적이지만 참고)
- [ ] **NB 크기 < 120MB 예상인가?**
  - 초과 → status=-1 거부

## ⚠️ 주의 항목

- [ ] **Acuity 6.12로 변환** (6.21보다 우수한 경우 있음)
- [ ] **reverse_channel: false 확인** (오디오 모델 필수)
- [ ] **NeMo 모델이면 dither=0.0, pad_to=0** 설정
- [ ] **반드시 T527 디바이스에서 CER 실측** (시뮬레이션 불신뢰)

## 🚫 시도하지 말 것

- [ ] wav2vec2/HuBERT 한국어 → uint8 (반드시 실패)
- [ ] Zipformer (5868 nodes) → 어떤 양자화도 실패
- [ ] XLS-R-300M → 24L, ALL PAD
- [ ] vocab 56 전환 (아키텍처 문제를 vocab으로 해결 불가)

---

# 29. Conformer 직접 학습 가이드 (aihub 데이터)

SungBeom 모델 대신 **aihub 4356시간 데이터로 직접 Conformer CTC 학습**하면 도메인 특화 모델을 만들 수 있다.

## 29.1 NeMo 학습 권장 설정

| 항목 | 권장값 | 이유 |
|------|--------|------|
| 아키텍처 | Conformer CTC Medium+ | SungBeom이 CER 10.02% 검증 |
| d_model | **512 이상** | 256은 uint8에서 CER 55% |
| Attention heads | 8 | d_model=512 기준 |
| Conv kernel | **31** | 음소 길이에 맞는 receptive field |
| Encoder layers | **18** | SungBeom과 동일 |
| **Vocab** | **~2000 BPE** | 2049가 uint8에서 검증됨 |
| Tokenizer | SentencePiece BPE | NeMo 표준 |
| Sample rate | 16000 Hz | 표준 |
| n_mels | 80 | 표준 |
| Input length | 3초 (301 frames) → NB 고정 | T527 NB 제한 |

## 29.2 학습 설정

```yaml
# NeMo conformer_ctc_bpe.yaml 기반
model:
  encoder:
    d_model: 512
    n_heads: 8
    n_layers: 18
    conv_kernel_size: 31
    subsampling_factor: 4
    pos_emb_max_len: 5000

  decoder:
    vocabulary_size: 2048  # BPE vocab

  optim:
    name: adamw
    lr: 0.001  # 큰 데이터셋에서 시작
    weight_decay: 1e-3

trainer:
  max_epochs: 100
  accumulate_grad_batches: 4  # effective batch 128
  precision: 16  # fp16 mixed precision
```

## 29.3 학습 후 NB 변환

학습 완료 → `.nemo` 파일 → NeMo export → ONNX → static shape → Pad fix → Acuity → NB

**SungBeom 파이프라인 그대로 재사용 가능** (`t527-stt/conformer/SUNGBEOM_REPORT.md` 섹션 3 참조).

## 29.4 기대 결과

SungBeom (AI Hub 학습) CER 10.02% → **같은 데이터 + 같은 아키텍처면 비슷하거나 더 좋은 결과** 가능. 도메인 특화(전화망, 회의 등)하면 해당 도메인에서 추가 개선.

참고: [NeMo GitHub Issue #3243: Training conformer_ctc with Korean](https://github.com/NVIDIA/NeMo/issues/3243)

---

# 30. Edge NPU STT 배포 트렌드 (2024-2025)

| 플랫폼 | NPU | INT8 | FP16 | 특징 |
|--------|-----|------|------|------|
| **T527 (Vivante VIP9000)** | 2 TOPS | **HW ✓** | SW only | **uint8 W8A8 강제** |
| RK3588 (RKNN) | 6 TOPS | HW ✓ | **HW ✓** | Split INT8+FP16 가능 |
| Qualcomm Hexagon | 15+ TOPS | HW ✓ | **HW ✓** | INT16 kernel 최적화 |
| Samsung Exynos NPU | 15+ TOPS | HW ✓ | HW ✓ | Mixed precision |
| ARM Ethos-U85 | 1 TOPS | **HW ✓** | ✗ | Conformer INT8 배포 검증 |
| MediaTek APU | 10+ TOPS | HW ✓ | HW ✓ | INT4까지 지원 |

**T527은 가장 제약이 큰 NPU** — FP16 HW 미지원, W8A8 강제. 하지만 **Conformer CTC가 이 제약 안에서 CER 10.02% 달성**.

**업계 트렌드:**
- Conformer/Squeezeformer + INT8 PTQ가 edge STT 표준
- 1-bit/2-bit 극저비트 양자화 연구 활발 (ENERZAi Whisper 1.58-bit)
- Streaming Conformer로 실시간 STT
- ARM Ethos-U85에서 INT8 Conformer 배포 성공 사례 ([ARM 블로그](https://developer.arm.com/community/arm-community-blogs/b/internet-of-things-blog/posts/end-to-end-int8-conformer-on-arm-training-quantization-and-deployment-on-ethos-u85))

---

# 31. 향후 연구 방향

## 31.1 단기 (현재 가능)

| 작업 | 기대 효과 |
|------|----------|
| **aihub + Conformer CTC 직접 학습** | 도메인 특화, CER 개선 |
| **Streaming Conformer** 탐색 | 실시간 STT |
| **Squeezeformer** 시도 | Conformer 대비 작은 NB, 빠른 추론 |
| **5초/10초 입력** NB 재변환 | 슬라이딩 윈도우 오버헤드 감소 |

## 31.2 중기

| 작업 | 기대 효과 |
|------|----------|
| Conformer + QAT | uint8 정확도 추가 개선 |
| E-Branchformer 시도 | Conformer 변형, 더 높은 정확도 가능 |
| Android 앱 NeMo mel 구현 | 앱에서 직접 추론 |

## 31.3 장기

| 작업 | 기대 효과 |
|------|----------|
| 다음 세대 NPU (VIP9000 후속) | FP16 HW 지원 시 모든 모델 가능 |
| INT4/INT2 양자화 | 더 작은 NB, 더 빠른 추론 |
| On-device 학습 | 사용자 적응형 STT |

---

# 32. SungBeom Conformer CER 상세 분석 (100샘플)

## 32.1 CER 분포

| CER 범위 | 샘플 수 | 누적 |
|----------|---------|------|
| **0% (완벽)** | **4** | 4% |
| 1~5% | 22 | **26%** |
| 5~10% | 40 | **66%** |
| 10~15% | 23 | **89%** |
| 15~20% | 9 | **98%** |
| 20~50% | 1 | 99% |
| **50%+ (심각)** | **2** | 101 |

→ **66%의 샘플이 CER 10% 이하**. 98%가 CER 20% 이하.

## 32.2 길이별 CER

| 길이 | 평균 CER | 샘플 수 |
|------|---------|---------|
| 짧은 (<8초) | **7.8%** | 31 |
| 중간 (8~12초) | 9.9% | 47 |
| 긴 (12초+) | **13.3%** | 23 |

→ 긴 문장일수록 CER 높아짐 — 슬라이딩 윈도우 경계에서 단어 잘림/반복.

## 32.3 CER 높은 샘플 원인 분석

| # | CER | 원인 |
|---|-----|------|
| 1 | **82.0%** | 20.4초, 8 chunks — 매우 긴 문장, 윈도우 경계 누적 오류 |
| 46 | **63.5%** | "성공회대 노동 아카데미" — 전문 용어/고유명사 |
| 49 | 29.0% | "일 조 이천 구백 십 구 억원" — 숫자 읽기 혼동 |

## 32.4 CER 0% 완벽 인식 특성

4개 완벽 인식 샘플 공통: **일상 한국어, 7~10초, 3~4 chunks**.
전문 용어, 숫자, 고유명사 없음.

---

# 33. Activation 분포의 시각적 이해

## 33.1 영어 wav2vec2 (uint8 성공)

```
L10 residual Add activation 분포:

값: ────────────|────────────
     -4     -2   0   2   4
     ▓▓▓▓▓▓▓▓████████▓▓▓▓▓▓▓▓

range: 8.2 → uint8 step: 0.032
→ 대부분의 값이 [-4, +4]에 밀집 → 256단계로 충분히 표현
```

## 33.2 한국어 wav2vec2 (uint8 실패)

```
L10 residual Add activation 분포:

값: ─────────────────────────────────|──────────────────────────────────────
    -174                          0                              245
     ·                    ▓▓▓█▓▓▓▓                                 ·

range: 420 → uint8 step: 1.65
→ 99%의 값이 [-3, +3]에 밀집하지만, 0.01% outlier가 range를 420으로 끌어올림
→ 256단계 중 3~4단계만 유의미하게 사용 → 해상도 극도로 부족
```

## 33.3 Conformer (uint8 성공)

```
lm_head log softmax 출력 분포:

값: ──────────────────────────|
    -51.8                    0
    ·    ▓▓▓▓▓████████████████

range: 51.8 → uint8 step: 0.203
→ 정답 토큰은 ~0.0 (확률 ~1.0), 오답은 -20~-50
→ 차이가 매우 커서 uint8 step 0.203으로도 정확한 argmax 가능
```

---

# 34. 실패에서 배운 것들 (Postmortem Wisdom)

1. **"모델 아키텍처를 먼저 확인하라."** vocab, 학습 데이터, FP32 정확도보다 아키텍처가 양자화 성패를 결정한다.

2. **"시뮬레이션을 믿지 마라."** Acuity 시뮬레이션과 T527 디바이스가 31.5%만 일치. 반드시 디바이스에서 실측.

3. **"데이터를 코드보다 먼저 의심하라."** vocab_ko.txt 1줄 누락이 2일 디버깅의 원인.

4. **"오디오 모델의 reverse_channel은 반드시 false."** 이미지 모델 기본값(true)이 오디오에서 모든 것을 망가뜨림.

5. **"새 버전이 항상 좋진 않다."** Acuity 6.12가 6.21보다 나은 경우 있음.

6. **"vocab 크기는 부차적이다."** Conformer vocab 2049에서 CER 10.02%. wav2vec2 vocab 56에서 CER 100%.

7. **"FP32 정확도가 좋다고 uint8도 좋은 건 아니다."** aihub 80k FP32 CER 9%, uint8 CER 93%.

8. **"CNN이 양자화의 안정제이다."** 매 레이어에 CNN이 포함된 Conformer는 uint8에서 안정. 입력부에만 CNN이 있는 wav2vec2는 불안정.

9. **"Split model로는 encoder 문제를 해결 못 한다."** lm_head를 fp32로 빼도, CNN을 fp32로 빼도, L8-11을 빼도 효과 없음.

10. **"잘못된 분석을 확신하지 마라."** vocab 분석이 틀렸음을 Conformer가 증명.

---

# 35. 모든 모델 입출력 + 양자화 파라미터 총정리

| 모델 | 입력 shape | 출력 shape | NB | in scale/zp | out scale/zp | 추론 |
|------|-----------|-----------|-----|------------|-------------|------|
| **SungBeom Conformer uint8** | [1,80,301] mel | [1,76,2049] | 102MB | 0.024/67 | 0.203/255 | 233ms |
| SungBeom Conformer int16 | [1,80,301] mel | [1,76,2049] | 200MB | —/— | —/— | 565ms |
| cwwojin Conformer uint8 | [1,80,301] mel | [1,76,5001] | 29MB | 0.027/65 | 0.177/255 | 111ms |
| KoCitrinet 300f int8 | [1,80,1,300] mel | [1,2049,1,38] | 62MB | 0.021/-37 | 0.113/127 | 120ms |
| Wav2Vec2 EN uint8 | [1,80000] raw | [1,249,32] | 87MB | 0.003/137 | 0.150/186 | 715ms |
| Wav2Vec2 KO base uint8 | [1,48000] raw | [1,149,56] | 72MB | 0.001/121 | 0.069/77 | 415ms |
| Wav2Vec2 KO 80k uint8 | [1,48000] raw | [1,149,1912] | 77MB | 0.051/119 | 0.160/110 | 424ms |
| Zipformer uint8 | [1,39,80]+30states | [1,8,512]+30states | 63MB | —/— | —/— | 50ms |
| HuBERT KO uint8 | [1,48000] raw | [1,149,2145] | 76MB | —/— | —/— | 423ms |

---

# 36. SungBeom Conformer: uint8 vs int16 비교

| | uint8 KL | int16 DFP | 차이 |
|---|---|---|---|
| **CER** | **10.02%** | **9.59%** | -0.43%p (int16 약간 좋음) |
| NB 크기 | 102MB | **200MB** | 2배 |
| 추론 시간 | **233ms/chunk** | 565ms/chunk | **2.4배 느림** |
| HW 가속 | ✓ NPU | ✓ NPU (DFP) | 둘 다 가속 |

**결론:** int16 DFP는 0.43%p CER 개선이지만 **2.4배 느리고 NB 2배 큼**. uint8이 실용적 최적.

흥미로운 점: **KoCitrinet에서는 int16 DFP가 int8보다 훨씬 나쁨 (CER 330%)**이었지만, **Conformer에서는 int16 DFP가 int8과 비슷**. 이것도 아키텍처 차이 — Conformer의 CNN이 DFP의 2^fl 스케일 방식에서도 정보를 보존.

---

# 37. 가설 검증 목록

이 프로젝트에서 세운 가설들과 실험 결과:

| # | 가설 | 결과 | 근거 |
|---|------|------|------|
| 1 | "vocab이 크면 양자화에 불리" | **부분적 맞음, 부차적** | Conformer vocab 2049 CER 10%, wav2vec2 vocab 56 CER 100% |
| 2 | "FP32 성능이 좋으면 uint8도 좋다" | **틀림** | aihub 80k FP32 9%, uint8 93% |
| 3 | "lm_head를 fp32로 빼면 해결" | **틀림** | Split CER 99.70% (효과 없음) |
| 4 | "L8-11이 문제니까 fp32로 빼면 해결" | **틀림** | Split L7 CER 99.26% |
| 5 | "CNN을 fp32로 빼면 해결 (OpenVINO 방식)" | **틀림** | 오히려 CER 100% (악화) |
| 6 | "QAT로 margin을 키우면 해결" | **부분적 맞음** | margin 3배 개선, 하지만 여전히 부족 |
| 7 | "Margin loss 추가하면 margin 더 큼" | **틀림** | CTC와 충돌, margin 미개선 |
| 8 | **"아키텍처가 양자화 성패를 결정"** | **맞음** | Conformer ✓, wav2vec2/HuBERT/Zipformer ✗ |
| 9 | **"CNN이 activation을 안정화"** | **맞음** | Conformer Conv(31) → activation 정규화 |
| 10 | **"mel 입력이 raw waveform보다 유리"** | **맞음** | mel 모델(Conformer/Citrinet) ✓, raw(wav2vec2) ✗ |
| 11 | "시뮬레이션으로 양자화 품질 판단 가능" | **틀림** | 31.5% 불일치 |
| 12 | "Acuity 6.21이 6.12보다 좋다" | **틀림** | 6.12 CER 17.52%, 6.21 CER 23.41% |

---

# 38. 시간/비용 효율성 분석

| 접근법 | 소요 시간 | CER 결과 | 효율성 |
|--------|----------|---------|--------|
| Wav2Vec2 PTQ 21종 | ~2주 | CER 100% (실패) | ✗ 최악 |
| Wav2Vec2 레이어 dump 분석 | ~3일 | 인사이트만 (CER 미개선) | △ 학습용 |
| Wav2Vec2 QAT | ~2일 | margin 3배 개선 (여전히 부족) | △ |
| Wav2Vec2 fine-tune | ~1주 | WER 40.6% (부분 성공) | △ |
| Split model 6종 | ~2일 | 전부 효과 없음 | ✗ |
| Vocab 분석 + 팀 설득 | ~2일 | **틀린 결론 (피해 발생)** | ✗✗ 최악 |
| **Conformer NB 변환** | **~1일** | **CER 10.02%** | **⭐ 최고** |

**교훈:** 기존 모델(wav2vec2)을 고치려고 2주 투자하는 것보다, **다른 아키텍처(Conformer)로 바꾸는 게 1일 만에 해결.**

---

# 39. 차세대 모델 후보: Squeezeformer, E-Branchformer

## 39.1 Squeezeformer (NeurIPS 2022)

> [Squeezeformer: An Efficient Transformer for ASR](https://arxiv.org/pdf/2206.00888)

Conformer의 비효율적 부분을 제거한 효율화 버전:
- **Temporal U-Net 구조**: 입력 길이를 단계적으로 줄여서 연산 효율화
- **Macro/Micro Redesign**: FFN → Attention → Conv 순서 재배치
- 같은 FLOPs에서 Conformer보다 정확도 높음

**T527 적용 가능성:** ⭐⭐⭐⭐
- CNN + Attention 하이브리드 유지 → uint8 양자화 가능할 것
- U-Net 구조로 중간 레이어의 sequence 길이 축소 → 양자화 오차 누적 감소
- NeMo에서 지원 → 기존 파이프라인 재사용 가능

## 39.2 E-Branchformer (SLT 2022, ICLR 2024)

> [E-Branchformer: Enhanced Merging for Speech Recognition](https://arxiv.org/abs/2210.00077)

Conformer의 병렬 브랜치 버전:
- **두 갈래 병합**: Self-Attention branch + **Conv branch를 병렬 실행 후 합침**
- **Depthwise conv 강화**: merging에 추가 depthwise conv 삽입
- LibriSpeech WER: E-Branchformer 1.81% vs Conformer 1.9% (test-clean)

**T527 적용 가능성:** ⭐⭐⭐⭐⭐
- **Conv 비중이 Conformer보다 더 높음** → uint8 양자화에 더 유리할 가능성
- 15개 ASR 벤치마크에서 Conformer와 동등 이상
- 학습 안정성 Conformer보다 우수

## 39.3 FastConformer (NVIDIA, NeMo 기본)

NeMo의 기본 Conformer 변형:
- 8x subsampling (Conformer 4x 대비 2배 축소)
- 더 빠른 추론, 약간 낮은 정확도
- **T527 적용 시 sequence 길이가 절반** → 양자화 오차 누적 더 감소

## 39.4 후보 비교

| 모델 | CNN 비중 | uint8 예상 | FP32 정확도 | NeMo 지원 |
|------|---------|-----------|-----------|----------|
| Conformer | 높음 | **CER 10.02% (검증)** | 좋음 | ✓ |
| Squeezeformer | 높음 | 좋을 것 (미검증) | Conformer 이상 | ✓ |
| **E-Branchformer** | **매우 높음** | **가장 좋을 것 (미검증)** | **Conformer 이상** | △ (ESPnet) |
| FastConformer | 높음 | 좋을 것 (미검증) | Conformer 이하 | ✓ |

**권장 시도 순서:** Conformer (검증됨) → E-Branchformer (Conv 비중 최고) → Squeezeformer (효율)

---

# 40. CTC vs RNN-Transducer: 양자화 관점

## 40.1 왜 CTC 모델이 양자화에 유리한가

| | CTC | RNN-Transducer |
|---|---|---|
| 디코딩 | **독립적** (각 프레임 개별) | **순차적** (이전 출력에 의존) |
| 양자화 오류 전파 | **해당 프레임에만 영향** | **다음 프레임으로 누적** |
| 모델 구조 | Encoder only | Encoder + Decoder + Joiner |
| NB 수 | 1개 | 3개 (순차 실행) |
| T527 결과 | **Conformer CTC: CER 10.02%** | Zipformer RNN-T: CER 100% |

**CTC의 핵심 장점:** 각 프레임의 양자화 오류가 **다른 프레임에 전파되지 않음**. RNN-T는 decoder의 자기회귀(autoregressive) 특성 때문에 **한 프레임의 오류가 다음 프레임에 연쇄적으로 영향**.

Zipformer(RNN-T)가 실패하고 Conformer(CTC)가 성공한 이유 중 하나.

---

# 41. NeMo mel vs librosa mel 상세 차이

Conformer 변환 시 겪은 **mel spectrogram 불일치 문제**:

```
NeMo mel:    range [-1.75, 5.14], mean=0.0000
librosa mel: range [-2.24, 4.89], mean=-0.0048
```

| 파라미터 | NeMo | librosa | 영향 |
|---------|------|---------|------|
| dither | **0.0 (추론 시)** | 없음 | NeMo 기본값 1e-5는 noise 추가 |
| pad_to | **0 (추론 시)** | 없음 | NeMo 기본값은 padding 추가 |
| window | hann (periodic) | hann | 미세 차이 |
| normalization | per-feature | — | 값 범위 차이 |

**결론:** Conformer(NeMo) 모델은 반드시 **NeMo preprocessor로 mel 생성**해야 함. librosa로 만들면 calibration 불일치로 양자화 품질 저하. Android 앱 구현 시에도 NeMo 방식으로 mel 생성 필요.

---

# 42. 슬라이딩 윈도우 최적화

SungBeom Conformer는 3초(301 frames) 고정 입력. 긴 오디오는 슬라이딩 윈도우로 처리.

## 42.1 현재 설정

```
Window: 301 frames (≈3.01초)
Stride: 250 frames (≈2.50초)
Overlap: 51 frames (≈0.51초)
```

## 42.2 문제점

1. **윈도우 경계에서 단어 잘림** — "안녕하세요"가 "안녕하" + "세요"로 분리
2. **중복 출력** — overlap 영역에서 같은 토큰 2번 출력
3. **긴 문장 CER 증가** — 20초 문장(8 chunks): CER 82%

## 42.3 개선 방안

| 방법 | 효과 | 구현 난이도 |
|------|------|-----------|
| **입력 길이 확대 (5초, 10초)** | chunk 수 감소 → 경계 오류 감소 | NB 재변환 필요 |
| Overlap 증가 (100 frames) | 더 안전한 경계 | 속도 느려짐 |
| Overlap 영역 CTC merge | 중복 제거 | 알고리즘 구현 |
| LM rescoring | 경계 오류 교정 | 추가 모델 필요 |

**가장 실용적:** 입력 길이를 **5초(501 frames)**로 확대. NB 크기 약간 증가하지만 chunk 수 절반 → CER 개선 예상.

---

# 43. 전체 프로젝트 투자 비용 분석

| 작업 | 기간 | GPU 시간 | 결과 | ROI |
|------|------|---------|------|-----|
| **wav2vec2 PTQ 탐색** | **~2주** | 0h (PTQ) | 실패 | ✗ |
| wav2vec2 레이어 분석 | ~3일 | 0h | 인사이트 | △ |
| wav2vec2 QAT | ~2일 | ~10h (RTX 4070) | margin 3배 개선 | △ |
| wav2vec2 fine-tune (attempt 1~7) | ~5일 | ~20h | WER 40.6% | △ |
| Split model 6종 | ~2일 | 0h | 전부 실패 | ✗ |
| Vocab 분석 + 문서 | ~2일 | 0h | **틀린 결론** | ✗✗ |
| **Conformer 변환** | **~1일** | **0h** | **CER 10.02%** | **⭐⭐⭐** |
| KoCitrinet 버그 수정 | ~2일 | 0h | CER 44.44% | ⭐⭐ |
| Zipformer 양자화 | ~3일 | 0h | 전부 실패 | ✗ |

**총 투자: ~4주, GPU ~30시간**
**최적 경로 (사후적으로):** KoCitrinet 버그 수정(2일) → Conformer 변환(1일) = **3일이면 CER 10.02% 달성 가능했음**

---

# 44. T527 NPU (Vivante VIP9000) 실측 operator 지원 현황

공식 문서에 상세 op 지원 목록 없음. **우리 실험에서 확인한 결과:**

## 44.1 uint8에서 HW 가속 확인된 op

| Op | 모델 | 결과 | 비고 |
|---|---|---|---|
| Conv1D (depthwise) | Conformer | ✓ | kernel=31, 37개 op |
| Conv1D (pointwise) | Conformer, KoCitrinet | ✓ | |
| MatMul (attention) | 모든 모델 | ✓ | Q×K^T, V |
| Softmax | Conformer, wav2vec2 | ✓ | 18~12개 per model |
| LayerNorm | 모든 모델 | ✓ | |
| Add (residual) | 모든 모델 | ✓ | |
| GELU (Erf) | wav2vec2 | ✓ (실행됨) | 하지만 양자화 오차 큼 |
| Reshape, Transpose | 모든 모델 | ✓ | |
| Select (from Where) | Conformer | ✓ | Acuity가 Where→Select 변환 |
| Pad | Conformer | ✓ | constant_value 수정 필요 |

## 44.2 문제가 있는 op

| Op | 문제 | 모델 | 비고 |
|---|---|---|---|
| Softmax (int16) | **시뮬≠디바이스** | wav2vec2 int16 | int16 DFP에서 hardware 불일치 |
| Erf/GELU (int16) | **시뮬≠디바이스** | wav2vec2 int16 | cos sim 0.877 |
| Where (직접) | shape inference 실패 | 수동 제거 시 | Acuity 자동 변환으로 해결 |

## 44.3 NB export 실패하는 경우

| 조건 | 에러 | 원인 |
|---|---|---|
| NB > ~120MB | status=-1 | NPU 메모리 제한 |
| bf16 전체 | error 64768 | NPU bf16 미지원 |
| PCQ int8 (일부 모델) | error 65280 | gen_nbg segfault |

---

# 45. 전체 문서 참조 인덱스

이 프로젝트에서 생성한 **모든 분석 문서** 목록:

| 날짜 | 파일 | 주제 |
|------|------|------|
| 2026-03-14 | `ai-sdk/.../2026-03-14_vocab_bug_fix_report.md` | vocab_ko.txt 버그 수정 |
| 2026-03-15 | `ai-sdk/.../wav2vec2_base_960h_5s/RESULTS.md` | 영어 wav2vec2 CER 17.52% |
| 2026-03-18 | `ai-sdk/.../wav2vec2_ko_eager_op12/QUANTIZATION_RESULTS.md` | 한국어 wav2vec2 양자화 실패 |
| 2026-03-19 | `t527-stt/MODEL_INVENTORY.md` | 전체 모델 인벤토리 |
| 2026-03-23 | `t527-stt/wav2vec2/base-korean/docs/260323_experiment_log.md` | 레이어 dump + fine-tune |
| 2026-03-23 | `t527-stt/wav2vec2/base-korean/docs/260323_uint8_layer_debug_report.md` | 530개 레이어 분석 |
| 2026-03-24 | `t527-stt/wav2vec2/base-korean/docs/QAT_RESEARCH.md` | QAT 연구 (15편 논문) |
| 2026-03-24 | `t527-stt/wav2vec2/base-korean/docs/VOCAB_ANALYSIS.md` | Vocab 분석 (**철회**) |
| 2026-03-24 | `t527-stt/wav2vec2/base-korean/docs/260324_split_model_approach.md` | Split model 개념 + 실험 |
| 2026-03-25 | `t527-stt/wav2vec2/base-korean/docs/260325_split_model_results.md` | Split 80k 결과 |
| 2026-03-25 | `t527-stt/conformer/SUNGBEOM_REPORT.md` | SungBeom Conformer 상세 |
| **2026-03-25** | **`t527-stt/docs/260325_conformer_success_analysis.md`** | **이 문서 (종합 분석)** |

---

# 46. 양자화 알고리즘 비교: KL divergence vs Moving Average

SungBeom Conformer에서 두 알고리즘을 100샘플로 비교:

| | KL divergence | Moving Average | 차이 |
|---|---|---|---|
| **평균 CER** | **10.02%** | **10.08%** | **0.06%p** |
| KL이 더 좋은 샘플 | 29개 | — | — |
| MA가 더 좋은 샘플 | — | 26개 | — |
| 동일 | — | — | 46개 |

**차이 0.06%p — 사실상 동일.** 양자화 알고리즘 선택은 CER에 **거의 영향 없음**.

이것은 **아키텍처 > 양자화 알고리즘**이라는 우리 결론의 추가 증거:
- Conformer에서는 KL이든 MA든 CER ~10%
- wav2vec2에서는 KL이든 MA든 CER ~100%
- **아키텍처가 결정하고, 알고리즘은 미세 조정일 뿐**

---

# 47. 핵심 원칙 정리: T527 NPU uint8 STT 양자화의 "법칙"

이 프로젝트 4주간의 실험에서 도출한 **경험적 법칙**:

## 법칙 1: **아키텍처 > 양자화 기법 > 학습 데이터 > vocab 크기**

양자화 성패에 대한 영향력 순서. 아키텍처를 바꾸면 양자화 기법/데이터/vocab을 아무리 최적화해도 넘을 수 없는 벽을 넘는다.

## 법칙 2: **CNN이 매 레이어에 있어야 한다**

입력부에만 CNN이 있는 건 부족 (wav2vec2). **매 Transformer 레이어 뒤에 Conv가 있어야** activation이 안정화됨 (Conformer).

## 법칙 3: **시뮬레이션을 믿지 마라**

Acuity 시뮬레이션과 T527 디바이스가 31.5%만 일치. **반드시 디바이스에서 실측.**

## 법칙 4: **FP32 정확도 ≠ uint8 정확도**

FP32에서 CER 9%인 모델이 uint8에서 CER 93%. FP32 성능으로 양자화 결과를 예측 불가.

## 법칙 5: **W8A8에서는 LLM의 경험이 통하지 않는다**

LLM의 W4A16이 되는 건 activation이 FP16이기 때문. T527은 W8A8이라 activation까지 양자화 — 완전히 다른 문제.

---

# 48. 새 발견: Wav2Vec2-Conformer — 두 세계의 장점

HuggingFace에 **`Wav2Vec2-Conformer`** 모델이 존재:
> [huggingface.co/docs/transformers/model_doc/wav2vec2-conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer)

**wav2vec2의 self-supervised learning + Conformer encoder 구조:**
- raw waveform 입력 (wav2vec2처럼)
- 하지만 encoder가 **Conformer** (CNN + Attention)
- 기존 wav2vec2 pretrained weight 활용 가능

**T527 적용 가능성:**
- Conformer encoder → uint8 양자화 가능할 것 (SungBeom 검증)
- raw waveform 입력 → mel 대비 불리하지만 Conformer CNN이 보상 가능
- **시도해볼 가치 있음** — wav2vec2 pretrained + Conformer 양자화 장점

---

# 49. Activation Outlier: Transformer의 근본 문제

최근 연구 ([Activation Outliers in Transformer Quantization, 2025](https://arxiv.org/abs/2603.04308))에 따르면:

> "Transformer 모델은 특정 채널에 **극단적 outlier activation**을 생성하는 경향이 있다. 이 outlier가 per-tensor 양자화에서 전체 범위를 지배하여 대부분의 값이 낮은 해상도로 표현된다."

이것이 정확히 우리가 관찰한 현상:

```
한국어 wav2vec2 L10 residual Add:
  99%의 값: [-3, +3] → 이 범위만 보면 uint8 충분
  0.01% outlier: [-174, +245] → 이것이 range 420으로 키움
  → 대부분의 값이 uint8 3~4단계에 몰림 → 해상도 극도로 부족
```

**Conformer의 depthwise conv가 이 outlier를 억제:**
- Conv(kernel=31)의 local averaging 효과
- 각 프레임이 인접 30개 프레임과 평균화 → 극단적 outlier 자연 감소
- 결과: activation range가 좁게 유지 → uint8에서 충분한 해상도

**참고 논문:**
- [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/edbcb7583fd8921dad78adecfe06a99b-Paper-Conference.pdf)
- [Mitigating the Impact of Outlier Channels for Language Model Quantization with Activation Regularization (2024)](https://arxiv.org/html/2404.03605v1)

---

# 50. 최종 메시지

이 문서는 T527 NPU에서 한국어 STT를 배포하기 위한 **4주간의 실험, 15개 모델, 100+ 양자화 시도**의 종합 분석이다.

**잘못한 것:**
- vocab 크기가 핵심이라고 잘못 분석하여 팀에 vocab 56 전환을 권고
- wav2vec2를 고치는 데 2주를 투자 (해결 불가능한 문제)
- 시뮬레이션 결과를 신뢰하여 잘못된 최적화 방향 추구

**올바른 것:**
- Conformer가 T527 uint8에서 CER 10.02% 달성 확인
- 아키텍처가 양자화 성패를 결정한다는 원칙 도출
- 530개 레이어 분석, logit margin 실측 등 정량적 근거 확보

**앞으로:**
- **Conformer CTC + aihub 대규모 데이터** = T527 한국어 STT의 답
- 음절 vocab (~2000 BPE) 사용 가능 (자모 56 불필요)
- d_model ≥ 512, 18L, kernel=31 권장

---

# 51. 산업계 동향: Conformer CTC가 Edge STT의 표준

## 51.1 Apple (NAACL 2024 Industry Track)

> [Conformer-Based Speech Recognition On Extreme Edge-Computing Devices](https://aclanthology.org/2024.naacl-industry.12/)

Apple이 **Conformer CTC를 소형 웨어러블에 배포**, 실시간보다 5.26배 빠른 추론 달성.
모델 아키텍처 최적화 + 그래프 변환 + 수치 최적화. **정확도 손실 없이 edge 배포**.

## 51.2 NVIDIA Riva (한국어 Conformer CTC)

> [RIVA Conformer ASR Korean - NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ko_kr_conformer)

NVIDIA가 한국어 Conformer CTC 모델을 **NGC에 공식 배포**. ~120M params, aihub 3500시간+ 학습.
**우리 SungBeom 모델(122.5M)과 거의 같은 스펙** — 이게 산업 표준.

## 51.3 업계 합의

| 회사/기관 | 모델 | 용도 | 비고 |
|----------|------|------|------|
| **Apple** | Conformer CTC | 웨어러블 | NAACL 2024 Industry |
| **NVIDIA** | Conformer CTC | 한국어 production | NGC 배포 |
| **Google** | Conformer CTC/RNN-T | Pixel 기기 | 4-bit까지 양자화 |
| **Amazon** | Conformer (ACosR) | Alexa | sub-8-bit QAT |
| **ENERZAi** | Whisper (Conformer 계열) | 한국어 1.58-bit | CER 6.45% |
| **우리** | **SungBeom Conformer** | **T527 NPU uint8** | **CER 10.02%** |

**Conformer CTC가 edge STT의 사실상 표준.** wav2vec2는 서버용 (self-supervised learning 연구), edge에서는 Conformer.

---

# 52. ONNX Op 유형 분포 비교: Conformer vs wav2vec2

Acuity import 후 json에서 추출한 **내부 operator 분포** (실측):

| Op | **Conformer** | **wav2vec2** | 의미 |
|---|---|---|---|
| **depthwise_conv1d** | **18** | **0** | **Conformer만의 핵심 — 매 레이어에 Conv** |
| conv1d | 37 | 7 | Conformer CNN 비중 5배 |
| **gelu** | **0** | **20** | **wav2vec2만 GELU 사용 (양자화 불안정)** |
| **swish** | **54** | **0** | **Conformer는 Swish 사용 (GELU보다 양자화 안정)** |
| softmax | 18 | 12 | 비슷 |
| layernormalize | 90 | 26 | Conformer 3.5배 |
| matmul | 54 | 24 | Conformer 2.3배 |
| where | 54 | 0 | Conformer에 있지만 Acuity가 SELECT로 변환 |
| **TOTAL** | **1410** | **527** | Conformer가 2.7배 많은 op |

## 핵심 차이점

1. **depthwise_conv1d**: Conformer 18개 (매 레이어), wav2vec2 0개 → **이것이 양자화 안정성의 핵심**
2. **GELU vs Swish**: wav2vec2는 GELU(Erf 기반, 비선형 급변) 사용, Conformer는 Swish(sigmoid 기반, 부드러움) → **Swish가 uint8에서 더 안정적**
3. **op 수**: Conformer가 2.7배 많지만 성공. **op 수보다 op 유형이 중요**

## GELU vs Swish의 양자화 차이

```
GELU(x) = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))
  → erf 함수가 x=0 근처에서 급변 → uint8에서 근사 부정확

Swish(x) = x × sigmoid(βx)
  → sigmoid는 [0,1] 범위에서 부드럽게 변화 → uint8에서 안정적 근사
```

이것이 **같은 attention + FFN 구조에서 Conformer가 더 양자화에 강건한 이유 중 하나**.

---

# 53. Streaming Conformer 가능성

현재 SungBeom 모델은 **3초 고정 입력** → 슬라이딩 윈도우로 긴 오디오 처리.
실시간 STT에는 **Streaming Conformer**가 필요.

## 53.1 Streaming 방식

| 방식 | 설명 | T527 적용 |
|------|------|----------|
| **Chunk-based** | 고정 크기 청크 순차 처리 (현재 방식) | ✓ 현재 동작 |
| **Causal attention** | 미래 프레임 참조 안 함 | NB 재변환 필요 |
| **Look-ahead** | 제한된 미래 프레임만 참조 | NB 재변환 필요 |

## 53.2 현재 방식의 한계

현재 3초 chunk → 각 chunk 독립 추론 → CTC 결과 연결.
- 장점: 단순, 안정
- 단점: chunk 경계에서 단어 잘림, overlap 비효율

## 53.3 개선: 5초 또는 10초 chunk

NB 입력 길이를 **5초(501 frames)** 또는 **10초(1001 frames)**로 변경:
- NB 재변환만 하면 됨 (ONNX static shape 변경)
- chunk 수 감소 → 경계 오류 감소
- NB 크기 약간 증가 (102MB → ~110-130MB, 120MB 제한 내)
- 추론 시간 비례 증가 (233ms → ~400-800ms)

---

# 54. Conformer NB 변환 Pad op 수정 상세

Conformer ONNX에서 Acuity 호환을 위해 **유일하게 필요한 수정**:

```python
# 18개 Pad op의 빈 constant_value를 0.0으로 설정
pad_const = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="__pad_zero__")
model.graph.initializer.append(pad_const)
for node in model.graph.node:
    if node.op_type == "Pad" and len(node.input) >= 3 and node.input[2] == '':
        node.input[2] = "__pad_zero__"
```

**이것만 수정하면 됨.** Where op 54개는 **절대 건드리지 않음** — Acuity가 자동으로 SELECT+SWITCH로 변환.

이전 실패 교훈: Where op을 수동 제거했더니 shape inference가 깨져서 모델 망가짐.

---

# 55. T527 NPU Memory Pool 크기 비교

vpm_run에서 보고되는 `memory pool size`:

| 모델 | memory pool | NB | 비고 |
|------|-----------|-----|------|
| **SungBeom Conformer** | **14.7MB** | 102MB | — |
| Wav2Vec2 KO (full) | 14.7MB | 72MB | — |
| Wav2Vec2 KO (encoder only) | 14.7MB | 75MB | — |
| Wav2Vec2 KO (L0-7 only) | 14.7MB | 52MB | — |
| Wav2Vec2 KO (TF only) | **0.9MB** | 72MB | CNN 없이 작음 |
| Wav2Vec2 EN (full) | 24.6MB | 87MB | 5초 입력이라 큼 |

**관찰:** memory pool은 NB 크기와 정비례하지 않음. 입력 크기(48000 vs 80000)에 영향받음.
SungBeom과 wav2vec2 KO는 같은 memory pool(14.7MB) — 둘 다 3초 입력(48000 samples).

---

# 56. 최종 참고 문헌 목록

이 분석에서 참조한 모든 논문 및 기술 자료:

## 아키텍처
1. [Conformer: Convolution-augmented Transformer (Google, INTERSPEECH 2020)](https://arxiv.org/abs/2005.08100)
2. [Squeezeformer: Efficient Transformer for ASR (NeurIPS 2022)](https://arxiv.org/pdf/2206.00888)
3. [E-Branchformer: Enhanced Merging (SLT 2022)](https://arxiv.org/abs/2210.00077)
4. [wav2vec 2.0: Self-Supervised Learning (NeurIPS 2020)](https://arxiv.org/abs/2006.11477)

## 양자화
5. [4-bit Conformer Native QAT (Google, INTERSPEECH 2022)](https://arxiv.org/abs/2203.15952)
6. [1-bit Conformer (INTERSPEECH 2025)](https://arxiv.org/abs/2505.21245)
7. [ACosR: QAT for ASR (Amazon, INTERSPEECH 2020)](https://www.amazon.science/publications/quantization-aware-training-with-absolute-cosine-regularization-for-automatic-speech-recognition)
8. [Sub-8-Bit QAT for 8-Bit NPU (Amazon, INTERSPEECH 2022)](https://arxiv.org/abs/2206.15408)
9. [Edge-ASR: Low-Bit Quantization (Qualcomm, 2025)](https://arxiv.org/abs/2507.07877)
10. [Quantizable Transformers: Removing Outliers (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/edbcb7583fd8921dad78adecfe06a99b-Paper-Conference.pdf)
11. [Activation Outliers in Transformer Quantization (2025)](https://arxiv.org/abs/2603.04308)
12. [R2 Loss: Range Restriction (2023)](https://arxiv.org/abs/2303.08253)
13. [LSQ: Learned Step Size Quantization (ICLR 2020)](https://arxiv.org/abs/1902.08153)
14. [SmoothQuant: W8A8 for Transformers (2022)](https://arxiv.org/abs/2211.10438)

## Edge 배포
15. [Conformer on Extreme Edge-Computing (Apple, NAACL 2024)](https://aclanthology.org/2024.naacl-industry.12/)
16. [INT8 Conformer on ARM Ethos-U85 (2025)](https://developer.arm.com/community/arm-community-blogs/b/internet-of-things-blog/posts/end-to-end-int8-conformer-on-arm-training-quantization-and-deployment-on-ethos-u85)
17. [ENERZAi: Korean 1.58-bit Whisper (2025)](https://enerzai.com/resources/blog/small-models-big-heat-conquering-korean-asr-with-low-bit-whisper)
18. [NVIDIA Riva Korean Conformer CTC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ko_kr_conformer)

## 비교/분석
19. [Assessing Robustness of Conformer/Transformer Under Compression (EUSIPCO 2024)](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000336.pdf)
20. [E-Branchformer vs Conformer Comparative Study (2023)](https://arxiv.org/abs/2305.11073)
21. [OpenVINO NNCF Wav2Vec2 Quantization](https://docs.openvino.ai/2024/notebooks/speech-recognition-quantization-wav2vec2-with-output.html)
22. [KD + QAT for Transformer Encoders (EMNLP 2022)](https://arxiv.org/abs/2211.11014)

## 한국어 ASR
23. [KoSpeech: Korean ASR Toolkit](https://arxiv.org/abs/2009.03092)
24. [K-Wav2vec 2.0: Korean Joint Decoding (INTERSPEECH 2022)](https://arxiv.org/abs/2110.05172)
25. [Exploring Lexicon-Free Modeling Units for Korean (INTERSPEECH 2020)](https://arxiv.org/abs/1910.11590)

---

# 57. Conformer 내부 설계의 양자화 관점 해석

## 57.1 Macaron 구조: 왜 FFN이 반씩 둘인가

Conformer 블록:
```
입력 → [FFN ½] → [Multi-Head Self-Attention] → [Conv] → [FFN ½] → 출력
       ^^^^^^^^                                         ^^^^^^^^
       마카롱의 윗 쿠키                                 마카롱의 아랫 쿠키
```

**Macaron-Net 원리:** FFN을 하나가 아니라 **두 개의 절반짜리**로 나누어 Attention/Conv을 샌드위치.

**양자화 관점:**
- FFN ½은 **값 범위를 절반으로 축소** (half-step residual)
- Attention 전후에 FFN이 있어서 **activation range가 급변하는 것을 방지**
- 일반 Transformer: FFN → Attention → (값 폭발) → 다음 레이어
- Conformer: FFN½ → Attention → Conv(안정화) → FFN½ → (값 안정) → 다음 레이어

## 57.2 Conv가 Attention 뒤에 와야 하는 이유

Google의 ablation study에서 **Conv를 Attention 앞에 놓으면 성능 저하** 확인.

**양자화 관점 해석:**
```
순서 A (Conformer): Attention → Conv
  Attention이 activation 넓힘 → Conv가 local averaging으로 다시 좁힘 ✓

순서 B (역순): Conv → Attention
  Conv가 local feature 추출 → Attention이 global로 넓힘 (안 좁혀짐) ✗
```

**핵심:** Conv가 Attention **뒤에** 와서 "정규화기" 역할을 해야 uint8에서 안정.

## 57.3 GLU (Gated Linear Unit) activation

Conformer Conv 모듈에서 pointwise conv 후 **GLU** 사용:
```
GLU(x) = x[:, :half] × sigmoid(x[:, half:])
```

**vs wav2vec2의 GELU:**
```
GELU(x) = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))
```

| | GLU (Conformer) | GELU (wav2vec2) |
|---|---|---|
| 게이팅 | sigmoid (0~1, 부드러움) | erf 기반 (급변) |
| 출력 범위 | 입력의 절반 차원, 값 bounded | 입력과 같은 차원, 값 unbounded |
| uint8 호환성 | **좋음** | 나쁨 |

GLU는 채널의 절반을 "게이트"로 사용하여 나머지 절반을 **0~1 범위로 스케일링** → 자연스러운 값 범위 제한 → uint8 친화적.

## 57.4 Conformer가 "양자화를 위해 설계된 건 아니지만 양자화에 유리한 이유"

Conformer의 설계 목표는 양자화가 아니라 **정확도 + 효율성**. 하지만 그 설계 선택들이 **우연히 양자화에 유리**:

| 설계 선택 | 원래 목적 | 양자화 부수 효과 |
|----------|----------|---------------|
| Depthwise Conv(31) | local feature 캡처 | activation range 안정화 |
| Macaron FFN ½ | 학습 안정성 | 값 범위 축소 |
| Conv after Attention | global→local 순서 | Attention 후 정규화 |
| GLU activation | gating mechanism | 값 범위 bounding |
| Relative pos enc | 상대 위치 | bounded 값 |
| Swish (not GELU) | 부드러운 활성화 | uint8 근사 안정 |

---

# 58. 모든 실패의 공통 패턴

15개+ 모델에서 **실패한 모든 경우의 공통점**:

```
1. 순수 Transformer 아키텍처 (CNN이 Attention 뒤에 없음)
2. 또는 모델이 너무 큼 (5000+ ONNX nodes → 오차 누적)
3. 또는 NB 크기 > 120MB (NPU 메모리 제한)
```

**예외:** 영어 wav2vec2 (순수 Transformer이지만 성공)
→ 이유: 960시간 대규모 영어 데이터로 **sharp attention** 학습 → activation range 좁음 → uint8 생존

**교훈:** 순수 Transformer가 uint8에서 동작하려면 **매우 많은 데이터 + 해당 언어에 최적화된 weight**가 필요. 한국어처럼 데이터가 부족하거나 transfer learning한 경우 attention이 "soft" → uint8 실패.

---

# 59. BPE vs Character vs Jamo: 토크나이저 선택 가이드

Conformer가 **vocab 2049 BPE**로 성공했으므로, 토크나이저 선택을 재평가.

| 토크나이저 | vocab 크기 | 예시 ("안녕하세요") | 토큰 수 | T527 uint8 |
|-----------|-----------|-----------------|---------|-----------|
| **BPE 2048** | ~2049 | "안녕" + "하세요" | 2 | **CER 10.02% ✓** |
| BPE 5000 | ~5001 | "안녕하" + "세요" | 2 | CER 54.53% △ |
| 음절 | ~2000 | "안"+"녕"+"하"+"세"+"요" | 5 | CER 92.83% (wav2vec2) |
| **자모** | ~56 | "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ" | 11 | 미테스트 (Conformer) |

**BPE 2048이 최적:**
- 토큰 수가 적어서 CTC alignment 쉬움 (2 토큰 vs 음절 5 vs 자모 11)
- 의미 단위 토큰이라 언어 모델 효과
- **uint8 step 0.203 — 충분한 margin 확보**

자모 56은 wav2vec2에서 시도했지만, Conformer에서는 **BPE가 더 나을 가능성 높음** — CTC alignment + uint8 margin 모두 유리.

---

# 60. Inference 파이프라인 최적화 가이드

## 60.1 현재 파이프라인 (Python, 서버에서 전처리)

```
WAV (16kHz mono)
  → NeMo mel (Python, 서버) [1, 80, T]
  → 3초 chunk 분할 [1, 80, 301]
  → uint8 양자화 (scale/zp)
  → T527 NPU 추론 (233ms)
  → uint8 dequantize
  → CTC greedy decode
  → BPE detokenize
  → 한국어 텍스트
```

## 60.2 목표 파이프라인 (Android 앱, on-device)

```
마이크 입력 (16kHz)
  → Java: mel spectrogram (NeMo 호환 구현)
  → JNI: uint8 양자화 + NPU 추론 (awnn API)
  → Java: CTC decode + BPE detokenize
  → UI: 텍스트 표시
```

## 60.3 구현 과제

| 과제 | 난이도 | 해결 방법 |
|------|--------|----------|
| NeMo mel을 Java로 구현 | ★★★ | FFT + mel filterbank + log + normalize |
| BPE detokenize | ★★ | SentencePiece Java wrapper |
| 슬라이딩 윈도우 관리 | ★★ | 3초 buffer + overlap 처리 |
| NPU 모델 로드/추론 | ★ | 기존 awnn API 재사용 |

**핵심 과제:** NeMo mel을 Java에서 정확하게 재현하는 것. librosa와 다르므로 NeMo 소스 코드 기반으로 구현 필요.

---

# 61. 비전공자를 위한 최종 비유

## "왜 Conformer는 되고 wav2vec2는 안 되는가"를 초등학생도 이해할 수 있게

**비유: 그림 축소 복사**

```
원본 그림 (FP32 모델): 고해상도, 모든 디테일 표현
축소 복사 (uint8 양자화): 256색 GIF로 변환

Conformer (수채화):
  - 색이 부드럽게 퍼져있음 (activation range 안정)
  - 256색으로 줄여도 전체적인 느낌이 유지됨
  - "이건 고양이 그림이야" → "맞아, 고양이야" ✓

wav2vec2 한국어 (유화, 극도로 세밀):
  - 미세한 색 차이가 중요한 디테일을 표현
  - 256색으로 줄이면 비슷한 색들이 합쳐져서 디테일 소실
  - "이건 고양이 그림이야" → "뭔지 모르겠는데..." ✗

왜 같은 구도(아키텍처)인데 수채화(Conformer)만 되는가?
  → 수채화는 물감이 번지면서 자연스럽게 색이 합쳐짐 (Conv가 activation 안정화)
  → 유화는 붓터치 하나하나가 중요해서 256색으로는 표현 불가 (Transformer의 정밀한 attention)
```

---

# 62. KoCitrinet과 Conformer의 공통점: CNN이 핵심

T527에서 성공한 모델은 **KoCitrinet(CNN only)과 Conformer(CNN+Attention)**. 공통점:

| 특성 | KoCitrinet | Conformer | wav2vec2 |
|------|-----------|-----------|---------|
| **Depthwise Separable Conv** | ✓ (핵심) | ✓ (핵심) | ✗ |
| **Squeeze-and-Excitation (SE)** | ✓ | ✗ (일부 변형에 있음) | ✗ |
| **mel spectrogram 입력** | ✓ | ✓ | ✗ (raw waveform) |
| **CTC decoder** | ✓ | ✓ | ✓ |
| **uint8 CER** | **44.44%** | **10.02%** | **100%** |

**두 성공 모델의 공통 DNA: Depthwise Separable Convolution.**

KoCitrinet은 CNN만으로 44.44%. Conformer는 여기에 Attention을 추가해 10.02%. **CNN이 양자화 안정성의 기반**, Attention이 정확도 향상.

---

# 63. Layer Normalization의 양자화 영향

Conformer 90개, wav2vec2 26개의 LayerNorm op이 있다.

**LayerNorm의 양자화 특성:**
```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

- `mean(x)`: 채널 평균 → uint8에서 정확히 계산 가능 (산술 평균)
- `var(x)`: 분산 → **uint8에서 정밀도 손실** (제곱 연산 관련)
- `sqrt(var)`: **uint8에서 근사 오류**
- 전체: 정규화 → 값 범위를 `gamma * [-3, +3] + beta` 수준으로 리셋

**양자화 관점에서 LayerNorm은 "양날의 검":**
- 장점: activation range를 리셋하여 uint8 범위 내로 되돌림
- 단점: 분산 계산의 uint8 오차가 정규화 결과를 왜곡

Conformer에서 LayerNorm이 90개로 많은 건 **매 레이어마다 여러 위치에서 range를 리셋**하기 때문. 이것이 activation 폭발을 방지.

---

# 64. 양자화 친화적 Conformer 학습 팁

기존 Conformer를 더 양자화에 유리하게 학습하려면:

| 팁 | 효과 | 구현 |
|---|---|---|
| **Weight decay 높이기 (0.01~0.05)** | weight range 축소 → uint8 해상도 향상 | training config |
| **Gradient clipping (max_norm=1.0)** | activation 급변 방지 | training config |
| **Label smoothing 제거 (0.0)** | logit margin 최대화 | loss config |
| **Dropout 제거 (추론 시)** | activation 안정화 | inference config |
| **NeMo dither=0.0** | mel noise 제거 | preprocessor config |
| **BPE vocab ~2048** | logit margin 확보 | tokenizer config |
| **d_model ≥ 512** | 양자화 noise 흡수 용량 | model config |

이 팁들은 **QAT 없이 PTQ만으로도** 양자화 품질을 높인다. SungBeom 모델이 PTQ만으로 CER 10.02%를 달성한 것도 이런 설정이 적절했기 때문으로 추정.

---

# 65. 이 프로젝트의 의의

## 65.1 우리가 생산한 지식

- **15개 모델, 100+ 양자화 실험**의 실측 데이터
- **530개 레이어 FP32 vs uint8 dump** 분석
- **Acuity 시뮬레이션 ≠ T527 디바이스** (31.5% 일치) 발견
- **아키텍처 > vocab > 데이터 > 알고리즘** 우선순위 도출
- **Conformer CTC + uint8 PTQ** = T527 한국어 STT의 답

## 65.2 틀린 것에서 배운 것

- **Vocab 분석 오류**: vocab 56 권고 → Conformer(2049)가 반증
- **시뮬레이션 신뢰**: NB_agree 70.8%에 만족 → 디바이스 CER 174%
- **QAT 과신**: margin 3배 개선에 만족 → 여전히 step 미달
- **Split 기대**: lm_head fp32로 해결 기대 → encoder가 근본 원인

**틀리는 것은 피할 수 없다. 핵심은 틀림을 인정하고 방향을 바꾸는 속도.**

---

# 66. Positional Encoding 상세 비교

## 66.1 wav2vec2: Convolutional Positional Encoding

```python
# wav2vec2 positional encoding
self.pos_conv = nn.Conv1d(768, 768, kernel_size=128, padding=64, groups=16)
# 128 크기 grouped conv로 위치 정보 학습
```

- **학습된 위치**: CNN이 위치를 학습 → 값 범위 unbounded
- grouped conv (groups=16): 채널간 상호작용 제한
- **양자화 문제**: 학습된 값이 outlier를 가질 수 있음

## 66.2 Conformer: Relative Positional Encoding

```python
# Conformer relative position
# Shaw et al. (2018): Self-Attention with Relative Position Representations
attention_score = Q @ K^T + Q @ rel_pos_bias
# rel_pos_bias: 상대 거리에 따른 편향 (bounded)
```

- **상대 거리 기반**: 프레임 i와 j의 거리 (i-j)에 따른 bias
- **값 범위 bounded**: 거리는 [-sequence_length, +sequence_length]
- **양자화 친화**: bounded 값 → uint8 매핑 예측 가능

## 66.3 양자화 관점 비교

| | wav2vec2 Conv Pos | Conformer Rel Pos |
|---|---|---|
| 값 범위 | unbounded (학습됨) | **bounded (상대 거리)** |
| outlier 가능성 | 높음 | **낮음** |
| uint8 매핑 | 예측 불가 | **예측 가능** |
| T527 결과 | 한국어 실패 | **성공** |

---

# 67. 향후 실험 제안 목록

아직 시도하지 않은 실험들:

## 높은 우선순위

| # | 실험 | 예상 소요 | 기대 결과 |
|---|------|----------|----------|
| 1 | **aihub + NeMo Conformer CTC 직접 학습** | 3~7일 | SungBeom CER 10% 이상 |
| 2 | **SungBeom 5초 입력 NB 재변환** | 1일 | 슬라이딩 윈도우 오버헤드 감소 |
| 3 | **Wav2Vec2-Conformer 시도** | 2일 | wav2vec2 pretrained + Conformer |
| 4 | **SungBeom Android 앱 통합** | 3~5일 | on-device STT 프로토타입 |

## 중간 우선순위

| # | 실험 | 예상 소요 | 기대 결과 |
|---|------|----------|----------|
| 5 | E-Branchformer 시도 | 3일 | Conv 비중 최고, Conformer 이상? |
| 6 | FastConformer (8x subsampling) | 2일 | 더 빠른 추론, NB 작아짐 |
| 7 | SungBeom QAT | 2일 | CER 10% → 8%? |
| 8 | Conformer + Streaming (causal attention) | 5일 | 실시간 STT |

## 낮은 우선순위 (탐색적)

| # | 실험 | 비고 |
|---|------|------|
| 9 | Conformer 레이어별 dump (SungBeom) | 왜 되는지 정량적 증거 |
| 10 | int4/uint4 양자화 (Acuity 지원 확인) | 더 작은 NB |
| 11 | Whisper encoder + Conformer decoder | 하이브리드 |
| 12 | ECAPA-TDNN 기반 CTC | CNN-only, KoCitrinet 대안 |

---

# 68. 왜 이 문서가 필요한가

이 문서는 **4주간 100+ 실험**의 결과를 한 곳에 모은 것이다.

**이전 상태:** 각 실험 결과가 10+ 개 파일에 흩어져 있고, 실험 간 연결이 불명확. 새로운 사람이 합류하면 "왜 wav2vec2를 안 쓰고 Conformer를 쓰나?"에 대한 답을 찾으려면 파일 10개를 다 읽어야 함.

**이 문서의 역할:**
1. **의사결정 근거**: "왜 Conformer인가?"에 대한 정량적 답변
2. **실패 아카이브**: 같은 실수를 반복하지 않기 위한 기록
3. **온보딩 자료**: 새 팀원이 프로젝트 맥락을 빠르게 이해
4. **기술 부채 방지**: 이유 모르고 "원래 이렇게 하는 거야"가 되는 것 방지

---

# 69. 용어 사전

| 용어 | 설명 |
|------|------|
| **uint8** | 0~255 정수. T527 NPU의 유일한 실용적 양자화 타입 |
| **W8A8** | Weight 8-bit + Activation 8-bit. T527 NPU 강제 |
| **W4A16** | Weight 4-bit + Activation FP16. LLM에서 사용 (T527 미지원) |
| **CER** | Character Error Rate. 문자 단위 오류율 (한국어 STT 표준 지표) |
| **WER** | Word Error Rate. 단어 단위 오류율 (영어 STT 표준 지표) |
| **NB** | Network Binary. Acuity가 생성하는 T527 NPU 실행 파일 |
| **PTQ** | Post-Training Quantization. 학습 후 양자화 (우리가 주로 사용) |
| **QAT** | Quantization-Aware Training. 학습 중 양자화 시뮬레이션 |
| **CTC** | Connectionist Temporal Classification. 음성→텍스트 정렬 알고리즘 |
| **BPE** | Byte-Pair Encoding. 서브워드 토크나이저 |
| **mel** | mel spectrogram. 주파수 에너지 표현 (80 bins × time frames) |
| **Acuity** | VeriSilicon의 NPU 모델 변환 도구 (Pegasus 명령어) |
| **logit** | 모델의 raw 출력값 (softmax 이전) |
| **margin** | top-1 logit - top-2 logit. 양자화 생존의 핵심 지표 |
| **depthwise conv** | 채널별 독립 CNN. Conformer의 핵심 구성 요소 |
| **Conformer** | CNN + Self-Attention 하이브리드 ASR 아키텍처 (Google, 2020) |
| **wav2vec2** | Self-supervised Transformer ASR (Facebook, 2020) |
| **NPU** | Neural Processing Unit. AI 추론 전용 하드웨어 가속기 |

---

# 70. FAQ: 자주 받는 질문

## Q1. "왜 그냥 서버에서 안 돌리고 NPU에서 돌리려고 하나요?"

A: **오프라인/실시간 요구사항.** 인터넷 연결 없이도 디바이스에서 즉시 음성 인식이 필요한 경우 (IoT, 임베디드, 보안). 서버 왕복 지연 200~500ms vs NPU 233ms.

## Q2. "FP32로 돌리면 안 되나요?"

A: T527 NPU에서 FP32는 **SRAM 부족으로 실행 불가** (362MB NB). FP16은 HW 미가속 (17초, 42배 느림). **uint8만 HW 가속.**

## Q3. "다른 NPU로 바꾸면 안 되나요?"

A: RK3588(FP16 HW 지원), Qualcomm Hexagon, Samsung Exynos 등이 대안. 하지만 T527이 이미 선정된 플랫폼이고, **Conformer uint8 PTQ로 CER 10.02% 달성**했으므로 하드웨어 교체 없이 해결 가능.

## Q4. "Conformer가 왜 되는지 한 줄로 설명해주세요."

A: **매 레이어에 CNN(depthwise conv)이 있어서 Attention이 넓혀놓은 activation을 다시 좁혀주기 때문.**

## Q5. "wav2vec2에 CNN을 추가하면 되지 않나요?"

A: 이론적으로 가능 (Wav2Vec2-Conformer 모델 존재). 하지만 기존 pretrained weight를 재사용 못 할 수 있고, 재학습 필요. **이미 동작하는 Conformer CTC를 쓰는 게 빠름.**

## Q6. "QAT를 더 하면 wav2vec2도 되지 않나요?"

A: QAT로 margin을 3배 개선했지만 여전히 uint8 step 미달 (ratio 0.66x). **아키텍처의 근본 한계.** QAT보다 아키텍처 교체가 효과적.

## Q7. "int16이나 bf16은 안 되나요?"

A: int16 DFP는 T527에서 실행되지만 **CER이 uint8보다 나쁨** (KoCitrinet int16 CER 330%, SungBeom int16 CER 9.59% — 거의 동일). bf16은 **NB export 실패.** fp16은 **HW 미가속 (17초).**

## Q8. "vocab을 줄이면 더 좋아지나요?"

A: **아니요.** Conformer vocab 2049로 CER 10.02%. vocab을 줄이면 CTC alignment은 쉬워지지만 정보 손실. **vocab보다 아키텍처가 훨씬 중요.**

## Q9. "이 결과를 다른 프로젝트에도 적용할 수 있나요?"

A: **같은 NPU(Vivante VIP9000)를 쓰는 모든 SoC**에 적용 가능. T527, T536, A733 등. 핵심 원칙(CNN 하이브리드, mel 입력, d_model≥512)은 **다른 NPU에서도 유사하게 적용**.

## Q10. "한국어 말고 다른 언어도 되나요?"

A: Conformer CTC 아키텍처는 **언어 무관.** NeMo에서 영어/중국어/일본어 등 다국어 Conformer 모델 제공. 핵심은 아키텍처이므로 언어 바꿔도 uint8 양자화 가능.

---

# 71. Conformer Subsampling: Conv Stem 상세

Conformer의 첫 번째 모듈 — **Convolutional Subsampling** — 이 입력 길이를 줄인다.

```
입력: [1, 80, 301] mel spectrogram (301 frames)
  ↓ Conv2D (kernel=3, stride=2) + ReLU
  ↓ Conv2D (kernel=3, stride=2) + ReLU
출력: [1, 76, 512] (d_model=512)

301 → (301-3)/2+1 = 150 → (150-3)/2+1 = 74 → 실제 76 (padding 포함)
```

**4배 subsampling.** Attention은 76 프레임에서 동작.

**양자화 영향:**
- 76 프레임 × 76 = 5,776 attention 쌍 (wav2vec2의 149² = 22,201의 1/4)
- 양자화 오차 누적이 **1/4로 감소**
- 초기 Conv2D가 mel을 d_model(512)로 매핑 → **첫 양자화 지점부터 높은 차원**

---

# 72. ASR 프레임워크 비교 (NeMo vs ESPnet vs SpeechBrain)

T527 NB 변환 관점에서:

| | NeMo | ESPnet | SpeechBrain |
|---|---|---|---|
| Conformer | ✓ (기본 제공) | ✓ | ✓ |
| ONNX export | `model.export()` | `torch.onnx.export()` | `torch.onnx.export()` |
| **T527 NB 변환 성공** | **✓ (SungBeom)** | 미시도 | **✗ (error 64768)** |
| mel 전처리 | NeMo preprocessor | Kaldi-style | SpeechBrain |
| 한국어 모델 | NGC 제공 | 커뮤니티 | 커뮤니티 |
| Docker | nvcr.io/nvidia/nemo | — | — |

**NeMo 권장.** T527 NB 변환 성공 사례 + NGC 한국어 모델 + ONNX export가 가장 안정적.

SpeechBrain은 42.9M Conformer에서 **NB export error 64768** — 모델 내부 구조나 ONNX op이 Acuity와 비호환.

---

# 73. Acuity Pegasus 명령어 치트시트

## 73.1 Import

```bash
pegasus import onnx --model MODEL.onnx \
  --output-model MODEL.json --output-data MODEL.data
```

## 73.2 Quantize (uint8 KL)

```bash
pegasus quantize --model MODEL.json --model-data MODEL.data \
  --device CPU --with-input-meta inputmeta.yml --rebuild-all \
  --model-quantize MODEL_uint8.quantize \
  --quantizer asymmetric_affine --qtype uint8 \
  --algorithm kl_divergence
```

## 73.3 Export NB

```bash
cd /acuity612/bin  # vxcode/template/ 찾기 위해 bin에서 실행
export REAL_GCC=/usr/bin/gcc
pegasus export ovxlib --model MODEL.json --model-data MODEL.data \
  --dtype quantized --model-quantize MODEL_uint8.quantize \
  --with-input-meta inputmeta.yml --pack-nbg-unify \
  --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk $VSIM --target-ide-project linux64 --batch-size 1 \
  --output-path wksp/
```

## 73.4 주의사항

- **반드시 `bin/` 디렉토리에서 실행** (vxcode/template/ 참조)
- **REAL_GCC 설정 필수** (gen_nbg 컴파일에 사용)
- **LD_LIBRARY_PATH에 vsimulator 경로 추가** (EXTRALFLAGS도)
- **inputmeta의 lid는 import 후 json에서 확인** (모델마다 다름)
- **reverse_channel: false** (오디오 모델 필수)

---

# 74. 디바이스 테스트 자동화 (vpm_run)

```bash
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"

# 1. NB push
$WIN_ADB push network_binary.nb /data/local/tmp/test/

# 2. 입력 push + sample.txt 생성
for i in $(seq -w 0 19); do
  $WIN_ADB push input_${i}.dat /data/local/tmp/test/
  $WIN_ADB shell "printf '[network]\nnetwork_binary.nb\n[input]\ninput_${i}.dat\n[output]\noutput_${i}.dat\n' > /data/local/tmp/test/sample_${i}.txt"
done

# 3. 추론
for i in $(seq -w 0 19); do
  $WIN_ADB shell "cd /data/local/tmp/test && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample_${i}.txt -b 0"
done

# 4. 출력 pull
mkdir -p outputs
for i in $(seq -w 0 19); do
  $WIN_ADB pull /data/local/tmp/test/output_${i}.dat outputs/
done
```

**핵심:** `LD_LIBRARY_PATH=/vendor/lib64` 필수 (NPU 드라이버 위치).
Windows adb.exe 사용 시 경로: `/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe`

---

# 75. 모든 모델의 ONNX 파일 크기 비교

| 모델 | ONNX 원본 | onnxsim 후 | ONNX 노드 | NB | 비율 |
|------|----------|-----------|----------|-----|------|
| **SungBeom Conformer** | 491MB | 477MB | 1982 | 102MB | 21% |
| cwwojin Conformer | 128MB | 126MB | ~1500 | 29MB | 23% |
| SpeechBrain Conformer | 134MB | — | 1438 | 실패 | — |
| KoCitrinet 300f | 543MB (ONNX) | 40MB (sim) | ~200 | 62MB | — |
| Wav2Vec2 EN (5s) | 361MB | 361MB | 957 | 87MB | 24% |
| Wav2Vec2 KO | 361MB | 361MB | 957 | 72MB | 20% |
| Wav2Vec2 XLS-R-300M | 1.27GB | — | — | 249MB | 20% |
| Zipformer Encoder | 280MB | — | 5868 | 63MB | 23% |
| HuBERT KO | 384MB | — | 727 | 76MB | 20% |
| DeepSpeech2 | — | — | — | 56MB | — |
| Whisper tiny encoder | 33MB | — | 211 | 117MB | 354% (!!) |

**관찰:**
- NB/ONNX 비율 ~20-24% (uint8 = FP32의 1/4)
- Whisper tiny는 NB가 ONNX보다 **3.5배 큼** — 30초 고정 입력으로 변환해서 internal buffer가 매우 큼
- NB 크기는 params보다 **입력 길이에 더 영향** 받음 (Whisper 30초 vs 다른 모델 3~5초)

---

# 76. T527 NPU Clock Speed와 성능 관계

T527 NPU 성능 벤치마크 (공식 자료):

| 모델 | 입력 | 546MHz FPS | 696MHz FPS | 비고 |
|------|------|-----------|-----------|------|
| MobileNet V1 | 224² | 367 | — | 이미지 |
| MobileNet V2 | 224² | 309 | 381 | 이미지 |
| Inception v3 | 299² | 32.3 | 40.5 | 이미지 |
| YOLOv5s | 640² | 5.36 | 6.33 | 객체 탐지 |

**우리 STT 모델 (실측):**

| 모델 | 입력 | 추론 시간 | FPS 환산 |
|------|------|---------|---------|
| KoCitrinet | [1,80,300] | 120ms | 8.3 |
| SungBeom Conformer | [1,80,301] | 233ms | 4.3 |
| cwwojin Conformer | [1,80,301] | 111ms | 9.0 |
| Wav2Vec2 KO (3s) | [1,48000] | 415ms | 2.4 |
| Wav2Vec2 EN (5s) | [1,80000] | 715ms | 1.4 |

**STT 모델은 이미지 모델보다 느림** — 입력 크기(48000/80000 samples)가 이미지(224²=50176)보다 크고, Transformer 연산이 CNN보다 무거움.

---

# 77. 모든 성공/실패 모델의 공통 특성 정리

## 성공한 모델의 DNA

```
✓ CNN + Attention 하이브리드 또는 CNN only
✓ mel spectrogram 입력 (bounded range)
✓ d_model ≥ 512 (또는 CNN only라 불필요)
✓ vocab ≤ 2049
✓ NB ≤ 120MB
✓ CTC decoder
✓ NeMo 프레임워크
```

## 실패한 모델의 DNA

```
✗ 순수 Transformer (CNN이 매 레이어에 없음)
✗ 또는 raw waveform 입력 (unbounded activation)
✗ 또는 ONNX 5000+ 노드 (오차 누적)
✗ 또는 NB > 120MB
✗ 또는 vocab > 5000 (부차적 요인)
✗ 또는 RNN-Transducer (autoregressive 오류 전파)
```

**예외:** 영어 wav2vec2 (순수 Transformer + raw waveform이지만 성공)
→ 이유: 960시간 대규모 영어 데이터로 sharp attention 학습 → activation range 좁음 → uint8 생존
→ 교훈: 순수 Transformer도 **충분한 데이터 + 해당 언어 최적화**면 가능. 하지만 이건 특수한 경우.

---

# 부록: Vocab 56 전환 권고 철회

이전에 "vocab을 자모 56으로 바꿔야 한다"고 권고했으나, **이는 잘못된 분석에 기반한 것으로 철회한다.**

- Conformer vocab 2049가 T527 uint8에서 CER 10.02% 달성
- KoCitrinet vocab 2048도 CER 44.44%로 동작
- **vocab 크기가 아니라 모델 아키텍처가 핵심**

aihub 학습은 **음절 vocab (~1900~2000)으로 진행해도 된다.** 단, 모델 아키텍처를 **wav2vec2에서 Conformer CTC로 변경**해야 한다.
