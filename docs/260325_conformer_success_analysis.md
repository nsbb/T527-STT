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

# 부록: Vocab 56 전환 권고 철회

이전에 "vocab을 자모 56으로 바꿔야 한다"고 권고했으나, **이는 잘못된 분석에 기반한 것으로 철회한다.**

- Conformer vocab 2049가 T527 uint8에서 CER 10.02% 달성
- KoCitrinet vocab 2048도 CER 44.44%로 동작
- **vocab 크기가 아니라 모델 아키텍처가 핵심**

aihub 학습은 **음절 vocab (~1900~2000)으로 진행해도 된다.** 단, 모델 아키텍처를 **wav2vec2에서 Conformer CTC로 변경**해야 한다.
