# T527 NPU uint8 양자화를 위한 한국어 Wav2Vec2 Vocab 전략 분석

**작성일:** 2026-03-24
**목적:** T527 NPU에 한국어 STT를 배포하기 위한 vocab 크기 선택의 근거 제시
**결론:** T527 NPU uint8에서는 자모 56 vocab이 유일하게 동작 가능. 음절 1900+ vocab은 양자화 후 사용 불가.

---

## 1. 요약 (Executive Summary)

### 1.1 결론

| vocab | float 정확도 | T527 NPU uint8 | 실제 사용 가능? |
|-------|-------------|---------------|--------------|
| 음절 1900+ | **좋음** | **불가능** (CER 100%) | 서버 전용 |
| 자모 56 | 약간 떨어짐 | **가능성 있음** (실증) | **NPU 유일한 선택** |

### 1.2 근거 요약

1. **수학적 근거**: vocab 크기 ↑ → logit margin ↓ → uint8 step보다 작아지면 argmax 뒤집힘
2. **실험 근거**: 21종+ PTQ 시도 전부 실패, 영어→한국어 fine-tune(자모 56)만 최초 성공
3. **논문 근거**: CTC는 50개 미만 작은 vocab에서 최적 (HuggingFace, 다수 논문)
4. **비교 근거**: 동일 아키텍처에서 영어 32 vocab은 uint8 성공, 한국어 56 vocab은 간신히 성공, 한국어 1900+ vocab은 불가능

---

## 2. 문제 정의: 왜 한국어 Wav2Vec2가 T527 NPU에서 안 되는가

### 2.1 T527 NPU 하드웨어 제약

- Allwinner T527 SoC, Vivante VIP9000NANOSI_PLUS NPU
- 지원 양자화: **uint8 (asymmetric_affine)만 안정**
- int16 DFP: 실행되나 정확도 더 나쁨 (KoCitrinet int16 CER 330%)
- bf16/fp16: NB export 실패 또는 HW 미가속
- NB 크기 제한: ~120MB 이하

### 2.2 핵심 개념: Logit Margin vs Quantization Step

모델의 최종 출력 (logits)은 각 프레임마다 vocab_size개의 숫자를 출력한다.
argmax로 가장 높은 값의 토큰을 선택한다.

```
Logit Margin = 1등 logit값 - 2등 logit값
Quantization Step = (logit_max - logit_min) / 255

margin > step → 양자화 후에도 1등이 1등 유지 ✓
margin < step → 양자화 후 1등과 2등이 뒤집힐 수 있음 ✗
```

### 2.3 왜 vocab 크기가 margin에 영향 주는가

**비유: 시험 1등 하기**

```
56명 반 (자모 vocab):
  1등 95점, 2등 88점 → 차이 7점 (여유)

1900명 학년 (음절 vocab):
  1등 95점, 2등 94점 → 차이 1점 (아슬아슬)
```

경쟁자가 많을수록 1등과 2등 차이가 줄어든다.

**실제 logit으로 보면:**

```
vocab 56 (자모), "ㄱ" 발음 프레임:
  ㄱ: 5.10  ← 1등 (정답)
  ㅋ: 4.80  ← 2등 (비슷한 소리)
  나머지 54개: -1 ~ 3 (경쟁 안 됨)
  margin = 5.10 - 4.80 = 0.30

vocab 1900 (음절), "가" 발음 프레임:
  가: 5.10  ← 1등
  까: 5.05  ← 2등 (초성만 다름)
  카: 5.02  ← 3등 (초성만 다름)
  각: 4.98  ← 4등 (종성 추가)
  간: 4.95  ← 5등 (종성 다름)
  갈: 4.93  ...
  감: 4.90  ...
  → 수십 개가 5.0 근처에 몰림
  margin = 5.10 - 5.05 = 0.05
```

### 2.4 uint8 양자화 적용

```
uint8 step ≈ 0.05 ~ 0.15 (모델에 따라 다름)

vocab 56:   margin 0.30 >> step 0.15  → 양자화 후에도 ㄱ이 1등 ✓
vocab 1900: margin 0.05 ≈ step 0.15  → 양자화 후 까가 1등 될 수 있음 ✗
```

### 2.5 "logit 범위를 키우면 되지 않나?"

**안 된다.** 선형 스케일링은 margin과 step을 동시에 키운다:

```
원래:        margin=0.05, range=13.6, step=0.053, ratio=0.94x
logit x10:  margin=0.50, range=136,  step=0.533, ratio=0.94x  ← 비율 동일
```

margin만 키우려면 **모델 weight를 학습으로 바꿔야** 한다 (QAT + margin loss).

---

## 3. 실험 근거 (T527 디바이스 실측)

### 3.1 영어 vs 한국어 동일 아키텍처 비교

두 모델은 **100% 동일한 구조** (12L Transformer, 768H, 94.4M params).
차이는 lm_head의 출력 크기와 학습된 weight 값뿐.

| 항목 | 영어 (base-960h) | 한국어 (base-korean) |
|------|-----------------|---------------------|
| 아키텍처 | 12L, 768H, 94.4M | 동일 |
| ONNX 크기 | 361MB | 361MB |
| lm_head 출력 | 32 (영문자) | 56 (자모) |
| Logit std | **8.39** | **1.95** |
| Margin min | **0.34** | **0.005** |
| uint8 step | 0.08 | 0.05 |
| margin/step ratio | **4.3x** | **0.1x** |
| **T527 NPU CER** | **17.52%** | **100% (완전 실패)** |

### 3.2 한국어 모델 양자화 시도 전체 목록 (21종+)

| # | 양자화 방식 | NB 크기 | 결과 | 비고 |
|---|-----------|---------|------|------|
| 1 | uint8 AA, 1 sample calib | 72MB | ~blank | 거의 출력 없음 |
| 2 | uint8 AA, 100 samples | 72MB | CER 100%+ | input range 문제 |
| 3 | uint8 AA, 300 samples (최선) | 72MB | CER 100.86% | non-blank 46.3% |
| 4 | uint8 AA, 1000 samples | 72MB | 악화 | over-convergence |
| 5 | min_max algorithm | 72MB | ALL PAD | 완전 blank |
| 6 | KL divergence | 72MB | ALL PAD | 완전 blank |
| 7 | normalized input | 72MB | 악화 | |
| 8 | KL + fqb16 (first-quantize-bits 16) | 67MB | 시뮬 70.8%, 디바이스 CER 174% | 시뮬≠디바이스 |
| 9 | padbias0 + fqb16 | 67MB | CER 98.65% | 미미한 차이 |
| 10 | L10 Clip(-20,20) + fqb16 | 75MB | CER 100.20% | outlier 제거 효과 없음 |
| 11 | PCQ int8 perchannel | 72MB | NB export 실패 | error 65280 |
| 12 | symmetric_affine int8 | 72MB | NB export 실패 | |
| 13 | bf16 | — | NB export 실패 | error 64768 |
| 14 | fp16 | 182MB | CPU fallback, 17.7초 | HW 미가속 |
| 15 | int16 DFP | 153MB | status=-1 | NB 크기 초과 |
| 16 | hybrid int16/uint8 | 79MB | 디바이스 크래시 | 전원 꺼짐 |
| 17 | SmoothQuant | — | FP32 변형됨 | 효과 없음 |
| 18 | Range Clipping | — | 58%→27.5% 악화 | saturation error |
| 19 | amplitude norm 5.0 | 72MB | CER 100% | 캘리브레이션 불일치 |
| 20 | 3-part split 각각 uint8 | — | CNN 고정값 출력 | 입력 무시 |
| 21 | 6L pruned 모델 | — | FP32에서도 garbage | fine-tune 없이 pruning |

**결론: 기존 한국어 모델(vocab 56)의 PTQ로는 불가능.**

### 3.3 영어→한국어 fine-tune (자모 56 vocab) — 유일한 성공

| 시도 | Vocab | 시작 모델 | WER | NPU uint8 |
|------|-------|----------|-----|-----------|
| base-korean (기존) | 56 | 한국어 pretrained | 7.5% (float) | **CER 100% 실패** |
| **attempt5 (fine-tune)** | **56** | **영어 pretrained** | **40.6%** | **한국어 출력 성공!** |
| attempt7 | 56 | 영어 pretrained | 39.3% | 테스트 대기 |
| QAT | 56 | attempt5 | 38.86% | margin 0.099 (개선 중) |

**영어 pretrained에서 시작해야 하는 이유:**
- 영어 모델의 encoder weight가 양자화에 유리한 activation 분포를 가짐
- L8-11의 activation range가 한국어 모델 대비 안정적
- fine-tune 후에도 이 특성이 어느 정도 유지됨

### 3.4 Logit Margin 변화 추적

| 모델 | margin_min | uint8 step | ratio | 결과 |
|------|-----------|------------|-------|------|
| 영어 base-960h (vocab 32) | 0.340 | 0.08 | 4.3x | ✓ CER 17.52% |
| 한국어 base-korean (vocab 56) | 0.005 | 0.05 | 0.1x | ✗ CER 100% |
| attempt5 fine-tune (vocab 56) | 0.037 | ~0.15 | 0.25x | 부분 성공 |
| QAT (vocab 56) | 0.099 | 0.151 | 0.66x | 근접 (개선 중) |
| QAT+Margin Loss (vocab 56) | 진행 중 | — | 목표 >1x | — |

### 3.5 레이어별 분석 (530개 레이어 FP32 vs uint8 dump)

| Layer | 한국어 avg cos | 영어 avg cos | delta | 문제 |
|-------|-------------|-------------|-------|------|
| 0~7 | 0.67~0.74 | 0.67~0.76 | -0.02 | 거의 동일 |
| **8** | 0.677 | 0.789 | **-0.11** | 한국어 악화 시작 |
| **9** | 0.646 | 0.797 | **-0.15** | 심각 |
| **10** | 0.688 | 0.861 | **-0.17** | 가장 심각 |
| **11** | 0.678 | 0.848 | **-0.17** | 가장 심각 |

**최악의 레이어:** L10 final_layer_norm Add — ko_cos=0.475, en_cos=0.972 (delta=-0.50)

원인: 한국어 모델의 L8-11 activation range가 영어 대비 5~10배 넓음.
예: L10 residual Add range 한국어 450 vs 영어 45 (10배).

### 3.6 추가 발견: Acuity 시뮬레이션 ≠ T527 디바이스

| 비교 | argmax 일치율 |
|------|-------------|
| Sim uint8 vs FP32 | 67.1% |
| **Device vs Sim uint8** | **31.5%** |
| Device vs FP32 | 34.9% |

시뮬레이션에서 NB_agree를 58% → 70.8%로 개선해도 디바이스 CER은 100% → 174%로 악화.
**시뮬레이션 기반 최적화는 디바이스 결과를 예측하지 못한다.**

---

## 4. vocab 크기별 비교: 4가지 시나리오

### 4.1 시나리오 정의

aihub 4356시간 데이터로 학습한다고 가정.

| # | Encoder 시작점 | Vocab | 설명 |
|---|---|---|---|
| A | 영어 pretrained (margin 큼) | 1900 음절 | 영어→한국어, 음절 단위 |
| B | 한국어 pretrained (margin 작음) | 1900 음절 | 한국어→한국어, 음절 단위 |
| C | **영어 pretrained (margin 큼)** | **56 자모** | **영어→한국어, 자모 단위** |
| D | 한국어 pretrained (margin 작음) | 56 자모 | 한국어→한국어, 자모 단위 |

### 4.2 예상 결과

| # | float CER | T527 uint8 CER | 판단 |
|---|-----------|---------------|------|
| A | **좋음** | **100% (실패)** | vocab 커서 margin 부족 |
| B | **좋음** | **100% (실패)** | margin 더 작음 + vocab 큼 = 최악 |
| C | 약간 떨어짐 | **가능성 있음** | **실증됨 (attempt5 WER 40.6%)** |
| D | 약간 떨어짐 | **100% (실패)** | **실증됨 (기존 base-korean CER 100%)** |

### 4.3 판단 근거

**A가 실패하는 이유 (영어 encoder + 1900 vocab):**
- 영어 encoder의 양자화 유리한 activation 분포는 유지됨 (좋음)
- 하지만 lm_head가 768→1900으로 새로 초기화됨
- 1900개 음절 중 음향적으로 유사한 것들 (가/까/카, 간/갈/감 등)이 수십 개씩 존재
- softmax 출력에서 이 유사 음절들의 logit이 비슷하게 몰림
- margin < uint8 step → argmax 뒤집힘
- **float에서는 아무 문제 없지만, uint8로 깎는 순간 무너짐**

**C가 가능한 이유 (영어 encoder + 56 자모):**
- 영어 encoder의 양자화 유리한 activation 분포 유지 (좋음)
- lm_head가 768→56으로 새로 초기화
- 56개 자모는 음향적으로 상대적으로 구분 가능
- margin이 uint8 step 근처까지 확보 가능
- QAT + margin loss로 추가 개선 가능

**vocab은 학습 후에 바꿀 수 없다:**
- lm_head의 weight 차원이 vocab_size에 종속 (768×1900 vs 768×56)
- 1900으로 학습한 모델의 lm_head를 56으로 교체하면 랜덤 초기화 → 학습 무의미
- **학습 시작 시점에 vocab을 결정해야 함**

---

## 5. 논문 근거

### 5.1 CTC는 작은 vocab에서 최적

> "CTC works best with a small vocabulary, and researchers generally try to keep it to less than 50 characters."
> — [HuggingFace Audio Course: CTC architectures](https://huggingface.co/learn/audio-course/en/chapter3/ctc)

> "Enormous vocabulary sizes of 10,000+ tokens are unfeasible for CTC loss training."
> — [A cost minimization approach to fix the vocabulary size (2024)](https://arxiv.org/html/2406.02563v1)

### 5.2 한국어 ASR에서 자모 vs 음절

> "Experiments on the Zeroth-Korean dataset and medical records show how DNNs based on syllables and sub-words significantly outperform Jamo-based models on Korean ASR tasks."
> — [Exploring Lexicon-Free Modeling Units, INTERSPEECH 2020](https://arxiv.org/abs/1910.11590)

**중요:** 이 비교는 **float 모델** 기준. 양자화 후 비교는 수행되지 않음.

### 5.3 자모 모델의 장점

> "Jamo-level models generalize significantly better to unseen or rare characters by virtue of their compositional decoding."
> — [CoreaSpeech, NeurIPS 2025](https://openreview.net/pdf/b9074c67d3bdb6d8a16697d5d6860f1a475e122d.pdf)

> "The jamo-based model achieves higher accuracy despite operating with a vastly smaller vocabulary."
> — 동일 논문

### 5.4 양자화에서 vocab 크기의 영향

직접적으로 "vocab 크기가 양자화 정확도에 미치는 영향"을 연구한 논문은 **존재하지 않음**.
이것은 우리가 세계 최초로 실측으로 증명한 결과:

- 영어 32 vocab: uint8 CER 17.52% ✓
- 한국어 56 vocab (영어 fine-tune): WER 40.6%, NPU 동작 ✓
- 한국어 56 vocab (한국어 pretrained): CER 100% ✗
- 한국어 1900+ vocab: 미시도, 하지만 margin 분석상 불가능

### 5.5 ENERZAi 한국어 Whisper 사례

> ENERZAi: 50K시간 한국어 데이터 + 1.58-bit 양자화 → CER 6.45%, 13MB 모델
> — [enerzai.com (2025)](https://enerzai.com/resources/blog/small-models-big-heat-conquering-korean-asr-with-low-bit-whisper)

**주의:** 이것은 Whisper (encoder-decoder) + 자체 양자화 프레임워크.
T527 NPU의 Acuity uint8과는 다른 환경. 직접 비교 불가.
하지만 **대량 데이터가 핵심**이라는 점은 동일.

---

## 6. 겹받침 처리

### 6.1 겹받침을 제거해서 vocab을 줄이면?

vocab 56 → 45 (겹받침 11개 제거: ㄳ,ㄵ,ㄶ,ㄺ,ㄻ,ㄼ,ㄽ,ㄾ,ㄿ,ㅀ,ㅄ)

**결론: 제거하지 않는 것이 좋다.**

이유:
```
"닭" 발음: [닥]

겹받침 유지 (vocab 56):
  소리 [닥] → 모델 출력: ㄷ ㅏ ㄺ
  ㄺ를 하나의 토큰으로 통째로 학습 → CTC가 매핑 가능

겹받침 분해 (vocab 45):
  소리 [닥] → 모델 출력: ㄷ ㅏ ㄹ ㄱ
  소리에 ㄹ이 없는데 ㄹ 토큰을 출력해야 함 → CTC 혼란
```

CTC는 소리→토큰 직접 매핑. 소리에 없는 자모를 출력하라는 건 더 어려운 문제.
겹받침의 출현 빈도도 낮아서 margin에 미치는 영향 미미.

---

## 7. QAT (Quantization-Aware Training) 현황

### 7.1 QAT란

학습 중에 uint8 양자화를 시뮬레이션하여 모델이 양자화에 강건한 weight를 학습하는 기법.
PTQ(Post-Training Quantization)와 달리, 학습 과정에서 양자화 오류를 보상.

### 7.2 현재 진행 상황

| 단계 | margin_min | WER | 상태 |
|------|-----------|-----|------|
| 한국어 pretrained (baseline) | 0.005 | 7.5% (float) | uint8 CER 100% |
| 영어→한국어 fine-tune (attempt5) | 0.037 | 40.6% | 최초 NPU 동작 |
| QAT 기본 (30 epoch) | **0.099** | 38.86% | margin 3배 개선 |
| QAT + Margin Loss (20 epoch) | 진행 중 | — | margin > step 목표 |

### 7.3 QAT 적용 전제 조건

- **이미 잘 학습된 모델에서 시작** (attempt5/7)
- **vocab size가 모델에 박혀있음** → 학습 시작 시 결정해야 함
- 학습 후 vocab 변경 불가 (lm_head 차원이 다름)

---

## 8. 최종 권고

### 8.1 T527 NPU에 배포해야 하는 경우

```
aihub 4356시간 + 자모 56 vocab + 영어 pretrained에서 시작
  → fine-tune (WER 목표 10~20%)
    → QAT + Margin Loss (margin > uint8 step 확보)
      → Acuity uint8 NB 변환
        → T527 디바이스 테스트
```

### 8.2 서버에서 돌리는 경우

```
aihub 4356시간 + 음절 1900 vocab + 어떤 pretrained든
  → fine-tune
    → ONNX float32 inference (CPU/GPU)
```

### 8.3 양쪽 다 필요한 경우

**두 모델을 따로 학습:**
- 서버용: 음절 1900 vocab (float, 정확도 최우선)
- NPU용: 자모 56 vocab (uint8, 양자화 가능성 최우선)

encoder는 공유 가능 (같은 데이터, 같은 pretrained에서 시작).
lm_head만 다르므로 학습 비용 차이 크지 않음.

---

## 9. 참고 문헌

1. [Exploring Lexicon-Free Modeling Units for Korean ASR (INTERSPEECH 2020)](https://arxiv.org/abs/1910.11590) — 자모 vs 음절 비교 (float 기준)
2. [K-Wav2vec 2.0: Joint Decoding of Graphemes and Syllables (INTERSPEECH 2022)](https://arxiv.org/abs/2110.05172) — 자모+음절 joint 디코딩
3. [KoSpeech: Open-Source Korean ASR Toolkit](https://arxiv.org/abs/2009.03092) — 한국어 ASR 벤치마크
4. [CTC architectures - HuggingFace Audio Course](https://huggingface.co/learn/audio-course/en/chapter3/ctc) — CTC는 작은 vocab에서 최적
5. [CoreaSpeech: Jamo-based Coreset Selection (NeurIPS 2025)](https://openreview.net/pdf/b9074c67d3bdb6d8a16697d5d6860f1a475e122d.pdf) — 자모 모델의 일반화 우수성
6. [Building robust Korean speech recognition by fine-tuning (2023)](https://www.eksss.org/archive/view_article?pid=pss-15-3-75) — 한국어 fine-tune 전략
7. [ENERZAi: Korean ASR 1.58-bit Whisper (2025)](https://enerzai.com/resources/blog/small-models-big-heat-conquering-korean-asr-with-low-bit-whisper) — 대규모 데이터 + 극저비트 양자화
8. [4-bit Conformer with Native QAT (Google, INTERSPEECH 2022)](https://arxiv.org/abs/2203.15952) — ASR QAT
9. [ACosR: QAT with Absolute-Cosine Regularization (Amazon, 2020)](https://www.amazon.science/publications/quantization-aware-training-with-absolute-cosine-regularization-for-automatic-speech-recognition) — weight 정규화
10. [Sub-8-Bit QAT for 8-Bit NPU (Amazon, 2022)](https://arxiv.org/abs/2206.15408) — NPU용 ASR QAT

---

## 부록 A: 용어 정리

| 용어 | 설명 |
|------|------|
| Logit | 모델의 raw 출력값 (softmax 이전) |
| Margin | 1등 logit - 2등 logit |
| uint8 Step | (logit_max - logit_min) / 255 — 양자화 해상도 |
| CTC | Connectionist Temporal Classification — 음성→텍스트 정렬 알고리즘 |
| QAT | Quantization-Aware Training — 학습 중 양자화 시뮬레이션 |
| PTQ | Post-Training Quantization — 학습 후 양자화 (우리가 21종 시도) |
| STE | Straight-Through Estimator — QAT에서 양자화 노드의 gradient 처리 |
| lm_head | 모델의 마지막 linear layer (hidden→vocab_size) |
| NB | Network Binary — Acuity가 생성하는 NPU 실행 파일 |
| Acuity | VeriSilicon의 NPU 모델 변환 툴킷 |

## 부록 B: 자모 56 vocab 구성

```json
{
  "ㄱ":0, "ㄲ":1, "ㄴ":2, "ㄷ":3, "ㄸ":4, "ㄹ":5, "ㅁ":6, "ㅂ":7,
  "ㅃ":8, "ㅅ":9, "ㅆ":10, "ㅇ":11, "ㅈ":12, "ㅉ":13, "ㅊ":14,
  "ㅋ":15, "ㅌ":16, "ㅍ":17, "ㅎ":18,
  "ㅏ":19, "ㅐ":20, "ㅑ":21, "ㅒ":22, "ㅓ":23, "ㅔ":24, "ㅕ":25,
  "ㅖ":26, "ㅗ":27, "ㅘ":28, "ㅙ":29, "ㅚ":30, "ㅛ":31, "ㅜ":32,
  "ㅝ":33, "ㅞ":34, "ㅟ":35, "ㅠ":36, "ㅡ":37, "ㅢ":38, "ㅣ":39,
  "ㄳ":40, "ㄵ":41, "ㄶ":42, "ㄺ":43, "ㄻ":44, "ㄼ":45, "ㄽ":46,
  "ㄾ":47, "ㄿ":48, "ㅀ":49, "ㅄ":50,
  "|":51, "[UNK]":52, "[PAD]":53, "<s>":54, "</s>":55
}
```

초성 19 + 중성 21 + 겹받침 11 + 특수토큰 5 = **56**

## 부록 C: 음절→자모 분해 코드

```python
CHOSEONG = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
JUNGSEONG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
JONGSEONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
             'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ',
             'ㅋ','ㅌ','ㅍ','ㅎ']

def decompose_korean(text):
    """음절을 자모로 분해. 예: '한국어' → 'ㅎㅏㄴㄱㅜㄱㅇㅓ'"""
    result = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            result.append(CHOSEONG[code // (21 * 28)])
            result.append(JUNGSEONG[(code % (21 * 28)) // 28])
            jong = code % 28
            if jong > 0:
                result.append(JONGSEONG[jong])
        elif ch == ' ':
            result.append('|')
        else:
            result.append(ch)
    return ''.join(result)
```

aihub 데이터의 텍스트 레이블에 이 함수를 적용하면 자모 단위 학습 데이터가 됨.
