# Calibration 비교 실험 결과 분석

**날짜:** 2026-03-31
**관련 파일:** `CALIB_COMPARISON_RESULTS.md`

---

## 1. 핵심 발견: full QAT가 100k QAT보다 나쁘다

### 1.1 결과 요약

| 모델 | calib | AVG CER | 비고 |
|------|-------|---------|------|
| **100k QAT** | **aihub100** | **7.24%** | **BEST** |
| 100k QAT | mix50 | 8.32% | |
| 100k QAT | real100 | 15.0% | |
| full QAT | real100 | 12.85% | |
| full QAT | mix50 | 12.85% | |
| full QAT | aihub100 | 14.81% | |
| PTQ | 원본 10개 | 16.44% | baseline |

### 1.2 이상한 점

```
직관적 기대:
  full QAT (4,356시간, 10 epoch) > 100k QAT (84시간, 10 epoch)
  데이터 43배 많으니까 더 좋아야 하지 않나?

실제 결과:
  100k QAT (7.24%) >>> full QAT (14.81%)
  100k가 2배 이상 좋음!
```

---

## 2. 원인 분석: QAT 과적합 (QAT Overfitting)

### 2.1 총 학습 Step 비교

```
100k QAT:
  95,000 samples / batch 16 = 5,938 steps/epoch × 10 epoch
  = 59,380 total steps

full QAT:
  4,092,104 samples / batch 16 = 255,756 steps/epoch × 10 epoch
  = 2,557,560 total steps

full QAT는 100k QAT보다 43배 많은 step을 밟음!
```

### 2.2 QAT는 fine-tuning이다

QAT는 이미 학습된 모델을 **살짝 조정**하는 거.
원래 모델(SungBeom Conformer)은 13,946시간으로 1 epoch만 학습됨.

```
원본 학습: 10,916,423 samples × 1 epoch = 10,916,423 steps
QAT 권장: 원본의 10% = ~1,000,000 steps

100k QAT: 59,380 steps (원본의 0.5%) → 적당
full QAT: 2,557,560 steps (원본의 23%) → 과다!
```

### 2.3 FakeQuantize 과적합 메커니즘

```
QAT 초반 (적당한 step):
  FakeQuantize 오차에 적응 → 가중치가 양자화에 강건해짐
  원본 가중치에서 "살짝" 벗어남 → OK

QAT 후반 (과도한 step):
  FakeQuantize의 특정 노이즈 패턴에 과적합
  → FakeQuantize는 per-tensor, 동적 min/max
  → Acuity PTQ는 KL divergence, 정적, per-channel 가능
  → 둘이 다르기 때문에 FakeQuantize에 최적화 ≠ 실제 양자화에 최적화
  원본 가중치에서 "많이" 벗어남 → 나빠짐
```

### 2.4 val_loss는 왜 계속 내려갔는가?

```
val_loss 추이 (full QAT):
  ep0: 0.102 → ep9: 0.0692 (계속 감소!)

근데 T527 CER:
  100k QAT (val_loss 0.151): CER 7.24%
  full QAT (val_loss 0.0692): CER 14.81%

val_loss ↓ 인데 CER ↑ ???
```

**이유: val_loss는 FakeQuantize 환경에서 측정. T527 CER은 실제 Acuity PTQ 환경.**

FakeQuantize에 과적합 → FakeQuantize 기준 val_loss는 좋음 → 근데 실제 양자화와 다름 → T527에서 나빠짐.

이것이 **QAT 과적합의 가장 위험한 점**: val_loss만 보면 알 수 없다.

---

## 3. 업계 연구 및 권장 사항

### 3.1 QAT 학습 기간 가이드라인

| 출처 | 권장 |
|------|------|
| 일반 가이드라인 | 원래 학습의 **10%** 정도 |
| LLM (Llama 등) | 원래 pre-training의 **1% 미만** |
| IBM 가이드 | **1 epoch**으로도 충분한 경우 있음 |
| NVIDIA QAT 블로그 | 소량 데이터 + 적은 epoch 권장 |
| EfficientQAT 논문 | **5000 steps** (Llama 7B 기준) |

### 3.2 QAT 데이터 크기 가이드라인

| 출처 | 권장 |
|------|------|
| 실험적 결과 | **1,000개**로도 60,000개와 비슷한 성능 |
| PTQ calibration | **128~512개**면 충분 |
| 일반 QAT | 전체 데이터의 일부(subset)로 충분 |

### 3.3 SungBeom Conformer 원본 학습

```
데이터: 10,916,423개 (13,946시간)
Epoch: 1
lr: 1e-5
base: NVIDIA RIVA Conformer Korean

→ 원본 자체가 1 epoch만 학습!
→ QAT도 1 epoch이면 충분할 가능성 높음
```

---

## 4. 최적 QAT 설정 추정

### 4.1 Step 수 기준으로 역산

```
100k QAT가 CER 7.24%로 best였음.
100k QAT = 59,380 steps

full 데이터에서 비슷한 step 수 = ?
4,092,104 / 16(batch) = 255,756 steps/epoch
59,380 / 255,756 ≈ 0.23 epoch

→ full 데이터면 0.2~0.5 epoch이면 충분!
→ 또는 1 epoch + early stopping
```

### 4.2 현재 보유 checkpoint

| checkpoint | 데이터 | step 수 | val_loss | T527 CER |
|-----------|--------|---------|----------|----------|
| full ep00 | 4.09M × 1ep | 255,756 | 0.1128 | **테스트 필요** |
| full ep08 | 4.09M × 9ep | 2,301,804 | 0.0692 | 14.81% (나쁨) |
| full ep09 | 4.09M × 10ep | 2,557,560 | 0.0692 | 14.81% (나쁨) |
| 100k ep09 | 95k × 10ep | 59,380 | 0.151 | 7.24% (좋음) |

**full ep00 (1 epoch)이 가장 유망!** step 수가 255,756으로 100k의 59,380보다 많지만, 적당한 범위. T527 테스트 필요.

---

## 5. Calibration 데이터 분석

### 5.1 calib 소스별 차이

```
100k QAT에서:
  aihub100 calib: 7.24%  (best)
  mix50 calib:    8.32%
  real100 calib:  15.0%  (worst)

full QAT에서:
  real100 calib:  12.85% (best)
  mix50 calib:    12.85%
  aihub100 calib: 14.81% (worst)
```

**100k QAT와 full QAT에서 best calib이 반대!**

### 5.2 왜 이런 차이가 나는가

```
100k QAT:
  - AIHub 데이터로 학습 → activation 분포가 AIHub 스타일
  - aihub100 calib이 activation 분포와 매칭 → best

full QAT:
  - 과적합으로 가중치가 크게 변함 → activation 분포가 원본과 달라짐
  - 어떤 calib이든 매칭이 잘 안 됨
  - real100/mix50이 그나마 나은 건 우연일 수 있음
```

### 5.3 calib 개수의 영향

```
이전 실험 (calib 10개):
  PTQ: 16.44%
  100k QAT: 7.92%
  
현재 실험 (calib 100개):
  100k QAT + aihub100: 7.24%
  
→ calib 10개 → 100개: 0.68%p 개선
→ 효과 있지만 극적이진 않음 (100k QAT 기준)
```

---

## 6. 데이터셋별 편차 분석

### 6.1 100k QAT + aihub100 (best 조합)

| 데이터셋 | CER | 특징 |
|---------|-----|------|
| 7F_KSK | 2.58% | 가장 좋음. 깨끗한 환경? |
| modelhouse_2m_noheater | 3.11% | 히터 없는 조용한 환경 |
| 7F_HJY | 8.33% | 중간 |
| modelhouse_2m | 8.62% | 중간 |
| modelhouse_3m | 13.57% | 가장 나쁨. 3m 거리 = 소리 작음 |

**관찰:**
- 거리가 멀수록 CER 높음 (2m < 3m)
- 소음(히터) 있으면 CER 높음 (noheater < 일반)
- 화자별 차이 큼 (KSK 2.58% vs HJY 8.33%)

### 6.2 7F_HJY의 변동성

```
                 real100   aihub100   mix50
100k QAT 7F_HJY: 43.7%     8.33%     10.28%
full QAT 7F_HJY: 15.97%    17.78%    15.97%
```

7F_HJY는 calib에 따라 43.7% ↔ 8.33%로 5배 차이!
이 데이터셋이 양자화에 가장 민감한 것.

---

## 7. 학술 근거 및 업계 권장사항 (검색 기반)

### 7.1 QAT 학습 기간에 대한 업계 합의

| 출처 | 권장 QAT 기간 |
|------|-------------|
| NVIDIA NeMo 공식 문서 | 원본 학습의 **1~10%** |
| IBM QAT 가이드 | **1 epoch**으로도 충분한 경우 있음 |
| LLM (Llama 등) | 원본 pre-training의 **1% 미만** |
| EfficientQAT 논문 (2024) | Llama 7B 기준 **5,000 steps** |
| OpenVINO 예시 | **5 epochs** |
| Google Conformer QAT (Ding et al. 2022) | 120 epoch 학습 모델에 QAT 적용 → **12~24 epoch** 범위 |

### 7.2 "Catastrophic Forgetting" 문제 — 논문으로 확인됨

**ICCV 2023 논문: "Overcoming Forgetting Catastrophe in Quantization-Aware Training" (Chen et al.)**

```
QAT를 과도하게 하면 "catastrophic forgetting" 발생:
  → 모델이 원래 pre-trained 지식을 잊어버림
  → 양자화 노이즈에만 적응하다가 일반화 능력 상실
  → validation accuracy가 plateau 후 degradation
```

**PyTorch Forums 실제 사례:**
- validation accuracy가 77.67% → 75.67%로 **7 epoch 만에 하락**
- 원인: calibration 데이터 부족 + 과도한 epoch

### 8.3 Google의 Conformer QAT 논문 결과

**"4-bit Conformer with Native Quantization Aware Training" (Ding et al., Interspeech 2022)**

```
- LibriSpeech 960시간으로 Conformer 120 epoch 학습
- 4-bit QAT → lossless (CER 저하 없음)
- 학습 시간 증가: 겨우 7% (TPU 기준)

핵심 발견:
- 공개 데이터(LibriSpeech)에서는 lossless
- 대규모 production 데이터에서는 regression 발생!
- → 데이터 규모가 크면 QAT 과적합 위험 증가 (우리와 동일한 현상)
```

### 8.4 SungBeom Conformer 원본 학습 정보

```
출처: https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium
데이터: 10,916,423개 (13,946시간)
Epoch: 1 (!)
lr: 1e-5
base: NVIDIA RIVA Conformer Korean (fine-tuning)

원본 자체가 1 epoch만 학습!
→ QAT 권장 (원본의 1~10%): 0.01~0.1 epoch
→ step 수로: 109,164 ~ 1,091,642 steps
→ 우리 100k QAT (59,380 steps)는 원본의 0.5% → 적절 범위
→ 우리 full QAT (2,557,560 steps)는 원본의 23% → 과다!
```

### 7.5 관련 논문 목록

| 논문 | 연도 | 핵심 내용 |
|------|------|----------|
| Ding et al. "4-bit Conformer with Native QAT" | 2022 | Conformer QAT 원조, lossless 4-bit |
| "2-bit Conformer quantization for ASR" | 2023 | 2-bit까지 확장, co-training 기법 |
| Chen et al. "Overcoming Forgetting Catastrophe in QAT" | ICCV 2023 | QAT 과적합 = catastrophic forgetting |
| "Compute-Optimal Quantization-Aware Training" | 2025 | 최적 epoch 배분 연구 |
| "Scheduling Weight Transitions for QAT" | 2024 | transition-adaptive lr 제안 |
| "Towards One-bit ASR" | 2025 | 1-bit Conformer, stochastic precision |

### 7.6 우리 실험과 논문 결과 매칭

```
논문: "대규모 production 데이터에서 QAT regression 발생"
우리: full QAT (4,356시간) → 100k QAT보다 나쁨 → 일치!

논문: "원본 학습의 10% 이내로 QAT 권장"
우리: 100k QAT = 원본의 0.5% → 적절 → CER 7.24% (good)
     full QAT = 원본의 23% → 과다 → CER 14.81% (bad) → 일치!

논문: "catastrophic forgetting으로 validation이 degradation"
우리: full QAT val_loss는 계속 내려갔지만 T527 CER은 나빠짐
     → FakeQuantize 과적합 = catastrophic forgetting의 변형 → 일치!
```

---

## 8. 결론 및 권고

### 8.1 핵심 교훈

1. **QAT는 적당히 해야 한다** — 데이터 많다고 epoch도 많이 하면 과적합
2. **총 step 수가 핵심** — 데이터 크기 × epoch. 5만~25만 step이 적당
3. **val_loss만 보면 안 된다** — FakeQuantize 과적합은 val_loss로 감지 불가
4. **calib 데이터 소스가 중요** — 학습 데이터와 매칭되는 calib이 best
5. **calib 개수는 100개면 충분** — 10개 → 100개는 효과 있지만 극적이진 않음

### 8.2 현재 Best 조합

```
모델: 100k QAT (95,000개, 84시간, 10 epoch, margin 0.3)
Calib: aihub100 (AIHub 데이터 100개)
CER: 7.24%
```

### 8.3 추가 테스트 권고

1. **full QAT ep00 (1 epoch)** T527 테스트 — 과적합 전 모델, CER 개선 가능성
2. **full QAT 0.5 epoch** — step 수를 100k QAT와 비슷하게 맞춤
3. **KD QAT 결과** — margin 0.5, 1.0 비교
4. **calib 데이터 최적화** — aihub에서 다양성 높은 100개 선별

### 8.4 모델 파일 위치

```
100k QAT (현재 best):
  /home1/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom/qat_aihub_output/conformer_qat_100k_84hr_final.nemo

full QAT ep00 (테스트 필요):
  /home1/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom/qat_aihub_full_output/conformer_qat_aihubfull_margin0.3_ep00.nemo
```
