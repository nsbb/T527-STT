# SLT 2026 논문 제출 계획

**학회:** IEEE SLT 2026 (Spoken Language Technology Workshop)
**장소:** 팔레르모, 시칠리아, 이탈리아
**마감:** 2026.06.17 (Regular Paper, 6p + 참고문헌 2p)
**통보:** 2026.09.01
**학회:** 2026.12.13-16
**수락률:** ~42% (ICASSP 45%보다 빡셈)

---

## 현실 진단

### "X에 올렸다"만으로는 논문 안 됨

팀원 말이 맞음. 검색 결과 **"특정 디바이스에 최초로 올렸다"만으로 수락된 논문은 0건.**

수락된 배포 논문들의 공통점:
- Google 4-bit Conformer (INTERSPEECH 2022) → **새 방법론** (native integer QAT)
- Amazon sub-8-bit (SLT 2022) → **새 알고리즘** (General Quantizer)
- Apple Conformer Watch (NAACL 2024 Industry) → **수치 안정성 이론** + 아키텍처 수정
- Qualcomm Edge-ASR (2025) → **8개 PTQ 방법 체계적 비교**

전부 **"왜/어떻게"에 대한 기여**가 있음. "T527에 올렸다"는 기여가 아님.

### 우리가 가진 것

| 자산 | 기여로 인정? | 보강 필요? |
|------|:-----------:|:---------:|
| T527에 처음 올렸다 | ❌ | — |
| 한국어 최초 | ❌ | — |
| 6개 아키텍처 중 Conformer만 생존 | △ | "왜"를 kurtosis/activation으로 증명 필요 |
| MarginLoss (m=0.3, λ=0.1) | O | ablation 필수 (margin sweep, λ sweep) |
| FakeQuantize 과적합 (100k > 4.09M) | O | training curve로 시각화 필요 |
| CTC hallucination 억제 (INT8 > FP32) | O | 보조 발견으로 충분 |
| int16 < int8 | O | 분석하면 추가 기여 |

---

## 논문 프레이밍

### 타이틀 (안)

> "Why Only Conformer Survives W8A8: Architecture-Aware Quantization Analysis and MarginLoss for Korean ASR on a 2 TOPS NPU"

### 기여 3개

1. **아키텍처 생존 분석** — 6개 ASR 모델의 W8A8 양자화 결과를 per-layer kurtosis, dynamic range, activation histogram으로 정량 분석. Conformer의 depthwise conv가 attention 이후 activation을 안정화(kurtosis < 10)하는 반면, 순수 Transformer(Wav2Vec2, Zipformer)는 kurtosis > 100으로 INT8 uniform quantization이 붕괴함을 증명.

2. **MarginLoss QAT** — 2,049 클래스 CTC에서 uint8 step size 0.19 문제를 해결하는 margin-based auxiliary loss 제안. Ablation으로 margin 값, λ 값, FakeQuantize 위치별 효과 검증. PTQ 열화의 78% 회복.

3. **FakeQuantize 과적합 발견** — 84시간(100k) QAT가 4,356시간(full) QAT보다 device CER이 좋은 현상을 training dynamics 분석으로 규명. 실용적 가이드라인 제시.

보조 발견: CTC hallucination 억제 (Section 6, fixed-window INT8 > full-length FP32)

---

## 필수 보강 실험

### A. 아키텍처 생존 분석 (GPU 거의 불필요, FP32 forward pass)

| # | 실험 | 방법 | 예상 소요 |
|---|------|------|----------|
| A1 | Per-layer activation histogram | Conformer conv 출력 vs Wav2Vec2 attention 출력 히스토그램 비교 | 반나절 |
| A2 | Per-layer kurtosis 비교 | 각 레이어 kurtosis(첨도) 측정, 그래프 (X: layer, Y: kurtosis) | 반나절 |
| A3 | Per-layer dynamic range | max activation magnitude per layer 비교 | A2와 동시 |

**산출물:** Figure 1개 (2-3 서브플롯)
- (a) Conformer layer 12 conv 출력 histogram (near-Gaussian, kurtosis ~3-10)
- (b) Wav2Vec2 layer 12 attention 출력 histogram (power-law, kurtosis >> 100)
- (c) Per-layer kurtosis: Conformer(파랑, 낮고 평평) vs Wav2Vec2(빨강, 높고 치솟음)

**과학적 근거 (선행연구):**
- PTQ4ViT (ECCV 2022): ViT의 softmax 출력이 power-law → INT8에서 붕괴
- Quantizable Transformers (NeurIPS 2023): "no-op" attention head가 outlier 생성, kurtosis 3076
- Bondarenko et al. (EMNLP 2021): activation quantization(W32A8)만으로 GLUE 83→71 붕괴

### B. MarginLoss Ablation (GPU 필요)

| # | 실험 | 설정 | 예상 소요 |
|---|------|------|----------|
| B1 | Margin 값 sweep | m = {0.0, 0.1, 0.2, 0.3, 0.5} 각각 QAT → PTQ → device CER | GPU 10시간 |
| B2 | λ 값 sweep | λ = {0.0, 0.05, 0.1, 0.2} (m=0.3 고정) | GPU 8시간 |
| B3 | Component ablation | Row1: PTQ only → Row2: QAT(m=0,λ=0) → Row3: QAT+MarginLoss | B1에 포함 |
| B4 | FakeQuantize 위치별 | 3곳(enc_in, enc_out, dec_out) 중 하나씩 빼고 비교 | GPU 6시간 |

**산출물:**
- Table: margin 값별 CER (m=0.0이 baseline, m=0.3이 최적임을 보여야)
- Table: λ 값별 CER
- Table: component ablation (각 요소 기여 증명)
- Table: FakeQuantize 위치 ablation

**핵심:** m=0.0(MarginLoss 없음)의 device CER이 m=0.3보다 **확실히 나빠야** MarginLoss가 기여로 인정됨. 만약 차이가 작으면 MarginLoss 기여가 약해지고, 논문 방향을 아키텍처 분석(A) 중심으로 전환해야 함.

### C. Training Dynamics (이전 학습 로그 활용)

| # | 실험 | 방법 | 예상 소요 |
|---|------|------|----------|
| C1 | 100k vs full training curve | train loss, val loss 곡선 비교 (train-val gap 시각화) | 로그 있으면 바로 |
| C2 | MarginLoss 유무 수렴 비교 | m=0 vs m=0.3 loss 곡선 | B1 결과에서 추출 |

**산출물:** Figure 1개
- 100k: train-val gap 작음 (정상)
- full: train-val gap 큼 (FakeQuantize 과적합)

---

## 학회 수락 기준 (SLT 기준)

### 필수 (없으면 리젝)

- [x] Bit-width 비교 (FP32 vs PTQ vs QAT) — 이미 있음 (Table 2)
- [ ] **Component ablation** (각 요소 하나씩 더하기) — **없음, 해야 함**
- [ ] **Hyperparameter sweep** (margin, λ) — **없음, 해야 함**
- [x] 기존 방법 비교 (PTQ vs QAT) — 이미 있음
- [x] 다수 데이터셋 (11개) — 이미 있음

### 강력 권장 (있으면 수락 확률 크게 올라감)

- [ ] **Training curve** — **없음, 로그에서 추출 필요**
- [ ] **Per-layer sensitivity 분석** — **없음, 해야 함 (A 실험)**
- [x] On-device latency + 모델 크기 — 이미 있음 (Table 3)
- [x] 다수 아키텍처 비교 — 이미 있음 (Table 1)
- [ ] **Activation 분포 시각화** — **없음, 해야 함 (A 실험)**

---

## 타임라인

| 기간 | 할 일 | 비고 |
|------|------|------|
| **4/7-4/13** | A 실험 (kurtosis, activation histogram) | GPU 거의 불필요 |
| **4/14-4/20** | B1-B2 (margin sweep, λ sweep) | GPU 집중 |
| **4/21-4/27** | B3-B4 (component ablation, FQ 위치) + C (training curve) | GPU + 로그 |
| **4/28-5/11** | device 검증 (각 QAT 모델 → PTQ → NB → T527 CER) | T527 필요 |
| **5/12-5/25** | 논문 재작성 (LaTeX, 6페이지) | |
| **5/26-6/7** | 내부 리뷰 + 수정 | |
| **6/8-6/17** | 최종 제출 | 마감 6/17 |

---

## 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| MarginLoss ablation에서 차이가 작음 | MarginLoss 기여 약화 | 아키텍처 분석(A)을 메인으로 전환 |
| kurtosis 측정에서 Conformer도 높게 나옴 | "왜 Conformer만 생존" 설명 약화 | dynamic range, weight 분포 등 다른 지표 탐색 |
| device 검증에 시간 초과 | 마감 못 맞춤 | server CER로 대체 (device CER은 일부만) |
| SLT 리뷰어가 "단일 하드웨어" 지적 | 일반화 의문 | Limitation에 명시 + "W8A8 NPU 전반 적용 가능" 논리 |

---

## 참고 논문 (우리가 비교해야 할 것들)

| 논문 | 학회 | 기여 | 우리와 차이 |
|------|------|------|------------|
| Google 4-bit Conformer | INTERSPEECH 2022 | Native integer QAT | 4-bit, 영어, GPU 기반 |
| Amazon sub-8-bit | SLT 2022 | General Quantizer | sub-8-bit, 영어, custom accelerator |
| Amazon Accelerator-Aware | SLT 2022 | NNA 에뮬레이션 QAT | 영어, 270K hours |
| DQ-Whisper | SLT 2024 | Joint KD+Quantization | Whisper, 다국어 |
| Apple Conformer Watch | NAACL 2024 | LayerNorm 안정성 이론 | FP16, Apple ANE |
| PTQ4ViT | ECCV 2022 | Hessian-guided metric | Vision, 비 ASR |
| Quantizable Transformers | NeurIPS 2023 | Clipped softmax | NLP, 비 ASR |

**우리 논문의 차별점:**
- **W8A8 (activation도 8-bit)** — 대부분 W4A16 or weight-only
- **실제 NPU 배포** — 대부분 GPU 시뮬레이션
- **한국어 (2,049 BPE)** — 대부분 영어 (32 chars)
- **아키텍처 간 생존 비교** — 대부분 단일 모델
- **MarginLoss for large-vocab CTC** — 대부분 small-vocab or RNN-T
