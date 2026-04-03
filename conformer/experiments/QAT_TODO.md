# QAT 실험 TODO List

**목표:** SLT 2026 논문용 ablation + 최적 QAT 레시피 탐색
**GPU:** RTX 6000 Ada (48GB)
**기준 모델:** SungBeom Conformer CTC Medium (122.5M, 2049 BPE)
**현재 최고:** 1M × 1ep = 8.86% (avg_all)

---

## 1. MarginLoss Ablation (논문 필수)

### 1-1. Margin 값 sweep (λ=0.1 고정)

| # | margin (m) | λ | 데이터 | epoch | 상태 |
|---|-----------|---|--------|-------|------|
| M1 | 0.0 (no margin) | 0.0 | 100k | 10 | |
| M2 | 0.1 | 0.1 | 100k | 10 | |
| M3 | 0.2 | 0.1 | 100k | 10 | |
| M4 | 0.3 | 0.1 | 100k | 10 | ✅ 9.30% |
| M5 | 0.5 | 0.1 | 100k | 10 | |
| M6 | 1.0 | 0.1 | 100k | 10 | |

**핵심:** M1(no margin)이 M4보다 확실히 나빠야 MarginLoss 기여 증명됨.

### 1-2. λ 값 sweep (m=0.3 고정)

| # | margin | λ | 데이터 | epoch | 상태 |
|---|--------|---|--------|-------|------|
| L1 | 0.3 | 0.0 | 100k | 10 | (= M1과 동일) |
| L2 | 0.3 | 0.01 | 100k | 10 | |
| L3 | 0.3 | 0.05 | 100k | 10 | |
| L4 | 0.3 | 0.1 | 100k | 10 | ✅ 9.30% |
| L5 | 0.3 | 0.2 | 100k | 10 | |
| L6 | 0.3 | 0.5 | 100k | 10 | |

---

## 2. FakeQuantize 위치 Ablation (논문 필수)

| # | enc_input | enc_output | dec_output | 데이터 | 상태 |
|---|:---------:|:----------:|:----------:|--------|------|
| F1 | O | O | O | 100k 10ep | ✅ 9.30% |
| F2 | X | O | O | 100k 10ep | |
| F3 | O | X | O | 100k 10ep | |
| F4 | O | O | X | 100k 10ep | |
| F5 | X | X | O | 100k 10ep | |
| F6 | O | X | X | 100k 10ep | |

**핵심:** 어느 위치가 제일 중요한지 밝혀야 함. dec_output(CTC logits)이 제일 중요할 것으로 예상.

---

## 3. 학습 데이터 크기/전략

### 3-1. 데이터 양 × epoch 조합

| # | 데이터 | epoch | 총 샘플 수 | 상태 |
|---|--------|-------|-----------|------|
| D1 | 50k | 10 | 500k | |
| D2 | 100k | 10 | 1M | ✅ 9.30% |
| D3 | 100k | 5 | 500k | |
| D4 | 100k | 3 | 300k | |
| D5 | 100k | 1 | 100k | |
| D6 | 200k | 5 | 1M | |
| D7 | 500k | 2 | 1M | |
| D8 | 1M | 1 | 1M | ✅ 8.86% |
| D9 | 2M | 1 | 2M | |
| D10 | 4.09M (full) | 1 | 4.09M | |
| D11 | 4.09M (full) | 10 | 40.9M | ✅ 14.81% |

**핵심:** "총 샘플 수"가 같을 때 다양성(unique 샘플)이 중요한지, 반복이 중요한지 규명. D6, D7은 D2, D8과 총량 같지만 구성이 다름.

### 3-2. 데이터 품질/도메인 선별

| # | 전략 | 설명 | 상태 |
|---|------|------|------|
| Q1 | Random (현재) | AIHub 전체에서 random 100k | ✅ |
| Q2 | Clean only | SNR 높은 깨끗한 샘플만 100k | |
| Q3 | Noisy only | 저음질/잡음 샘플만 100k | |
| Q4 | Short only (≤3s) | 짧은 발화만 100k (fixed-window 최적화) | |
| Q5 | Long only (>5s) | 긴 발화만 100k | |
| Q6 | Balanced | 도메인별 균등 배분 100k | |
| Q7 | Hard sample mining | FP32에서 CER 높은 샘플 위주 100k | |
| Q8 | Curriculum | 쉬운 것부터 → 어려운 것 순서로 학습 | |

---

## 4. Calibration 데이터 (PTQ 단계)

| # | calib 소스 | 샘플 수 | 설명 | 상태 |
|---|-----------|---------|------|------|
| C1 | AIHub random | 100 | 현재 설정 | ✅ |
| C2 | AIHub random | 50 | 줄여도 되는지 | |
| C3 | AIHub random | 200 | 늘리면 좋아지는지 | |
| C4 | AIHub random | 500 | | |
| C5 | 자체 녹음 데이터 | 100 | 배포 환경 매칭 | |
| C6 | QAT 학습 데이터 | 100 | 학습-캘리 분포 일치 | |
| C7 | 007 저음질 | 100 | 노이즈 환경 | |
| C8 | Mixed (AIHub+자체) | 100 | 50:50 섞기 | |

**핵심:** 캘리 데이터와 학습 데이터 분포가 일치해야 한다는 가설 검증.

---

## 5. 학습 하이퍼파라미터

### 5-1. Learning rate

| # | lr | scheduler | 데이터 | 상태 |
|---|-----|-----------|--------|------|
| H1 | 1e-5 | cosine | 100k 10ep | ✅ 9.30% |
| H2 | 5e-6 | cosine | 100k 10ep | |
| H3 | 2e-5 | cosine | 100k 10ep | |
| H4 | 5e-5 | cosine | 100k 10ep | |
| H5 | 1e-5 | constant | 100k 10ep | |
| H6 | 1e-5 | warmup+cosine | 100k 10ep | |

### 5-2. Frozen layers

| # | Frozen | 설명 | 상태 |
|---|--------|------|------|
| FR1 | frontend only | 현재 설정 (conv subsampling만 freeze) | ✅ |
| FR2 | nothing | 전부 학습 | |
| FR3 | frontend + 처음 6 layers | 앞쪽 레이어는 양자화 영향 적을 수 있음 | |
| FR4 | frontend + 마지막 6 layers | 뒤쪽 freeze | |
| FR5 | encoder 전체 (decoder만 학습) | decoder head만 fine-tune | |

---

## 6. 추가 QAT 기법

### 6-1. Knowledge Distillation

| # | 방법 | 설명 | 상태 |
|---|------|------|------|
| KD1 | Logit KD | FP32 teacher의 softmax 출력과 student의 KL divergence | |
| KD2 | Feature KD | encoder 중간 레이어 feature 매칭 | |
| KD3 | CTC + Margin + KD (λ_kd=0.1) | 3개 loss 조합 (λ 작게) | |
| KD4 | CTC + KD only (no margin) | KD vs MarginLoss 비교용 | |

### 6-2. Progressive QAT

| # | 방법 | 설명 | 상태 |
|---|------|------|------|
| P1 | 2-stage | Stage1: CTC only 5ep → Stage2: CTC+Margin 5ep | |
| P2 | Gradual FQ | epoch 1-3: FQ noise 50% → epoch 4-10: FQ noise 100% | |
| P3 | Layer-wise progressive | epoch 1-3: decoder FQ만 → epoch 4-7: +enc_out → epoch 8-10: +enc_in | |

### 6-3. FakeQuantize 방식

| # | 방법 | 설명 | 상태 |
|---|------|------|------|
| FQ1 | Per-tensor asymmetric (현재) | 텐서 전체 1개 scale/zp | ✅ |
| FQ2 | Per-channel | 채널별 scale/zp (더 정밀) | |
| FQ3 | LSQ (Learned Step Quantization) | scale을 학습 가능 파라미터로 | |
| FQ4 | EWGS (gradient scaling) | gradient에 quantization error weight | |

---

## 7. Activation 분석 (논문 필수, 학습 불필요)

| # | 분석 | 방법 | 상태 |
|---|------|------|------|
| A1 | Per-layer kurtosis | Conformer vs Wav2Vec2 100 샘플 forward pass | |
| A2 | Per-layer dynamic range | max abs activation per layer | |
| A3 | Activation histogram | attention 후 vs conv 후 분포 비교 | |
| A4 | Per-layer quantization sensitivity | leave-one-out: 한 레이어만 FP32, 나머지 INT8 → CER | |

---

## 우선순위 (시간 없을 때)

### 반드시 해야 함 (논문 필수)
1. **M1** (no margin baseline) — MarginLoss 기여 증명
2. **M2, M3, M5** — margin sweep
3. **L2, L3, L5** — λ sweep
4. **F2, F3, F4** — FakeQuantize 위치 ablation
5. **A1, A2, A3** — activation 분석 (학습 불필요)

### 하면 좋음 (논문 강화)
6. **D5, D6, D7** — 데이터 양 vs 다양성 분석
7. **KD1, KD4** — KD vs MarginLoss 비교
8. **C2, C3, C4** — calib 수 영향

### 여유 있으면
9. 나머지 전부
