# 실험 로그 2026-03-23~24 — 전체 정리

## 1. Fine-tuning 학습 (attempt1~7)

영어 `facebook/wav2vec2-base-960h` → 한국어 CTC fine-tune (Zeroth-Korean 50시간)

| Attempt | Epochs | LR | Frozen | WER | Margin min | 비고 |
|---------|--------|-----|--------|-----|------------|------|
| 1 | 30 | 1e-4 | CNN | 100% | 0.0000 | CTC 수렴 실패 |
| 2 | +50 | 3e-5 | CNN+L0-5 | 54.18% | 0.0196 | blank 탈출 |
| 3 | +50 | 1e-5 | CNN | 44.04% | 0.0018 | T527 최초 한국어 출력 |
| 4 | +30 | 5e-6 | CNN | 42.06% | 0.0110 | |
| 5 | +50 | 2e-6 | CNN | 40.60% | 0.0365 | |
| 6 | +100 | 1e-6 | CNN | **39.38%** | 0.0245 | 수렴 |
| 7 | +50 | 5e-7 | CNN | **39.23%** | — | 수렴 확정 |

## 2. QAT (Quantization-Aware Training)

| QAT 방식 | FQ 개수 | Epochs | WER | Margin mean | Margin min |
|---------|--------|--------|-----|-------------|------------|
| Manual (attn+head) | 50 | 30 | 41.73% | 8.96 | 0.0006 |
| Manual (ALL Linear) | **73** | 15 | 42.48% | **9.32** | **0.0181** |
| 기준: attempt7 (no QAT) | 0 | — | 39.23% | 6.74 | 0.0245 |

QAT로 **margin mean 38% 개선** (6.74→9.32)되었으나, T527 uint8 결과에서 의미있는 개선 없음.

## 3. T527 NPU 테스트 비교

### attempt6 uint8 NB (72MB, ~400ms)

| 샘플 | GT | NPU |
|------|-----|------|
| ko_test_0003 | 그리고 이 나무는... | 그리구어기 나 |
| ko_test_0006 | 야권은 여당에서... | 약구워된 여단 |
| ko_test_0009 | 이번에 발견된... | 이 벋에 발껼께 |

### QAT uint8 NB (72MB, ~400ms)

| 샘플 | GT | NPU |
|------|-----|------|
| ko_test_0003 | 그리고 이 나무는... | 그리구 어기 난 |
| ko_test_0006 | 야권은 여당에서... | 야궈든 여당 |
| ko_test_0009 | 이번에 발견된... | 이벋에 발껼 깨아 |

**거의 동일.** QAT 효과 미미.

### 슬라이딩 윈도우 (attempt6, 전체 음성)

| 샘플 | 길이 | GT | NPU (sliding) |
|------|------|-----|------|
| ko_test_0003 | 16.6초 | 그리고 이 나무는 태즈메이니아 남부...자랍니다 | 그리구어기 나스 메이 미아 남부 보는...랍니다 |
| ko_test_0006 | 8.9초 | 야권은 여당에서...높이고 있다 | 약구워된 여단...오끼고 있다 |
| ko_test_0009 | 14.6초 | 이번에 발견된 화석은...닮았다 | 이 벋에 발껼께 애우노...맜따 |

## 4. 다른 한국어 모델 탐색 (40건 시도)

### NB 생성 + T527 동작 성공

| 모델 | NB | 추론 | 한국어 | 비고 |
|------|-----|------|--------|------|
| **KoCitrinet 256** | **62MB** | **120ms** | **CER 8.44%** | **최고, 운용중** |
| Wav2Vec2 KO fine-tune | 72MB | ~400ms | WER 39% | 부분 동작 |
| EN Conformer small | 14MB | 74ms | 영어만 | fine-tune 필요 |
| Whisper tiny encoder | 117MB | 937ms | Enc-Dec | Decoder 필요 |
| HuBERT KO | 76MB | 423ms | 실패 | 동일 토큰 반복 |

### NB 생성 실패

| 모델 | 원인 |
|------|------|
| KO Conformer medium (SpeechBrain) | error 64768 (42.9M 너무 큼) |

### 양자화 실패 (NB는 생성되나 출력 garbage)

| 모델 | 원인 |
|------|------|
| Wav2Vec2 base-korean (기존) | logit margin 0.005 < uint8 step 0.05 |
| HuBERT base Korean | Transformer 양자화 문제 |
| Zipformer | 5868 nodes 에러 누적 |

## 5. 레이어별 분석

영어 vs 한국어 레이어별 SNR은 **비슷** (31~36 dB). 차이는 **logits에서만**:
- 영어: logit std **8.39**, margin min **0.34** → uint8 생존
- 한국어: logit std **1.95**, margin min **0.005** → uint8 실패

**중간 레이어 양자화 에러 누적이 아니라, LM head 출력 분포 자체의 문제.**

## 6. 핵심 발견

1. **T527 NPU에서 한국어 wav2vec2 uint8 동작 가능** — 영어 base-960h에서 fine-tune하면 됨
2. **KoCitrinet이 여전히 최고** — CNN 기반, uint8 양자화에 강함, CER 8.44%
3. **Transformer 기반 한국어 모델은 uint8 양자화 비적합** — logit margin 부족
4. **QAT는 margin mean만 개선** — margin min은 개선 안 됨, 실제 효과 미미
5. **데이터가 핵심** — 50시간(Zeroth-Korean)으로는 한계. NAS 4356시간 필요
6. **영어 Conformer small (14MB, 74ms) NB 성공** — 한국어 fine-tune하면 최적의 모델 될 가능성

## 7. 다음 단계

1. NAS 접속 복구 → 4356시간 데이터로 재학습
2. 영어 Conformer small → 한국어 fine-tune (14MB NB, 74ms)
3. KoCitrinet + Language Model (CER 8.44% → 개선)
4. QAT margin min 개선 (custom loss 필요)
