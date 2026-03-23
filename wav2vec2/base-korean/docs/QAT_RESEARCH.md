# QAT (Quantization-Aware Training) 연구 조사 및 적용 방안

**작성일:** 2026-03-24
**목적:** T527 NPU uint8 양자화에서 한국어 wav2vec2 동작을 위한 QAT 기법 조사

---

## 1. 배경: 왜 QAT가 필요한가

### 1.1 우리의 문제

한국어 wav2vec2-base-korean (94.4M params, 12L Transformer)이 T527 NPU uint8 양자화 시 CER 100% (완전 실패).
동일 아키텍처의 영어 모델은 CER 17.52%로 정상 동작.

**근본 원인: Logit Margin < Quantization Step Size**

```
영어:  margin=0.34,  uint8 step=0.08  → margin/step=4.3x  → argmax 보존 ✓
한국어: margin=0.005, uint8 step=0.05  → margin/step=0.1x  → argmax 뒤집힘 ✗
```

PTQ (Post-Training Quantization)로는 해결 불가 — 21종+ 시도 전부 실패:
- uint8 asymmetric_affine (다양한 calibration)
- KL divergence + fqb16
- Per-channel symmetric (PCQ)
- SmoothQuant, Range Clipping
- Hybrid int16/uint8 (디바이스 크래시)
- bf16, fp16 (NB export 실패 / HW 미가속)

**추가 발견: Acuity 시뮬레이션 ≠ T527 디바이스 (31.5% argmax 일치)**
→ 시뮬레이션 기반 최적화는 디바이스 결과를 예측 못함.

### 1.2 QAT의 핵심 아이디어

> 학습 중에 양자화를 시뮬레이션하면, 모델이 양자화 오류를 보상하는 가중치를 스스로 학습한다.

PTQ: 학습된 모델 → (양자화) → 오류 발생 → 사후 보정 시도
QAT: 학습 중 양자화 시뮬레이션 → 모델이 양자화에 강건한 가중치 직접 학습

---

## 2. QAT 핵심 메커니즘

### 2.1 Fake Quantization (가짜 양자화)

Forward pass에서 텐서를 양자화(int으로 반올림)한 후 즉시 역양자화(float으로 복원).
연산 자체는 float로 수행되지만, 양자화 오류가 loss에 반영됨.

```python
# Forward (fake quantize)
x_q = clamp(round(x / scale + zero_point), 0, 255)
x_dq = (x_q - zero_point) * scale     # ← 이 값으로 계산 계속
# x와 x_dq의 차이가 양자화 오류 → loss에 반영 → 역전파로 보정
```

### 2.2 Straight-Through Estimator (STE)

양자화 함수(round)의 미분은 거의 어디서나 0 → 역전파 불가.
STE는 양자화 노드를 **항등 함수**로 취급하여 gradient를 그대로 통과시킨다.

```
Forward:  x → round(x/s + z) → (x_q - z) * s = x_dq
Backward: ∂L/∂x ≈ ∂L/∂x_dq  (STE: gradient 그대로 통과)
```

이론적으로 부정확하지만, 실전에서 매우 효과적.
최근 연구(2025, arXiv:2505.18113)에서 STE의 수렴 보장이 이론적으로도 증명됨.

### 2.3 Observer (관측기)

양자화 파라미터(scale, zero_point)를 결정하기 위해 activation 분포를 관찰.

| Observer 종류 | 방법 | 장단점 |
|---|---|---|
| MinMax | 전체 min/max 사용 | 단순, outlier에 민감 |
| MovingAverageMinMax | EMA로 min/max 추적 | 안정적, 실전 기본값 |
| Histogram (KL) | KL divergence로 최적 clipping | 정밀, 느림 |
| PerChannel | 채널별 독립 scale | weight 양자화에 효과적 |

**PyTorch 권장:**
- Weight: symmetric per-channel + MinMax
- Activation: affine per-tensor + MovingAverageMinMax

### 2.4 양자화 Granularity (입도)

| 단위 | 설명 | 정확도 | 하드웨어 지원 |
|------|------|--------|-------------|
| Per-tensor | 텐서 전체에 1개 scale/zp | 낮음 | 모든 HW |
| Per-channel | 채널마다 별도 scale/zp | 높음 | 대부분 HW |
| Per-group | 그룹(예: 32개)마다 별도 | 매우 높음 | 일부 HW만 |
| Per-token | 토큰마다 별도 (activation) | 최고 | GPU만 |

**T527 NPU (VIP9000NANOSI_PLUS):**
- 지원: per-tensor asymmetric_affine uint8만 안정
- per-channel (PCQ int8): NB export 가능하나 디바이스에서 결과 악화
- per-group: 미지원

---

## 3. ASR 분야 QAT 논문 분석

### 3.1 4-bit Conformer with Native QAT (Google, INTERSPEECH 2022)

**arXiv:2203.15952** — Ding et al.

- Conformer ASR 모델을 4-bit까지 양자화 (LibriSpeech)
- "Native QAT": 학습 중 실제 정수 연산으로 forward pass → 시뮬레이션 ≠ 디바이스 갭 제거
- **결과**: float32 대비 lossless 4-bit 모델, 7.7x 크기 축소
- **Mixed 4/8-bit**: practical ASR에서 lossless, 5x 축소
- **핵심**: 학습 시간 +7%만 증가

**우리 상황에 적용:**
- Native QAT는 Acuity/Vivante NPU에서는 불가 (커스텀 정수 연산 커널 필요)
- 하지만 mixed precision 아이디어는 적용 가능 — 민감한 레이어(L8-11)만 8-bit, 나머지 4-bit

### 3.2 Absolute-Cosine Regularization (Amazon, INTERSPEECH 2020)

**Amazon Science** — Nguyen et al.

- RNN-T ASR 모델에 QAT 적용
- **ACosR (Absolute-Cosine Regularizer)**: weight 분포를 양자화 centroid 근처로 몰리게 하는 정규화
- **결과**: float32 ≈ 8-bit ≈ 6-bit (zero to little degradation)
- weight 분포가 실제로 양자화 레벨 근처에 cluster됨을 확인

**핵심 아이디어:**
```python
# ACosR: weight를 가장 가까운 양자화 레벨로 밀어주는 정규화
loss = CTC_loss + lambda * sum(|cos(pi * w / step)|)
# cos(pi * w/step) = 0일 때 w가 정확히 양자화 레벨 위에 있음
```

**우리 상황에 적용:**
- weight를 양자화 friendly하게 만드는 직접적 방법
- 구현 간단 (loss에 항 하나 추가)
- activation의 margin 문제는 직접 해결 못함 → activation regularization과 병행 필요

### 3.3 Sub-8-Bit QAT for 8-Bit NPU (Amazon, INTERSPEECH 2022)

**arXiv:2206.15408** — Zhen et al.

- **핵심**: 8-bit NPU에서 돌릴 모델을 sub-8-bit로 학습 → 8-bit 양자화 시 더 강건
- MRACos (Multi-Regional ACos): Lloyd-Max 압축 이론 기반
- **결과**: 모델 크기 늘려도 WER 4-16% 상대 개선, latency 5% 개선

**우리 상황에 적용:**
- T527은 uint8 NPU → sub-8-bit QAT 접근이 정확히 우리 상황
- 더 낮은 비트로 학습하면 8-bit에서의 내성이 극대화

### 3.4 Edge-ASR: Low-Bit Quantization (Qualcomm, 2025)

**arXiv:2507.07877**

- Whisper + Moonshine에 8가지 SOTA PTQ 적용 벤치마크
- **3-bit까지도 가능** (high capacity 모델 + advanced PTQ)
- W3A8에서 10x 메모리 절감, negligible accuracy loss
- **주의**: 이건 PTQ 벤치마크이고, Whisper/Moonshine은 원래 양자화에 강건한 모델

### 3.5 ENERZAi: Korean ASR with 1.58-bit Whisper (2025)

**enerzai.com** — 한국어 ASR에 가장 직접적으로 관련된 사례

- Whisper Small을 50K 시간 한국어 데이터로 재학습
- 1.58-bit 극저비트 양자화 (ternary: {-1, 0, +1} × scale)
- **결과**: CER 18.05% → 6.45% (한국어), 13MB 모델이 3GB Large 능가
- **group size = channel size**가 최적 (정확도/크기/속도 Pareto)
- Synaptics NPU에서 on-device 실행 확인

**우리 상황에 적용:**
- Whisper는 encoder-decoder 구조로 wav2vec2와 다름
- 하지만 "대량 데이터 + QAT"가 핵심이라는 점은 동일
- 1.58-bit까지 가능하다는 건 8-bit QAT는 충분히 실현 가능하다는 의미
- **50K 시간 데이터가 핵심 — 우리의 50시간은 매우 적음**

### 3.6 R2 Loss: Range Restriction (2023)

**arXiv:2303.08253**

- Pre-training 시 weight의 outlier를 제거하여 양자화 friendly하게 만듦
- 3가지 변형: L-inf, Margin, Soft-Min-Max
- **결과**: MobileNet-V2 2-bit PTQ 50.66% → 59.49% (+8.8%p)
- 양자화 전 학습 단계에서 range를 제한하면 PTQ 결과가 대폭 개선

**우리 상황에 적용:**
- L10 Add의 range 420 문제를 학습 단계에서 해결 가능
- R2-Loss를 추가하면 outlier가 줄어들어 uint8 step이 작아짐

### 3.7 Learned Step Size Quantization (LSQ, ICLR 2020)

**arXiv:1902.08153** — Esser et al.

- 양자화 step size를 학습 가능한 파라미터로 취급
- 각 layer별 독립적 step size → layer 특성에 맞는 최적 양자화
- **결과**: 3-bit에서 full precision 정확도 달성 (ImageNet)

**우리 상황에 적용:**
- Acuity가 per-layer scale을 결정하므로 직접 적용은 어려움
- 하지만 학습 중 per-layer step size를 최적화하면 → export 후 Acuity의 scale이 자연스럽게 좋아짐

### 3.8 Knowledge Distillation + QAT (EMNLP 2022)

**arXiv:2211.11014** — BERT Transformer 양자화

- FP32 teacher → quantized student로 knowledge distillation
- attention-map loss + attention-output loss가 MSE보다 효과적
- **결과**: quantized BERT의 정확도를 크게 회복

**우리 상황에 적용:**
- FP32 wav2vec2를 teacher로, uint8 QAT 모델을 student로 distillation
- attention layer(L8-11)의 출력을 teacher와 맞추도록 학습
- 구현: teacher model forward → student forward → attention KD loss + CTC loss

---

## 4. 우리 상황에 대입: 적용 가능한 방법들

### 4.1 현재 QAT (진행 중)

**방법:** attempt5(WER 40.6%) → FakeQuantize wrapper 삽입 → 30 epoch 학습
**현황:** WER 41.3% → 38.86% (개선 중)
**한계:** 기본 STE + per-tensor fake quant — 가장 단순한 형태

### 4.2 권장 개선 방안 (우선순위순)

#### 방안 A: Margin Loss 추가 (즉시 적용 가능, 1시간)

현재 QAT에 margin loss를 추가하여 logit margin을 직접 늘린다.

```python
# Top-1과 Top-2 logit의 차이를 최소 margin_target 이상으로 강제
sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
margins = sorted_logits[:, :, 0] - sorted_logits[:, :, 1]
margin_target = 0.1  # uint8 step의 ~2배
margin_loss = F.relu(margin_target - margins).mean()
total_loss = ctc_loss + 0.1 * margin_loss
```

**장점:** 우리 문제(margin < step)를 직접 해결
**단점:** CTC 수렴과 충돌할 수 있음 (CTC는 blank을 많이 출력하려 하고, margin loss는 비-blank 차이를 벌리려 함)

#### 방안 B: ACosR (Absolute-Cosine Regularization) (2시간)

Amazon이 검증한 방법. weight를 양자화 centroid 근처로 몰리게.

```python
# step = expected uint8 step size (range / 255)
acos_reg = torch.mean(torch.abs(torch.cos(math.pi * weight / step)))
total_loss = ctc_loss + lambda_acos * acos_reg
```

**장점:** weight 분포가 양자화에 최적화됨 (논문에서 6-bit까지 lossless 검증)
**단점:** activation margin은 직접 제어 못함

#### 방안 C: R2 Loss (Range Restriction) (2시간)

L10 Add의 range 420 문제를 학습에서 해결.

```python
# Soft-Min-Max R2 Loss on L8-11 activations
for name, module in model.named_modules():
    if 'layers.10' in name or 'layers.11' in name:
        # Hook으로 activation 캡처, range penalty 추가
        range_loss += softmax(activation.max() - activation.min() - target_range)
```

**장점:** outlier를 학습에서 제거 → PTQ 후에도 좋은 range
**단점:** activation hook 필요, 구현 복잡도 높음

#### 방안 D: Knowledge Distillation + QAT (4시간)

FP32 model을 teacher로 사용.

```python
# Teacher: FP32 wav2vec2 (no quantization)
# Student: QAT wav2vec2 (fake quantized)
# Loss = CTC_loss + alpha * KD_loss(student_logits, teacher_logits)
kd_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * T * T
```

**장점:** teacher가 "정답" 출력 분포를 제공 → student가 양자화 속에서도 teacher를 따라감
**단점:** 메모리 2배 (teacher + student 동시 forward), 구현 복잡

#### 방안 E: 대용량 데이터 QAT (서버 필요, 1일+)

NAS 4356시간 데이터로 QAT.

**장점:** ENERZAi 사례에서 50K시간 데이터가 핵심이었음. 데이터 양이 곧 성능.
**단점:** GPU 시간, 데이터 접근 필요

#### 방안 F: Whisper로 모델 교체 (1~2주)

wav2vec2 대신 Whisper tiny/small 사용.
- Whisper는 encoder-decoder로 양자화에 더 강건 (ENERZAi 1.58-bit 성공)
- 한국어 Whisper 모델이 이미 다수 존재 (HuggingFace)
- **하지만 T527 NPU에서 Whisper가 동작하는지 미검증**

---

## 5. 구현 우선순위 로드맵

```
[현재] QAT 기본 (진행 중, WER 38.86%)
  ↓ 결과 확인 (ONNX → NB → 디바이스 CER)
  ↓
[1순위] Margin Loss 추가 QAT (1시간)
  ↓ margin > step 달성 여부 확인
  ↓
[2순위] ACosR + Margin Loss 병행 (2시간)
  ↓
[3순위] Knowledge Distillation + QAT (4시간)
  ↓
[4순위] 대용량 데이터 QAT (서버, 1일)
  ↓
[최후] Whisper 모델 교체 (1~2주)
```

---

## 6. T527 NPU 특수 제약사항

| 제약 | 영향 | 대응 |
|------|------|------|
| uint8 only 안정 | int16/bf16 NB → 크래시 또는 실패 | QAT로 uint8 내성 강화 |
| per-tensor quant only | per-channel 미지원 | per-tensor 기준 QAT |
| NB 크기 ~120MB 제한 | int16/fp16 NB 실행 불가 | uint8 (72MB) 유지 |
| 시뮬레이션 ≠ 디바이스 | 시뮬 최적화 무의미 | **반드시 디바이스 직접 테스트** |
| SRAM 제한 | fp32 (362MB) 실행 불가 | 양자화 필수 |

---

## 7. 실험 결과 (2026-03-24)

### 7.1 QAT 기본 (FakeQuantize only, 30 epoch)

시작: attempt5 (WER 40.6%, margin_min=0.037)
완료: WER **38.86%**, margin_min=**0.099** (20배 개선)

| 지표 | attempt5 (baseline) | QAT |
|------|-------------------|-----|
| WER | 40.6% | **38.86%** |
| margin_min | 0.037 | **0.099** |
| uint8 step | ~0.15 | 0.151 |
| ratio (margin/step) | 0.25x | **0.66x** |
| uint8 생존 | NO | NO (근접) |

margin이 0.037 → 0.099로 3배 증가했지만 아직 step(0.151)보다 작다.

### 7.2 QAT + Margin Loss (진행 중, 20 epoch)

시작: QAT 결과 (WER 38.86%)
변경: non-blank 프레임의 margin을 0.2 이상으로 강제하는 margin loss 추가
진행 중 — 결과 대기.

**초기 문제**: blank 프레임 포함 시 min_margin=0.0000 (blank은 원래 margin 0)
**수정**: non-blank 프레임만 margin loss 적용

---

## 8. 핵심 교훈 및 Premise

1. **PTQ로 안 되면 QAT** — PTQ 21종 실패 후 QAT가 유일한 대안
2. **Margin > Step이 uint8 양자화 성공의 필요조건** — logit margin 분석으로 사전 판단 가능
3. **시뮬레이션 ≠ 디바이스** — NPU 시뮬레이션 결과를 신뢰하지 말 것
4. **데이터 양이 곧 성능** — ENERZAi 50K시간 vs 우리 50시간
5. **모델 아키텍처 선택이 양자화 성패를 좌우** — CNN(Citrinet)은 쉬움, Transformer는 어려움

---

## 참고 문헌

1. [4-bit Conformer with Native QAT (Google, INTERSPEECH 2022)](https://arxiv.org/abs/2203.15952)
2. [ACosR: QAT with Absolute-Cosine Regularization (Amazon, INTERSPEECH 2020)](https://www.amazon.science/publications/quantization-aware-training-with-absolute-cosine-regularization-for-automatic-speech-recognition)
3. [Sub-8-Bit QAT for 8-Bit NPU (Amazon, INTERSPEECH 2022)](https://arxiv.org/abs/2206.15408)
4. [Edge-ASR: Low-Bit Quantization Benchmark (Qualcomm, 2025)](https://arxiv.org/abs/2507.07877)
5. [ENERZAi: Korean ASR 1.58-bit Whisper (2025)](https://enerzai.com/resources/blog/small-models-big-heat-conquering-korean-asr-with-low-bit-whisper)
6. [R2 Loss: Range Restriction (2023)](https://arxiv.org/abs/2303.08253)
7. [LSQ: Learned Step Size Quantization (ICLR 2020)](https://arxiv.org/abs/1902.08153)
8. [KD + QAT for Transformer Encoders (EMNLP 2022)](https://arxiv.org/abs/2211.11014)
9. [SmoothQuant: W8A8 for Transformers (2022)](https://arxiv.org/abs/2211.10438)
10. [PyTorch Quantization Best Practices](https://pytorch.org/blog/quantization-in-practice/)
11. [NVIDIA QAT with TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
12. [STE: Beyond Discreteness (2025)](https://arxiv.org/abs/2505.18113)
13. [Teacher Intervention for QAT Convergence (EACL 2023)](https://aclanthology.org/2023.eacl-main.64.pdf)
14. [SpeechBrain Model Quantization Tutorial](https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/advanced/model-quantization.html)
15. [OpenVINO Wav2Vec2 Quantization](https://docs.openvino.ai/2024/notebooks/speech-recognition-quantization-wav2vec2-with-output.html)
