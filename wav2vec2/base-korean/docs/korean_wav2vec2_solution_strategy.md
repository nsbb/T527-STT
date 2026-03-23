# 한국어 Wav2Vec2 T527 NPU 배포 전략서

> 작성일: 2026-03-17 | 갱신: 2026-03-18 | 상태: ONNX 재변환 검증 완료, QAT/KD 실행 대기

## 1. 문제 요약

한국어 wav2vec2-base (94.4M, 12L Transformer)가 T527 NPU uint8에서 완전 실패.
영어 동일 아키텍처는 CER 17.52%로 정상 동작. **60종+ 시도, 21종 NPU 실측, 전부 실패.**

## 2. 근본 원인 (신규 발견 포함)

### 2.0 ONNX Export 구조 차이 — 2026-03-18 신규 발견

> **상세 분석: [onnx_structure_comparison.md](onnx_structure_comparison.md)**

**영어와 한국어 ONNX의 attention 구현이 완전히 다르다.**

| 항목 | EN (opset 12) | KO (opset 14) | 양자화 영향 |
|------|:----------:|:----------:|------------|
| **노드 수** | 957 | 1306 (+349) | 불필요한 복잡성 |
| **Scale 위치** | Q 선곱 (pre-MatMul) | Q@K^T 후곱 (post-MatMul) | MatMul 출력 8배 큼 → 거친 양자화 |
| **MatMul rank** | 3D `[12, seq, 64]` | 4D `[1, 12, seq, 64]` | Acuity 처리 차이 가능 |
| **Softmax axis** | axis=2 | axis=3 | Acuity op mapping 차이 |
| **동적 Shape ops** | 0 | 216 (18/layer × 12) | Acuity 호환성 문제 가능 |

**원인**: 현재 HuggingFace transformers (4.41+)가 `scaled_dot_product_attention` (SDPA)를 기본 사용.
SDPA는 ONNX opset 14+ 전용 → 다른 그래프 구조 생성.

**해결**: `attn_implementation="eager"` + `opset_version=12`로 재변환 → **EN과 957 nodes 100% 동일 구조**.

**검증 결과 (2026-03-18)**:
- Pegasus 시뮬: argmax agreement **46.3% → 78.1%** (+31.8%p)
- T527 NPU: 72MB NB, 415ms 추론, NPU vs FP32 agreement 89.9%
- **한계**: non-blank 프레임 accuracy 0% — attention 분포 근본 원인은 미해결

**영향 정도**: 중간. 근본 원인(attention 분포)이 지배적이지만, ONNX 구조가 양자화를 추가 악화시킴.
**비용**: 0 (재변환만). 후속 모든 전략의 기반 ONNX로 사용해야 함.

### 2.1 Attention 분포 차이 — 핵심 증거

| 지표 | English L11 | Korean L11 | 의미 |
|------|:----------:|:----------:|------|
| **top-1 attention** | **66.6%** | **1.8%** | EN: 한 위치에 집중 / KO: 거의 균일 분산 |
| **attention zeros** | **92.6%** | **2.7%** | EN: 대부분 0 / KO: 거의 없음 |
| **entropy** | **1.10** | **4.90** | EN: near one-hot / KO: near uniform |
| **uint8 argmax 보존** | **99.4%** | **51.0%** | EN: 무손실 / KO: 절반 파괴 |
| **unique uint8 값** | 256/256 | **51/256** | KO: 값이 좁은 범위에 몰림 |

**English**: near-one-hot attention (92.6% 0, 나머지 255) → uint8이 trivially 표현 가능.
**Korean**: near-uniform attention (모든 값이 ~0.007) → uint8이 구별 불가능.

### 2.2 왜 Korean이 uniform attention을 학습했나?

1. **attention_dropout=0.0**: Kkonjeong 모델은 attention dropout 없이 학습. English(0.1)과 다름.
   → Dropout이 없으면 attention이 극단적 패턴(uniform 또는 peaked)으로 수렴 가능.

2. **학습 출발점 차이**: Kkonjeong은 `wav2vec2-base` (pretrained)에서 출발.
   `wav2vec2-base-960h` (English CTC fine-tuned, uint8-friendly)에서가 아님.
   → English CTC 학습이 peaked attention 패턴을 형성했는데, Korean 학습은 이 패턴 없이 시작.

3. **언어적 요인**: Korean은 교착어로 형태소가 복잡하지만, 이것이 반드시 uniform attention을 요구하지는 않음.
   실험 증거: L8-11에 temperature=2 적용 시 96% 출력 유지 → **분산 attention은 부분적으로 중복 정보**.

### 2.3 per-layer 오류 축적

| 모델 | per-layer argmax error | 4 layers 후 | 8 layers 후 | 12 layers 후 |
|------|:---------------------:|:-----------:|:-----------:|:------------:|
| Korean wav2vec2 | **6.2%** | 77.4% | 59.8% | **46.3%** |
| English wav2vec2 | **1.3%** | 94.8% | 89.7% | **85.0%** |

→ **레이어 수가 양자화 생존에 결정적**. 4 layers면 Korean도 77.4% (borderline usable).

---

## 3. 해결 전략 (우선순위순)

### 전략 A: QAT Fine-tuning (최우선, 성공 확률 60%)

**개요**: English CTC 모델(uint8-proven)에서 출발, QAT로 uint8 호환성 유지하면서 한국어 학습.

**근거**:
- arXiv 2501.03643: wav2vec2-base QAT INT8 → WER 5.80% vs FP32 5.78% (**거의 무손실**)
- 동일 아키텍처에서 QAT가 작동한다는 실측 증거 존재
- English 모델은 이미 peaked attention (uint8-safe) → QAT가 이 패턴 유지하면서 Korean 학습

**구현**:
```python
# 1. English CTC 모델 로드 (uint8-proven 출발점)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 2. 한국어 CTC head 교체 (32→56 vocab)
model.lm_head = nn.Linear(768, 56)

# 3. Attention Q/K/V/out에 FakeQuantize 삽입
for layer in model.wav2vec2.encoder.layers:
    layer.attention.q_proj = FakeQuantWrapper(layer.attention.q_proj)
    layer.attention.k_proj = FakeQuantWrapper(layer.attention.k_proj)
    # ... v_proj, out_proj, FFN도 동일

# 4. Zeroth-Korean + 월패드 데이터로 QAT fine-tune (3-5 epochs)
# 5. FakeQuantize 제거 → clean FP32 ONNX export
# 6. Acuity uint8 → NB → T527 테스트
```

**리소스**:
- GPU: RTX 4070 Super 16GB (로컬 보유)
- 시간: ~2-4시간 (batch_size=8, 10 epochs)
- 데이터: Zeroth-Korean (51시간, 캐시 보유)

**위험**: QAT로도 Korean attention이 충분히 peaked되지 않을 가능성.

**완화**: Clipped Softmax 또는 Gated Attention 병용 (아래 참조).

---

### 전략 A+: Gated Attention + QAT (전략 A 실패 시)

**개요**: Attention에 학습 가능한 gate `σ(G(x)) ⊙ A(x)` 추가. "null operation" attention head가 0을 출력하도록 허용.

**근거**:
- Interspeech 2024 (arXiv 2406.11022): Whisper INT8에서 WER 9.7% → **7.7%** (gating 적용)
- Kurtosis (outlier 지표): 105.4 → **42.8**
- Korean L11의 1.8% uniform attention = 정확히 "null operation" 패턴

**구현**: `gated_att(x) = σ(Linear(x)) * Attention(x)` — Sigmoid gate 1개 추가.
ONNX export 시 sigmoid + multiply 노드 추가 → Acuity가 지원하는 표준 연산.

---

### 전략 A++: Clipped Softmax (전략 A와 병용 가능)

**개요**: `softmax(x)` → `(1-α) * softmax(x) + α` (α=-0.025). "완전한 0"을 제거.

**근거**: Qualcomm NeurIPS 2023 — attention outlier의 근본 원인은 softmax가 0에 도달하려고 입력을 극단적으로 증폭하는 것. Clipped softmax는 이를 원천 차단.

**장점**: 아키텍처 변경 최소 (softmax 함수 교체만), ONNX 호환, QAT와 조합 용이.

---

### 전략 B: 4-6 Layer 축소 모델 + QAT (성공 확률 50%)

**개요**: wav2vec2-base를 4-6 layer로 축소, QAT + Korean fine-tune.

**근거**:
- 4 layers → 77.4% argmax agreement (12 layers 46.3%보다 훨씬 양호)
- NB 크기: 38-52MB (현재 72MB보다 작음)
- 추론 시간: ~200-300ms (추정)

**구현**:
```python
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
config.num_hidden_layers = 6  # 12 → 6
config.vocab_size = 56
model = Wav2Vec2ForCTC(config)
# CNN 가중치를 base-960h에서 복사, Transformer L0-5도 복사
# QAT + Korean fine-tune
```

**위험**: Acuity가 비표준 layer 수의 그래프를 처리하지 못할 수 있음 (이전 pruned 모델 status=-1 경험).
**완화**: pruning이 아닌 처음부터 clean하게 생성 → 정상 ONNX 그래프.

---

### 전략 C: Zipformer Encoder 실기기 테스트 (즉시 실행 가능)

**개요**: 이미 변환 완료된 Zipformer encoder NB (63MB, uint8)를 T527에서 테스트.

**근거**:
- Zipformer은 Conv + Attention hybrid → Conv 비율이 높아 uint8에 유리할 수 있음
- KsponSpeech 969시간 학습 (Zeroth-Korean 51시간보다 19배 많은 데이터)
- NB 파일 이미 보유 (`zipformer/encoder/network_binary.nb`, 63MB)

**실행**: vpm_run으로 즉시 테스트 가능 (30분). 결과에 따라 decoder+joiner 파이프라인 구축.

**위험**: 32개 입력 텐서 관리 (캐시 31개 + 주 입력 1개)가 복잡.

---

### 전략 D: QuartzNet 15x5 한국어 학습 (uint8 보장, 성공 확률 90%)

**개요**: 100% CNN 아키텍처 (attention 0개). uint8 양자화 **보장**.

| 항목 | 값 |
|------|-----|
| 파라미터 | **18.9M** |
| 아키텍처 | 79개 1D Separable Conv 레이어, CTC |
| NB 크기 (추정) | **~19MB** |
| 추론 시간 (추정) | **< 50ms** |
| uint8 호환 | **보장** (KoCitrinet과 동일 계열) |

**근거**: CitriNet의 전신. 동일 아키텍처 계열인 KoCitrinet이 이미 T527 uint8에서 동작 확인.

**요구**: 한국어 음성 데이터로 처음부터 학습 필요 (NeMo 프레임워크).
KsponSpeech (969시간) 또는 Zeroth-Korean (51시간) + 월패드 데이터.

**기대 CER**: KoCitrinet과 유사하거나 더 좋을 수 있음 (아키텍처 차이는 SE block 유무).

---

### 전략 D2: Knowledge Distillation — Transformer Teacher → CNN Student (성공 확률 85%)

**개요**: 한국어 Transformer 모델(wav2vec2-base-korean, CER 9.5%)을 teacher로, CNN 모델(CitriNet/QuartzNet)을 student로 Knowledge Distillation.

**핵심 아이디어**: CNN student는 T527 uint8에서 **동작이 보장**됨 (KoCitrinet으로 검증). 문제는 학습 데이터/신호의 품질이지 아키텍처가 아님. Transformer teacher의 지식을 CNN에 전이하면 **양자화 호환성 + 높은 정확도** 동시 달성 가능.

**근거**:
- TutorNet (arXiv 2008.00671): cross-architecture KD (CNN Jasper ↔ RNN DeepSpeech2) 성공
- DistillW2V2 (arXiv 2303.09278): wav2vec2 KD로 12배 축소, WER +9-23% relative
- KoCitrinet CER 44.44%의 원인은 CNN 한계가 아니라 **학습 데이터/설정 부족** → KD로 해결 가능

**구현**:
```python
# Teacher: wav2vec2-base-korean (FP32, CER 9.5%)
# Student: CitriNet-256 또는 CitriNet-512 (NeMo)

# Loss = alpha * CTC_loss(student, labels)
#       + beta * KL_div(student_logits/T, teacher_logits/T)
#       + gamma * MSE(student_hidden, project(teacher_hidden))

# Data: Zeroth-Korean 51시간 (최소) / KsponSpeech 969시간 (최적)
# 학습: 1 GPU, 1-3일
# 양자화: Acuity uint8 직접 변환 (CNN = outlier 문제 없음)
```

**기대 CER**: 12-20% (51시간) / 9-15% (969시간) — 현재 44.44%에서 2-3배 개선.

**장점**:
- CNN student → uint8 **보장** (별도 QAT/Gated Attention 불필요)
- 기존 Acuity/Pegasus/Android 파이프라인 그대로 사용 가능
- NeMo CitriNet 학습 레시피 활용 가능

**단점**: Teacher 모델 추론으로 soft label 생성 필요 (GPU 시간 추가).

---

### 전략 D3: Activation Kurtosis Regularization + QAT (전략 A 강화)

**개요**: QAT에 kurtosis regularization 추가. 학습 중 activation 분포를 uint8-friendly하게 유도.

**근거**:
- KURE (NeurIPS 2020): weight kurtosis → 1.8 (uniform)으로 정규화, INT8 PTQ 성공
- arXiv 2404.03605: activation kurtosis 정규화로 W4A4 = W16A16 수준 달성
- ACosR (Amazon Interspeech 2020): 음성인식(RNN-T)에서 8-bit 무손실, 6-bit 무시할만한 열화

**구현**:
```python
# L_total = L_CTC + lambda_kurt * L_kurtosis
# L_kurtosis = mean(|Kurt(activation_i) - 1.8|^2)
# lambda_kurt = 1e-5 (activation용)

# Forward hook으로 각 encoder layer 출력의 kurtosis 측정
# Gradient가 흐르도록 detach() 하지 않음
```

**장점**: QAT와 결합 시 근본 원인(wide activation range) 직접 공격. 구현 간단 (hook 기반).

---

### 전략 E: KoCitrinet 월패드 Fine-tuning (Fallback, 성공 확률 90%)

**개요**: 현재 KoCitrinet (CER 44.44%)을 월패드 데이터로 추가 학습.

**장점**: 이미 uint8 동작 확인, 파이프라인 완비, 리스크 최소.
**단점**: CER 개선 한계 (44% → 30-35% 추정). 아키텍처 자체의 한계.

---

## 4. 불가능한 경로 (검증 완료)

| 접근법 | 이유 |
|--------|------|
| **Post-training 양자화 (PTQ)** | 60종+ 시도, 전부 실패. 근본적 불가 |
| **int16/bf16/fp16 NB** | T527 NPU가 uint8만 HW 가속 |
| **Whisper (모든 크기)** | RK3588, MediaTek 등 **모든 NPU에서 INT8 실패** 확인. Autoregressive decoder 비호환 |
| **300M+ 모델 (XLS-R 등)** | uint8 양자화 열화가 더 심함 |
| **SmoothQuant/AttnClip/RangeClip** | 10종+ 시도, 전부 실패 또는 악화 |

---

## 5. 추천 실행 순서

```
[즉시] 전략 C: Zipformer encoder vpm_run 테스트 (30분)
  ├── 성공 → Zipformer 풀 파이프라인 구축
  └── 실패 ↓

[1일차] 전략 A: QAT + Kurtosis Reg (전략 D3 결합)
  ├── PyTorch CUDA 설치
  ├── QAT + kurtosis regularization 학습 스크립트 작성
  ├── Zeroth-Korean 데이터 로드 (캐시 보유)
  └── QAT fine-tune 시작 (2-4시간)

[2일차] 전략 A 결과 확인
  ├── ONNX export → Acuity uint8 → T527 테스트
  ├── 성공 → CER 측정, 보고
  └── 실패 ↓

[3일차] 전략 A+/A++: Gated Attention 또는 Clipped Softmax 추가
  ├── 아키텍처 수정 + 재학습 (2-4시간)
  └── 재테스트

[병렬] 전략 D2: Knowledge Distillation (CNN student)
  ├── wav2vec2-base-korean으로 soft label 생성
  ├── NeMo CitriNet student 학습 (1-3일)
  └── Acuity uint8 변환 (기존 파이프라인)

[병렬] 전략 D: QuartzNet 학습 데이터 준비 (KsponSpeech 확보 가능 여부 확인)
```

---

## 6. GPU 리소스

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 4070 Super |
| VRAM | 16GB |
| QAT batch_size=8 | ~10.6 GB (가능) |
| 학습 시간 (10 epochs) | ~2시간 |
| PyTorch | 2.4.1 (CUDA 버전 설치 필요) |
| 데이터 | Zeroth-Korean 캐시 보유 (2.7GB) |

---

## 7. 성공 기준

| 지표 | 현재 (KoCitrinet) | 목표 | 달성 시 의미 |
|------|:-----------------:|:----:|------------|
| **CER** | 44.44% | **< 30%** | 의미 있는 개선 |
| **추론 시간** | 120ms | **< 1000ms** | 실시간 가능 |
| **NPU 동작** | O | **O** | uint8 HW 가속 |
| **NB 크기** | 62MB | **< 100MB** | T527 메모리 내 |

---

## 8. 핵심 참고 문헌

| 논문/자료 | 핵심 내용 |
|----------|----------|
| arXiv 2501.03643 | wav2vec2 QAT INT8 = FP32과 동일 성능 (WER +0.02%p) |
| arXiv 2406.11022 (Interspeech 2024) | Gated Attention → Whisper INT8 WER 9.7%→7.7% |
| Qualcomm NeurIPS 2023 | Clipped Softmax → outlier 근본 제거 |
| RK3588 issue #314 | Whisper INT8 모든 NPU에서 실패 확인 |
| arXiv 2511.08093 | Static PTQ = Whisper 실패, Dynamic만 성공 |
| arXiv 2008.00671 (TutorNet) | Cross-architecture KD (CNN↔RNN) for ASR |
| arXiv 2303.09278 (DistillW2V2) | wav2vec2 KD 12배 축소, WER +9-23% relative |
| NeurIPS 2020 (KURE) | Weight kurtosis→1.8 정규화 → INT8 PTQ 개선 |
| arXiv 2404.03605 | Activation kurtosis 정규화 → W4A4 = W16A16 |
| Amazon Interspeech 2020 (ACosR) | ASR QAT 8-bit 무손실, 6-bit 무시할만한 열화 |
