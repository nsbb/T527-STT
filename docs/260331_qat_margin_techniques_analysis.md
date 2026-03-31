# QAT Margin 기법 분석 및 성능 개선 전략

**날짜:** 2026-03-31
**맥락:** T527 NPU uint8 양자화, Conformer CTC, QAT 성능 개선 방법 탐색

---

## 1. 문제 정의

### 1.1 왜 margin이 중요한가

T527 NPU는 uint8(256칸)으로 모든 값을 표현함. 모델의 마지막 출력(logit)에서 1등(정답)과 2등의 점수 차이(margin)가 uint8 한 칸 크기(step)보다 작으면 같은 정수로 뭉개져서 argmax가 뒤집힘.

```
FP32:  "안" = -0.200,  "앉" = -0.205  → 차이 0.005 → 구분 가능
uint8: "안" = 118,      "앉" = 118     → 차이 0    → 구분 불가!
```

### 1.2 현재 상태

| 모델 | 원래 margin | uint8 step | ratio | CER |
|------|-----------|-----------|-------|-----|
| Conformer (PTQ) | 0.180 | 0.191 | 0.94x | 10.02% |
| Conformer (QAT 100k) | 더 넓어짐 | 0.191 | >1x | 5.3% |
| wav2vec2 한국어 | 0.005 | 0.050 | 0.10x | 100.86% |

### 1.3 margin을 결정하는 요인

```
아키텍처 → activation range → logit 분포 → margin

Conformer: CNN이 매 레이어마다 range 안정화 → margin 자연스럽게 넓음
wav2vec2:  순수 Transformer, range 누적 폭발 → margin 거의 0
```

**vocab 크기는 margin과 상관없음** (이전에 틀렸던 분석).
Conformer vocab 2049에서 성공, wav2vec2 vocab 56에서 실패.

---

## 2. Logit과 Vocab의 관계

### 2.1 기본 원리

모델은 **매 프레임마다** vocab 크기(2049개)의 logit 배열을 출력함.
각 칸의 index가 vocab에 대응. 값(logit)은 입력 음성마다 달라짐.

```
vocab:  [0:"가", 1:"나", ..., 523:"안", 524:"앉", ..., 2048:"blank"]

프레임1 logit: [-15.3, -18.7, ..., -0.2, -0.8, ..., -12.1]
                                     ^^^
                                  argmax → "안"

프레임2 logit: [-20.1, -11.5, ..., -8.3, -2.1, ..., -0.1]
                                                      ^^^
                                                   argmax → "blank"
```

### 2.2 CTC Decoding

```
프레임별 argmax: [안, 안, 안, 녕, 녕, blank, 하, 하, 세, 세, 요]
CTC decode:     → "안녕하세요" (중복 제거 + blank 제거)
```

음성 인식 = **프레임 단위 classification의 연속** + CTC decode.
영상 프레임마다 classification 하는 것과 원리 동일.

### 2.3 양자화 문제 발생 지점

uint8로 변환 시 logit 값이 정수로 뭉개짐.
margin이 step보다 작은 프레임에서 argmax 뒤집힘 → CER 상승.

---

## 3. Margin Loss 기법 종류

### 3.1 Hinge Margin (현재 사용 중)

```python
# train_qat.py의 MarginQATWrapper
margin_loss = ReLU(margin_target - (top1_logit - top2_logit)).mean()
loss = CTC_loss + margin_lambda * margin_loss
```

- top1-top2 차이가 margin_target 미만이면 penalty
- 직접적으로 logit 차이를 타겟
- 현재 설정: margin_target=0.3, margin_lambda=0.1

**장점:** 단순, 직관적, 우리 문제에 직접 대응
**단점:** margin_target 튜닝 필요, 너무 크면 CTC loss와 충돌

### 3.2 Triplet / Contrastive Loss

```python
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

- 임베딩 공간에서 같은 클래스는 가깝게, 다른 클래스는 멀게
- margin 직접 조절 가능

**적합성: 낮음**
- 임베딩 기반이라 logit 직접 제어와 맞지 않음
- CTC 출력에 적용하려면 큰 변형 필요

### 3.3 ArcFace / CosFace (Angular Margin)

```python
# CosFace
cosine = F.linear(F.normalize(x), F.normalize(W))
phi = cosine - m  # margin 삽입
logits = s * phi  # scale

# ArcFace
phi = torch.cos(torch.acos(cosine) + m)
logits = s * phi
```

- 분류 모델의 마지막 FC layer에 angular margin 삽입
- m (margin 크기)과 s (scale) 2개 파라미터
- 일반적으로 m ≈ 0.3~0.5, s ≈ 30~64

**적합성: 중간**
- 단일 분류용 설계인데, CTC는 프레임마다 분류라 적용 가능
- decoder(Linear) 레이어에 CosFace 적용 가능
- 하지만 CTC loss와의 호환성 검증 필요

### 3.4 Temperature Scaling

```python
# 추론 시 logit을 temperature로 나눔
scaled_logits = logits / T

# T < 1.0: 분포 샤프 (높은 값은 더 높게, 낮은 값은 더 낮게) → margin 증가
# T > 1.0: 분포 완만 → margin 감소
# T = 1.0: 변화 없음
```

**적합성: 높음**
- QAT 재학습 없이 추론 시 적용 가능
- logit을 T=0.5로 나누면 모든 차이가 2배로 커짐
- 단, log_softmax 이후에는 효과 없음. softmax 이전에 적용해야 함

**주의:** Conformer는 log_softmax 출력이라 직접 적용 어려울 수 있음. ONNX에서 log_softmax 전에 temperature division 삽입 필요.

### 3.5 Knowledge Distillation (KD)

```python
# FP32 모델 = teacher, QAT 모델 = student
teacher_logits = fp32_model(audio)  # FP32 출력 (정확)
student_logits = qat_model(audio)   # QAT 출력 (FakeQuantize 포함)

# KD loss: student가 teacher의 출력 분포를 따라가도록
kd_loss = KL_div(student_logits / T, teacher_logits / T)

# 최종 loss
loss = CTC_loss + alpha * kd_loss + beta * margin_loss
```

**적합성: 매우 높음**
- 학술적으로 QAT + KD 조합이 가장 효과 좋다고 입증됨
- FP32 모델의 "지식"을 유지하면서 양자화에 적응
- margin만 벌리면 CTC 정확도가 떨어질 수 있는데, KD가 정확도를 보존

**장점:** margin loss의 "CTC와 충돌" 문제 완화
**단점:** teacher 모델을 동시에 메모리에 올려야 함 (GPU 메모리 2배)

### 3.6 Label Smoothing

```python
# hard label: [0, 0, 1, 0, 0]
# soft label: [0.025, 0.025, 0.9, 0.025, 0.025]
```

**적합성: 낮음 (역효과)**
- label을 부드럽게 만들어서 logit 분포가 완만해짐
- margin이 좁아지는 방향 → 우리 목적과 반대

---

## 4. 기법 비교 및 추천

### 4.1 종합 비교

| 기법 | 적합성 | 재학습 필요 | 구현 난이도 | 기대 효과 |
|------|--------|-----------|-----------|----------|
| **Hinge Margin (현재)** | ★★★★ | O | 쉬움 (파라미터만 변경) | 검증됨 |
| **margin_target 올리기** | ★★★★ | O | 매우 쉬움 | 중간 |
| **Knowledge Distillation** | ★★★★★ | O | 중간 | 높음 |
| **Temperature Scaling** | ★★★ | X | 쉬움 (ONNX 수정) | 낮음~중간 |
| **CosFace/ArcFace** | ★★ | O | 어려움 | 불확실 |
| **Triplet Loss** | ★ | O | 어려움 | 불확실 |
| **Label Smoothing** | ☆ | O | 쉬움 | 역효과 |

### 4.2 추천 우선순위

**1순위: margin_target 올리기 (지금 바로 가능)**
```bash
# 현재
--margin-target 0.3 --margin-lambda 0.1

# 시도
--margin-target 0.5 --margin-lambda 0.1
--margin-target 1.0 --margin-lambda 0.1
--margin-target 0.5 --margin-lambda 0.3  # lambda도 올려보기
```

**2순위: Knowledge Distillation 추가 (코드 수정 필요)**
```python
# train_qat.py 수정
teacher_model = original_fp32_model  # freeze
student_model = qat_model

loss = ctc_loss + 0.1 * margin_loss + 0.5 * kd_loss
```

**3순위: Temperature Scaling (ONNX 수정)**
```python
# ONNX export 후 log_softmax 전에 temperature division 삽입
# T=0.5로 하면 logit 차이 2배
```

---

## 5. Margin과 CTC의 충돌 문제

### 5.1 왜 margin을 무한정 넓힐 수 없는가

```
margin_target이 너무 크면:
  모델: "top1-top2 차이를 5.0 이상으로 만들어야 해!"
  → 가중치를 극단적으로 바꿈
  → 원래 잘 맞추던 음성도 틀리기 시작
  → CTC loss ↑ (정확도 ↓)
  → CTC loss와 MarginLoss가 싸움
  
최적점: CTC 정확도를 유지하면서 margin이 최대인 지점
```

### 5.2 wav2vec2에서 QAT가 안 되는 이유

```
Conformer:
  원래 margin 0.18 → 0.3까지 벌리면 됨 (1.7배)
  → 가중치 살짝 조정 → CTC 정확도 유지

wav2vec2 한국어:
  원래 margin 0.005 → 0.3까지 벌려야 함 (60배!)
  → 가중치 대폭 변경 필요 → CTC 정확도 붕괴
  → margin 벌리면 정확도 떨어지고, 정확도 올리면 margin 좁아짐
  → 해결 불가 → 아키텍처를 바꿔야 함
```

### 5.3 Knowledge Distillation이 이 문제를 완화하는 이유

```
loss = CTC + margin + KD

KD가 하는 역할:
  "FP32 모델이 이렇게 출력했으니까 너도 비슷하게 해"
  → CTC 정확도를 FP32 수준으로 유지시켜줌
  → margin loss가 좀 세게 눌러도 정확도가 덜 떨어짐
  → 결과적으로 더 넓은 margin을 달성 가능
```

---

## 6. QAT 외 양자화 성능 개선 방법

### 6.1 T527 NPU 제약

| 타입 | T527 지원 | 비고 |
|------|----------|------|
| **uint8** | **HW 가속** | 유일하게 실용적 |
| int16 DFP | 부분 지원 | 2.4배 느림, NB 2배 |
| fp16 | SW 에뮬레이션 | 42배 느림 |
| bf16 | export 실패 | 사용 불가 |
| hybrid (uint8+int16) | 디바이스 크래시 | 사용 불가 |

**uint8 범위 안에서만 개선 가능.**

### 6.2 QAT 재학습 없이 가능한 것

| 방법 | 설명 | 기대 효과 |
|------|------|----------|
| Per-channel quantization | `perchannel_symmetric_affine`으로 변경 | 중간 |
| KL divergence + iterations 500~1000 | calibration 정밀도 향상 | 낮음 |
| auto 알고리즘 | Acuity 자동 최적화 | 낮음 |
| Temperature Scaling | ONNX에서 logit/T 삽입 | 낮음~중간 |
| 커스텀 .quantize | 레이어별 min/max 직접 최적화 | 중간 |

### 6.3 QAT 재학습 필요한 것

| 방법 | 설명 | 기대 효과 |
|------|------|----------|
| margin_target 올리기 | 0.3 → 0.5~1.0 | 중간 |
| Knowledge Distillation 추가 | FP32 teacher + QAT student | 높음 |
| FakeQuantize 위치 늘리기 | 3곳 → 전 레이어 | 중간 |
| 2단계 학습 | AIHub QAT → 자체 데이터 fine-tune | 중간 |

### 6.4 val_loss ≠ T527 CER

QAT에서 val_loss가 가장 낮은 checkpoint가 T527에서 가장 좋은 CER을 보이지 않음.

```
AIHub 100k QAT:
  ep04 val_loss 0.1416 (best) → T527 CER 100% (실패)
  ep09 val_loss 0.151 (worse) → T527 CER 5.3% (성공)
```

**원인:** FakeQuantize는 이상적 uint8 시뮬레이션이고, 실제 NPU는 다르게 동작함 (시뮬레이션-디바이스 일치율 31.5%).

**결론:** 여러 epoch의 모델을 T527에서 직접 테스트해야 함.

---

## 7. 현재 QAT 실험 결과 종합

| # | 실험 | 데이터 | 시간 | Epoch | T527 CER |
|---|------|--------|------|-------|----------|
| 0 | PTQ baseline | - | - | - | 10.02% |
| 1 | AIHub 100k final | 100k개, 84hr | 2hr 학습 | 10 | **5.3%** |
| 2 | 자체데이터 best | 294개, 18min | 3min 학습 | 13 | **~6%** |
| 3 | AIHub 전체 best | 4.09M개, 4356hr | 42hr 학습 | 9 (val 0.0692) | **테스트 필요** |
| 4 | AIHub 전체 final | 4.09M개, 4356hr | 42hr 학습 | 10 | **테스트 필요** |

---

## 8. 다음 시도 우선순위

1. **AIHub 전체 QAT T527 테스트** — ep08, ep09 모델
2. **margin_target 올린 QAT** — 0.5, 1.0으로 재학습
3. **Knowledge Distillation + QAT** — train_qat.py에 KD loss 추가
4. **Per-channel quantization** — Acuity 옵션 변경만으로 가능
5. **2단계 학습** — AIHub QAT 모델 → 자체 데이터 fine-tune
6. **커스텀 .quantize 파이프라인** — 로드맵 Step 3
