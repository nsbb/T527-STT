# QAT 기술 심화 토론 기록

**날짜:** 2026-03-31
**맥락:** QAT+KD 학습 진행 중 발생한 기술 토론

---

## 1. Logit과 Vocab의 관계

### 1.1 기본 원리

모델은 **매 프레임마다** vocab 크기(2049개)의 logit 배열을 출력.
각 index가 vocab에 대응. 값(logit)은 입력 음성마다 달라짐.

```
vocab:  [0:"가", 1:"나", ..., 523:"안", 524:"앉", ..., 2048:"blank"]
logit:  [-15.3,  -18.7, ..., -0.2,    -0.8,    ..., -12.1  ]
                               ^^^ argmax → "안"
```

### 1.2 음성 인식 = 프레임 단위 Classification

```
이미지 분류:  1장 → [고양이 0.9, 개 0.1] → "고양이"
음성 인식:    76프레임 × [2049개 logit] → 프레임마다 분류 → CTC decode

프레임별 argmax: [안, 안, 안, 녕, 녕, blank, 하, 하, 세, 세, 요]
CTC decode:     → "안녕하세요" (중복 제거 + blank 제거)
```

영상의 연속된 프레임을 classification 하는 것과 동일.
중복 프레임 = "아직 같은 글자", blank = "글자 사이 경계".

### 1.3 양자화 문제 = Classification 오류

```
FP32:  "안" logit = -0.200,  "앉" logit = -0.205
       차이 0.005 → 구분 가능 → argmax = "안" ✓

uint8: "안" = 118,  "앉" = 118 (같은 정수로 뭉개짐)
       차이 0 → 구분 불가 → argmax 뒤집힐 수 있음 ✗
```

---

## 2. Margin과 아키텍처의 관계

### 2.1 이전 분석의 오류와 교정

```
틀렸던 논리:
  vocab 많음 → logit margin 좁음 → uint8 실패

실제:
  wav2vec2 (vocab 56):    margin 0.005 → 실패
  Conformer (vocab 2049): margin 0.180 → 성공
  → vocab 크기는 margin과 무관
```

### 2.2 왜 Conformer는 margin이 넓은가

```
Conformer (1개 레이어):
  [FFN] → [Attention] → [Conv] → [FFN]
                          ^^^
                    range를 리셋 (local averaging)

wav2vec2 (1개 레이어):
  [Attention] → [FFN] → [Attention] → [FFN]
  → range가 계속 누적 → logit이 뭉개짐 → margin 거의 0
```

CNN이 매 레이어마다 activation range를 안정화 → 최종 logit이 깔끔 → margin 자연스럽게 넓음.

### 2.3 wav2vec2에 MarginLoss를 적용하면?

이론적으로 가능하지만 한계가 큼:

```
Conformer: margin 0.18 → 0.3 (1.7배 벌리기) → 쉬움
wav2vec2:  margin 0.005 → 0.3 (60배 벌리기) → 불가능에 가까움
  → 가중치를 극단적으로 바꿔야 함
  → CTC 정확도 자체가 붕괴
  → MarginLoss와 CTC Loss가 충돌
```

---

## 3. FakeQuantize 상세

### 3.1 FakeQuantize란

학습 중에 uint8 양자화를 **흉내내는** 가짜 양자화.

```
진짜 양자화 (PTQ):
  float → uint8 정수 → 끝 (gradient 계산 불가)

FakeQuantize (QAT):
  float → uint8로 변환 → 다시 float로 복원 → 학습 계속
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         양자화 오차가 발생하지만 float 상태 유지
         → gradient 계산 가능 → 역전파 → 가중치 업데이트
```

```python
# 실제 동작
x = 3.141592              # 원래 float
scale = (max - min) / 255  # scale 계산
q = round(x / scale) + zp  # uint8로 변환: 157
x_fake = (q - zp) * scale  # 다시 float로: 3.14
# 오차 0.001592 발생, 이 오차 포함한 채로 학습 계속
```

### 3.2 QAT에서의 FakeQuantize 위치 (현재)

```
음성 입력
  ↓
[Preprocessor] → mel spectrogram
  ↓
★ FakeQuantize 1 (입력 양자화 시뮬레이션)
  ↓
[Encoder] (18 Conformer layers)
  ↓
★ FakeQuantize 2 (중간 feature 양자화 시뮬레이션)
  ↓
[Decoder] → logits (2049개)
  ↓
★ FakeQuantize 3 (출력 양자화 시뮬레이션)
  ↓
CTC Loss 계산
```

### 3.3 QAT 1 step의 흐름

일반 fine-tuning과 횟수는 동일. FakeQuantize가 중간에 끼어있을 뿐.

```
일반 fine-tuning 1 step:
  입력 → [모델] → 출력 → loss → 역전파 → 가중치 업데이트

QAT 1 step:
  입력 → [모델앞] → FQ → [모델중간] → FQ → [모델뒤] → FQ → 출력 → loss → 역전파 → 가중치 업데이트
```

scale/zp는 매 forward마다 해당 텐서의 min/max에서 자동 계산. 별도로 찾는 과정 아님.

---

## 4. QAT와 PTQ의 관계

### 4.1 왜 QAT 후에 PTQ를 다시 하는가

QAT와 PTQ는 **역할이 다름:**

```
QAT:  "양자화돼도 안 깨지는 가중치"를 만듦
      → 출력: 강건한 가중치 (.nemo)
      → scale/zp는 시뮬레이션용 임시값, 저장 안 함

PTQ:  "이 가중치를 실제 uint8 정수로 변환하는 최적 매핑"을 찾음
      → 출력: scale/zp (.quantize) + uint8 NB 파일
```

```
파이프라인:
  QAT 학습 → .nemo (강건한 가중치)
                ↓
  ONNX export
                ↓
  Acuity PTQ → .quantize (scale/zp) + .nb (배포용)
```

### 4.2 QAT의 scale/zp를 PTQ에 넘기면 더 좋지 않냐?

논리적으로 유효한 질문. 문제는:

**1) QAT scale/zp는 동적:**
```
step 1: 데이터 A → scale=0.033
step 2: 데이터 B → scale=0.029
step 3: 데이터 C → scale=0.034
→ 어떤 값을 넘겨줌? 고정값이 아님
```

**2) QAT와 PTQ의 양자화 방식이 다름:**
```
QAT FakeQuantize: per-tensor, 동적 min/max
Acuity PTQ:       per-channel 가능, KL divergence 기반, 정적
→ 방식이 달라서 직접 대체 불가
```

**3) Acuity가 외부 scale을 직접 받지 않음:**
- .quantize 파일에 min/max를 쓸 수는 있음
- 하지만 Acuity가 그걸로 자기 방식으로 scale/zp 재계산

### 4.3 이 불일치가 문제를 일으킴

QAT의 FakeQuantize ≠ Acuity PTQ ≠ 실제 T527 NPU

```
시뮬레이션-디바이스 일치율: 31.5%
→ QAT가 시뮬레이션한 양자화 ≠ 실제 디바이스 양자화
→ val_loss best ≠ T527 CER best (실제 확인됨)
```

### 4.4 해결 방향: LSQ (Learned Step Size Quantization)

```
일반 QAT:  scale = (max - min) / 255  (데이터에서 계산, 매번 변함)
LSQ:       scale = 학습 가능한 파라미터  (gradient descent로 최적화)
```

- scale 자체를 가중치처럼 학습 → 최적 scale 수렴
- 학습된 scale을 배포에도 사용 → QAT-PTQ 불일치 해소
- .quantize 파일에 LSQ로 찾은 min/max를 써넣으면 Acuity에서도 활용 가능
- 논문: Esser et al. 2019 "Learned Step Size Quantization"

---

## 5. Knowledge Distillation + QAT

### 5.1 구조

```
Teacher: FP32 원본 모델 (frozen, 학습 안 함)
Student: QAT 모델 (FakeQuantize 삽입, 학습)

loss = CTC_loss + 0.1 × MarginLoss + 0.5 × KD_Loss
       ↑ 정확도     ↑ 양자화 강건성      ↑ FP32 지식 보존
```

### 5.2 KD Loss란

Teacher와 Student의 출력 분포 차이를 KL divergence로 측정.

```python
T = 2.0  # temperature
student_soft = log_softmax(student_logits / T)
teacher_soft = softmax(teacher_logits / T)
kd_loss = KL_div(student_soft, teacher_soft) * T²
```

- Temperature로 분포를 부드럽게 만들어서 "어두운 지식"(dark knowledge)도 전달
- 예: teacher가 "안"에 0.9, "앉"에 0.05를 줬다면 "앉이 안과 비슷하다"는 정보도 학습

### 5.3 KD가 QAT를 개선하는 이유

```
QAT만 할 때:
  MarginLoss: "logit 차이 벌려!"
  CTC Loss:   "정답 맞춰!"
  → 두 목표가 충돌 가능

QAT + KD:
  MarginLoss: "logit 차이 벌려!"
  CTC Loss:   "정답 맞춰!"
  KD Loss:    "FP32처럼 해!" → CTC 정확도 보존 역할
  → MarginLoss가 세게 눌러도 KD가 정확도를 잡아줌
  → 더 넓은 margin 달성 가능
```

### 5.4 KD Loss의 KL divergence vs PTQ의 KL divergence

같은 수학 공식이지만 용도가 다름:

```
KD의 KL divergence:
  → teacher 출력 분포 vs student 출력 분포 비교
  → student가 teacher를 따라가도록 학습하는 loss

PTQ calibration의 KL divergence:
  → FP32 activation 분포 vs uint8 activation 분포 비교
  → 최적의 scale/zp(clipping point)를 찾는 용도
```

### 5.5 현재 실험 상태

```
실험: QAT + KD, AIHub 100k, GPU 1개
  - margin_target: 0.5 (이전 0.3에서 올림)
  - kd_lambda: 0.5
  - kd_temperature: 2.0
  - SLURM Job: 6639, gpu-114
  - 예상 소요: ~4.5시간
```

---

## 6. Margin 기법 비교

### 6.1 현재 방식 (Hinge Margin)

```python
margin_loss = ReLU(margin_target - (top1 - top2)).mean()
```

top1-top2 차이가 target 미만이면 penalty. 단순하고 직접적.

### 6.2 ArcFace / CosFace (Angular Margin)

```python
# CosFace: logit에서 margin만큼 깎음
logits = s * (cosine - m)

# ArcFace: 각도에 margin 추가
logits = s * cos(arccos(cosine) + m)
```

분류 모델용 설계. CTC에 적용 가능하지만 복잡.

### 6.3 Temperature Scaling

```python
scaled_logits = logits / T  # T < 1.0이면 분포 샤프 → margin 증가
```

QAT 재학습 없이 추론 시 적용 가능.

### 6.4 적합성 순위

1. **Hinge Margin + margin_target 올리기** — 가장 쉽고 검증됨
2. **Knowledge Distillation** — 학술적으로 가장 효과적
3. **LSQ** — QAT-PTQ 불일치 해소, 근본적 해결
4. **Temperature Scaling** — 재학습 없이 가능, 효과 제한적
5. **ArcFace/CosFace** — CTC 호환성 불확실

---

## 7. QAT 용어 정리

| 용어 | 설명 |
|------|------|
| QAT | Quantization-Aware Training. FakeQuantize 넣고 학습 |
| PTQ | Post-Training Quantization. 학습 후 양자화 (Acuity) |
| FakeQuantize | 학습 중 양자화를 흉내냄 (float→uint8→float) |
| STE | Straight-Through Estimator. FakeQuantize의 역전파 방법 |
| MarginLoss | top1-top2 logit 차이를 강제로 벌리는 loss |
| KD | Knowledge Distillation. teacher-student 학습 |
| LSQ | Learned Step Size Quantization. scale을 학습 |
| scale | float 범위를 uint8 범위로 매핑하는 비율 |
| zero_point | float 0.0이 uint8 몇에 매핑되는지 |
| calibration | PTQ에서 scale/zp를 찾기 위해 데이터를 넣어보는 과정 |

---

## 8. 핵심 인사이트 요약

1. **QAT ⊂ fine-tuning** — QAT는 fine-tuning의 한 종류. FakeQuantize가 있으면 QAT.
2. **QAT는 scale/zp를 찾는 게 아니라 강건한 가중치를 만드는 것.**
3. **QAT의 scale/zp ≠ PTQ의 scale/zp** — 이 불일치가 시뮬레이션-디바이스 갭의 원인.
4. **LSQ로 이 불일치를 해소할 수 있음** — .quantize 파일 편집으로 실현 가능.
5. **KD는 MarginLoss의 "CTC 충돌" 문제를 완화** — margin을 더 넓힐 수 있게 해줌.
6. **Logit margin은 양자화 성패의 직접 원인이지만, margin을 결정하는 건 아키텍처.**
