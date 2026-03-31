# QAT, KD, CTC, Margin Loss 상세 설명

**날짜:** 2026-03-31
**목적:** QAT 학습 파이프라인의 모든 구성 요소를 최대한 자세히 설명

---

## 1. CTC (Connectionist Temporal Classification)

### 1.1 CTC가 필요한 이유

음성 인식의 근본 문제: **프레임별 정답이 없다.**

```
음성 "안녕하세요" (3초, 76프레임)
정답 "안녕하세요" (5글자)

문제: 76프레임 중 어디가 "안"이고 어디가 "녕"인지 모름
     사람마다, 발화마다 타이밍이 다름
     이걸 수동으로 표시(alignment)하는 건 수백만 데이터에 불가능
```

### 1.2 CTC가 해결하는 방법

**"프레임별 정답 없이도 학습 가능"**

```
일반 classification:
  프레임 1 → 정답: "안" (수동 지정 필요)
  프레임 2 → 정답: "안" (수동 지정 필요)
  프레임 3 → 정답: "녕" (수동 지정 필요)
  ...
  → 프레임마다 개별 정답 필요 → 현실적으로 불가능

CTC:
  76프레임 전체 → 정답: "안녕하세요"
  → 텍스트 하나만 있으면 학습 가능!
```

### 1.3 CTC Loss 동작 원리

```
모델 출력 (76프레임 × 2049 vocab):
  프레임1: [가:-15, 나:-18, ..., 안:-0.2, 앉:-0.8, ..., blank:-12]
  프레임2: [가:-16, 나:-17, ..., 안:-0.1, 앉:-0.9, ..., blank:-11]
  ...
  프레임76: [가:-14, 나:-19, ..., 요:-0.3, 용:-0.7, ..., blank:-10]

CTC가 하는 일:
  1. 정답 "안녕하세요"가 되는 모든 가능한 경로를 찾음:
     - [안,안,녕,blank,하,세,요,blank,...] → "안녕하세요" ✓
     - [안,녕,녕,하,하,세,요,...] → "안녕하세요" ✓
     - [blank,안,녕,하,세,요,요,...] → "안녕하세요" ✓
     - ... (수천~수만 개의 유효 경로)
  
  2. 모든 유효 경로의 확률을 합산
  
  3. 이 합산 확률이 최대가 되도록 loss 계산
     → 역전파 → 가중치 업데이트
```

### 1.4 CTC Loss vs CTC Decode

```
CTC Loss (학습용):
  "어떤 타이밍이든 최종 결과가 '안녕하세요'가 되는 확률을 높여라"
  → 모든 유효 경로의 확률 합산 → loss 계산
  → 역전파 → 가중치 업데이트

CTC Decode (추론용):
  모델 출력에서 텍스트를 추출하는 후처리
  
  Step 1: 프레임별 argmax
    [안, 안, 안, 녕, 녕, blank, 하, 하, 세, 세, 요, 요]
  
  Step 2: 연속 중복 제거
    [안, 녕, blank, 하, 세, 요]
  
  Step 3: blank 제거
    [안, 녕, 하, 세, 요]
  
  결과: "안녕하세요"
```

이름만 같은 CTC. loss는 학습용, decode는 추론용.

### 1.5 CTC 덕분에 데이터가 간단함

```json
{"audio_filepath": "file.wav", "text": "안녕하세요", "duration": 3.5}
```

이게 끝. 프레임별 alignment 불필요. 이래서 대량 음성 데이터 학습이 가능한 거.

---

## 2. FakeQuantize (QAT의 핵심)

### 2.1 FakeQuantize란

학습 중에 uint8 양자화를 **흉내내는** 가짜 양자화.
진짜 uint8로 바꾸면 gradient 계산이 안 돼서 학습 불가능. 
그래서 **float 상태를 유지하되 양자화 오차를 포함**시킴.

```
진짜 양자화 (PTQ):
  float 3.141592 → uint8 정수 157 → 끝 (gradient 못 구함)

FakeQuantize (QAT):
  float 3.141592
    → uint8로 변환: round(3.141592 / 0.02) + 128 = 285 → clamp → 255
    → 다시 float로 복원: (255 - 128) * 0.02 = 2.54
    → 오차 0.60 발생, 이 오차 포함한 채로 학습 계속
    → gradient 계산 가능!
```

### 2.2 FakeQuantize 코드 상세

```python
class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmin, qmax = 0, 255                    # uint8 범위
        x_min = x.min()                         # 이 텐서의 최솟값
        x_max = x.max()                         # 이 텐서의 최댓값
        scale = (x_max - x_min) / 255           # 한 칸의 크기
        scale = torch.clamp(scale, min=1e-8)    # 0 방지
        zero_point = 0 - round(x_min / scale)   # 0.0이 uint8 몇에 매핑되는지
        zero_point = clamp(zero_point, 0, 255)
        
        # 양자화: float → uint8 (정수)
        x_q = clamp(round(x / scale + zero_point), 0, 255)
        
        # 역양자화: uint8 → float (오차 포함)
        x_dq = (x_q - zero_point) * scale
        
        return x_dq  # float인데 양자화 오차가 포함된 값

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # STE: gradient 그대로 통과
```

### 2.3 STE (Straight-Through Estimator)

```
문제: round() 함수는 미분 불가능
     round(3.7) = 4, round(3.2) = 3
     → gradient가 거의 모든 곳에서 0

해결: STE — backward에서 round를 무시하고 gradient를 그대로 통과시킴

Forward: x → round(x) → 계단 함수 (미분 불가)
Backward: grad → 그대로 → 직선 (미분 가능한 척)

수학적으로 정확하지는 않지만, 실제로 잘 작동함.
```

### 2.4 FakeQuantize 삽입 위치

현재 train_qat.py에서 3곳에 삽입:

```
음성 입력
  ↓
[Preprocessor] → mel spectrogram (80 bins × 76 frames)
  ↓
★ FakeQuantize 1 — 입력 양자화 시뮬레이션
  ↓                 (mel 값이 uint8로 잘릴 때의 오차)
[Encoder] (18 Conformer layers)
  ↓
★ FakeQuantize 2 — 중간 feature 양자화 시뮬레이션
  ↓                 (encoder 출력이 uint8로 잘릴 때의 오차)
[Decoder] → logits (2049개)
  ↓
★ FakeQuantize 3 — 출력 양자화 시뮬레이션
  ↓                 (logit이 uint8로 잘릴 때의 오차)
CTC Loss 계산 → 역전파 → 가중치 업데이트
```

실제 T527 NPU에서는 **모든 레이어의 입출력이 uint8**이지만,
QAT에서는 3곳만 시뮬레이션. 이것이 시뮬레이션-디바이스 불일치(31.5%)의 원인 중 하나.

### 2.5 FakeQuantize에서 scale/zp는 어떻게 되는가

```
매 forward마다 해당 텐서의 min/max에서 새로 계산:

Step 1 (배치 A): mel 값 범위 [-1.6, 4.5] → scale=0.024
Step 2 (배치 B): mel 값 범위 [-1.3, 4.2] → scale=0.022
Step 3 (배치 C): mel 값 범위 [-1.8, 4.7] → scale=0.025
...

→ scale/zp가 매번 다름
→ 고정된 값이 아님
→ "대략 이 정도 양자화 노이즈가 생긴다"를 시뮬레이션하는 것
→ 특정 scale/zp에 최적화하는 게 아님
```

---

## 3. QAT vs PTQ: 왜 분리되어 있는가

### 3.1 역할이 다름

```
QAT의 목적:  "양자화돼도 안 깨지는 가중치"를 만듦
             → 출력: .nemo (강건한 가중치)
             → scale/zp는 시뮬레이션용, 저장 안 함

PTQ의 목적:  "이 가중치를 실제 uint8 정수로 변환하는 최적 매핑"을 찾음
             → 출력: .quantize (scale/zp) + .nb (NPU 바이너리)
```

### 3.2 파이프라인

```
QAT 학습
  원본 .nemo → FakeQuantize 넣고 학습 → 강건한 .nemo 저장
  (scale/zp는 버려짐)
      ↓
ONNX export
  .nemo → .onnx
      ↓
Acuity PTQ
  .onnx → calibration 데이터로 scale/zp 새로 계산 → .quantize
  (KL divergence 등으로 최적 scale/zp 결정)
      ↓
NB export
  .onnx + .quantize → .nb (T527 NPU용 바이너리)
```

### 3.3 "QAT에서 찾은 scale/zp를 PTQ에 넘기면 더 좋지 않냐?"

논리적으로 유효한 질문. 하지만:

**1) QAT scale/zp는 동적:**
```
매 배치마다 다른 scale → 어떤 값을 넘겨줌?
```

**2) QAT와 PTQ의 양자화 방식이 다름:**
```
QAT FakeQuantize: per-tensor, 동적 min/max, 단순 round
Acuity PTQ:       per-channel 가능, KL divergence 기반, 정적 최적화
```

**3) 이 불일치가 성능 갭의 원인:**
```
QAT 시뮬레이션 ≠ Acuity PTQ ≠ 실제 T527 NPU
→ 시뮬레이션-디바이스 일치율 31.5%
→ val_loss best ≠ T527 CER best (실제 확인됨)
```

**4) 해결 방향: LSQ (Learned Step Size Quantization)**
```
일반 QAT:  scale = (max-min)/255  (매번 새로 계산)
LSQ:       scale = 학습 가능한 파라미터 (gradient descent로 최적화)

→ 학습된 scale을 .quantize 파일에 써넣으면 QAT-PTQ 불일치 해소
→ 논문: Esser et al. 2019
```

---

## 4. Margin Loss 상세

### 4.1 왜 Margin Loss가 필요한가

CTC loss만으로는 **argmax만 맞으면 됨.** 차이가 0.001이든 10이든 상관없음.

```
CTC loss 관점에서 둘 다 OK:
  "안" = -0.200, "앉" = -0.201  (차이 0.001)  ← 위험!
  "안" = -0.200, "앉" = -5.000  (차이 4.800)  ← 안전

근데 uint8로 양자화하면:
  차이 0.001 → 같은 정수 → argmax 뒤집힘!
  차이 4.800 → 다른 정수 → argmax 유지
```

Margin Loss: "차이를 최소 N 이상으로 유지해라"

### 4.2 Hinge Margin Loss (현재 사용 중)

```python
# top1-top2 logit 차이 계산
sorted_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
margins = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]  # [B, T]

# margin이 target 미만이면 penalty
margin_loss = ReLU(margin_target - margins).mean()

# 전체 loss에 추가
loss = ctc_loss + margin_lambda * margin_loss
```

### 4.3 Margin Loss 동작 예시

```
margin_target = 0.5 일 때:

프레임 A: top1=-0.2, top2=-0.8, margin=0.6
  → 0.6 > 0.5 → ReLU(0.5-0.6) = ReLU(-0.1) = 0 → penalty 없음 ✓

프레임 B: top1=-0.2, top2=-0.3, margin=0.1
  → 0.1 < 0.5 → ReLU(0.5-0.1) = ReLU(0.4) = 0.4 → penalty 있음!
  → 역전파: "이 프레임의 차이를 벌려라" → 가중치 조정

프레임 C: top1=-0.2, top2=-0.205, margin=0.005
  → 0.005 < 0.5 → ReLU(0.5-0.005) = 0.495 → 큰 penalty!
  → 역전파: "이 프레임 위험! 강하게 조정해라"
```

### 4.4 margin_target 값의 의미

```
uint8 step size ≈ 0.2 (Conformer logit 범위 기준)

margin_target = 0.3 → uint8 step의 1.5배 여유 (보수적)
margin_target = 0.5 → uint8 step의 2.5배 여유 (적극적)
margin_target = 1.0 → uint8 step의 5배 여유 (공격적)
```

### 4.5 margin_target이 너무 크면?

```
margin_target = 10.0 (극단적)
  → 모든 프레임에서 top1-top2 차이가 10 이상이어야 함
  → 가중치를 극단적으로 바꿔야 함
  → CTC loss: "정답 맞춰!" vs MarginLoss: "차이 벌려!"
  → 두 목표가 충돌 → 학습 불안정 → 정확도 붕괴
```

### 4.6 margin_lambda의 역할

```
loss = CTC + margin_lambda × MarginLoss

margin_lambda = 0.1:  CTC가 10배 중요 → 정확도 우선
margin_lambda = 0.5:  CTC가 2배 중요 → 균형
margin_lambda = 1.0:  CTC = MarginLoss → margin에 많은 비중
```

---

## 5. Knowledge Distillation (KD)

### 5.1 KD 구조

```
같은 음성 입력
     ↓
┌──────────────────┐      ┌──────────────────┐
│  Teacher (FP32)   │      │  Student (QAT)    │
│  원본 모델        │      │  FakeQuantize 有  │
│  가중치 고정      │      │  가중치 학습 중   │
│  gradient 안 흐름 │      │  gradient 흐름    │
└──────────────────┘      └──────────────────┘
     ↓                          ↓
Teacher 출력                Student 출력
(FP32, 정확함)              (QAT, 양자화 오차 포함)
     ↓                          ↓
     └────────── 비교 ──────────┘
                  ↓
            KD Loss 계산
```

### 5.2 KD가 전달하는 것: "Dark Knowledge"

```
Teacher 출력 (정답이 "안"인 프레임):
  "안" = -0.2   (1등, 확률 높음)
  "앉" = -15.3  (2등)
  "않" = -16.1  (3등)
  "녕" = -18.7  (멀리)
  "가" = -25.0  (아주 멀리)

이 분포가 담고 있는 정보:
  - "안"이 정답이다 (CTC도 이건 알려줌)
  - "앉"이 "안"과 가장 비슷하다 (이건 CTC가 안 알려줌!)
  - "않"도 비슷한 부류다
  - "녕"은 좀 다르다
  - "가"는 완전 다르다

→ 이런 "누가 누구와 비슷한지" 정보 = Dark Knowledge
→ Student가 이걸 배우면 더 정교한 표현 가능
```

### 5.3 KD Loss 계산

```python
T = 2.0  # temperature

# Temperature로 분포를 부드럽게 (dark knowledge가 드러남)
student_soft = F.log_softmax(student_logits / T, dim=-1)
teacher_soft = F.softmax(teacher_logits / T, dim=-1)

# KL divergence로 두 분포 차이 측정
kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
```

### 5.4 Temperature의 역할

```
T = 1.0 (날카로운 분포):
  "안" = 0.95, "앉" = 0.03, "않" = 0.01, ... → 거의 one-hot
  → "안이 정답이다"만 전달, dark knowledge 없음

T = 2.0 (부드러운 분포):
  "안" = 0.40, "앉" = 0.15, "않" = 0.12, "녕" = 0.08, ...
  → "안이 정답인데, 앉/않도 비슷하고, 녕은 좀 다르다" 전달
  → dark knowledge가 드러남

T = 10.0 (매우 부드러운 분포):
  "안" = 0.12, "앉" = 0.11, "않" = 0.10, ... → 거의 균일
  → 정보가 너무 흐릿해짐
```

### 5.5 KD가 QAT를 개선하는 원리

```
QAT만 (MarginLoss + CTC):
  MarginLoss: "차이 벌려!"
  CTC:        "정답 맞춰!"
  → 두 목표가 충돌 가능
  → margin 벌리다가 CTC 정확도 하락

QAT + KD (MarginLoss + CTC + KD):
  MarginLoss: "차이 벌려!"
  CTC:        "정답 맞춰!"
  KD:         "Teacher 분포 따라해!" → 정확도 보존 보조
  → MarginLoss가 세게 당겨도 KD가 정확도를 잡아줌
  → 더 넓은 margin 달성 가능
```

### 5.6 세 가지 Loss의 상호작용

```
loss = CTC + 0.1 × MarginLoss + 0.5 × KD

각 Loss가 당기는 방향:

CTC:        "프레임별 argmax가 정답이면 됨" → ↑
MarginLoss: "top1-top2 차이를 벌려라"      → ← →
KD:         "Teacher처럼 출력해라"          → → (원래 방향 유지)

세 힘의 균형 → 최적점
```

### 5.7 Lambda(가중치)로 균형 조절

```
kd_lambda=0.5, margin_lambda=0.1
→ KD가 5배 더 강함 → "Teacher 따라가기" 우선 → 정확도 보존 강조

kd_lambda=0.1, margin_lambda=0.5
→ Margin이 5배 더 강함 → "차이 벌리기" 우선 → 양자화 강건성 강조
```

완벽하게 모든 목표를 동시에 달성하는 건 불가능.
트레이드오프. 최적의 균형점을 찾는 게 하이퍼파라미터 튜닝.

### 5.8 KD는 margin을 좁히는 게 아닌가?

```
의문: Teacher는 FP32라서 margin이 좁을 수 있음.
     KD로 Teacher를 따라하면 margin도 좁아지는 거 아님?

답: KD는 "정확한 값"이 아니라 "분포의 모양(순서, 비율)"을 전달.

Teacher:  안=-0.2, 앉=-15.3 (차이 15.1)
Student:  안=-0.5, 앉=-8.2  (차이 7.7)
  → KD: "안이 1등이고 앉이 2등인 건 맞아. OK."
  → 값 자체(-0.2)를 따라할 필요 없음

MarginLoss가 차이를 벌리면:
Student:  안=-0.2, 앉=-20.0 (차이 19.8)
  → KD: "안이 1등이고 앉이 2등. 순서 맞네. OK."
  → margin이 넓어져도 KD loss는 크게 안 올라감

KD가 잡는 건:
Student:  앉=-0.1, 안=-0.5 (순서 뒤집힘!)
  → KD: "안이 1등이어야 하는데 앉이 1등이네?! penalty!"
  → 순서가 바뀌는 걸 방지
```

---

## 6. 전체 QAT+KD 학습 흐름

### 6.1 한 Step의 전체 과정

```
1. 배치 로드
   음성 16개 + 정답 텍스트 16개

2. Teacher Forward (gradient 없음)
   음성 → [Teacher FP32 모델] → teacher_logits [16, 76, 2049]

3. Student Forward (gradient 있음)
   음성 → [Preprocessor] → mel
        → ★FakeQuantize → mel_q (양자화 오차 포함)
        → [Encoder] → encoded
        → ★FakeQuantize → encoded_q
        → [Decoder] → student_logits
        → ★FakeQuantize → student_logits_q [16, 76, 2049]

4. Loss 계산
   CTC_loss = CTC(student_logits_q, 정답텍스트)
   
   margins = top1 - top2 of student_logits_q
   margin_loss = ReLU(0.5 - margins).mean()
   
   kd_loss = KL_div(student_soft, teacher_soft) * T²
   
   total_loss = CTC_loss + 0.1 * margin_loss + 0.5 * kd_loss

5. 역전파
   total_loss.backward()
   → Student 가중치에 gradient 전파
   → Teacher 가중치는 고정 (gradient 안 흐름)

6. 가중치 업데이트
   optimizer.step()
   → Student 가중치만 업데이트됨

7. 다음 배치로 반복
```

### 6.2 QAT만 vs QAT+KD 비교

```
QAT만 (train_qat.py):
  모델 1개 (Student만)
  loss = CTC + MarginLoss
  GPU 메모리: ~9GB

QAT+KD (train_qat_kd.py):
  모델 2개 (Teacher + Student)
  loss = CTC + MarginLoss + KD_Loss
  GPU 메모리: ~18GB (2배)
  학습 속도: ~1.5배 느림 (teacher forward 추가)
```

---

## 7. 현재 실험 설정 비교

### 7.1 실험 히스토리

| 실험 | Loss | margin_target | KD | 데이터 | T527 CER |
|------|------|--------------|-----|--------|----------|
| PTQ (baseline) | - | - | X | - | 10.02% |
| QAT margin0.3 | CTC+Hinge | 0.3 | X | 100k | 5.3% |
| QAT+KD margin0.5 | CTC+Hinge+KD | 0.5 | O | 100k | 테스트 필요 |
| QAT+KD margin1.0 | CTC+Hinge+KD | 1.0 | O | 100k | 학습 중 |

### 7.2 변경한 변수

```
실험 1→2: KD 추가 + margin 0.3→0.5
  → 두 변수를 동시에 바꿔서 어떤 게 효과인지 분리 불가
  → 이상적으로는 하나만 바꿔야 하지만, 시간 제약으로 동시 변경

실험 2→3: margin 0.5→1.0 (KD 유지)
  → margin_target만 변경 → 공정한 비교 가능
```

---

## 8. 용어 정리

| 용어 | 풀네임 | 설명 |
|------|--------|------|
| CTC | Connectionist Temporal Classification | 프레임별 정답 없이 학습 가능하게 하는 loss |
| CTC decode | - | 모델 출력에서 텍스트 추출 (중복 제거 + blank 제거) |
| QAT | Quantization-Aware Training | FakeQuantize 넣고 학습 |
| PTQ | Post-Training Quantization | 학습 후 양자화 (Acuity) |
| FakeQuantize | - | float→uint8→float 변환 (양자화 오차 시뮬레이션) |
| STE | Straight-Through Estimator | round()의 gradient를 1로 근사 |
| KD | Knowledge Distillation | Teacher-Student 학습 |
| Hinge Margin Loss | - | ReLU(target - margin) 형태의 loss |
| Dark Knowledge | - | Teacher 출력 분포에 담긴 클래스 간 유사도 정보 |
| Temperature | - | softmax 분포의 날카로움 조절 (T↓ 날카로움, T↑ 부드러움) |
| scale | - | float 범위를 uint8 범위로 매핑하는 비율 |
| zero_point | - | float 0.0이 uint8 몇에 매핑되는지 |
| LSQ | Learned Step Size Quantization | scale을 gradient descent로 학습 |
| argmax | - | 배열에서 가장 큰 값의 index |
| logit | - | 모델이 각 vocab에 대해 매기는 점수 (softmax 이전) |
| margin | - | top1 logit과 top2 logit의 차이 |
