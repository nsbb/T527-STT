# 양자화 고수가 되기 위한 로드맵

**날짜:** 2026-03-26
**현재 수준:** 양자화 도구 사용자
**목표:** 양자화 전문가 (새 NPU + 새 모델 조합에서 독립적으로 판단 및 최적화 가능)

---

# 1. 현재 수준 진단

## 1.1 할 줄 아는 것

- Acuity 스크립트 돌리기 (import → quantize → export)
- 결과 보고 잘 됐는지 안 됐는지 판단 (CER 측정, 비교)
- 모델 찾아서 ONNX 변환 → NB 파이프라인 태우기
- 아키텍처별 양자화 친화성 판단 (실험 기반)
- QAT 학습 스크립트 실행
- W8A8 vs W4A16 차이 이해
- Split Model 접근법 시도 (lm_head FP32 분리)
- activation range가 양자화 성패를 결정한다는 경험적 이해

## 1.2 못 하는 것

- `scale = (max - min) / 255`, `zp = round(-min / scale)` 직접 유도
- KL divergence calibration이 내부에서 뭘 하는지 코드 레벨로 설명
- .quantize 파일의 `fl: 9` (fractional length)가 뭘 의미하는지 정확히 모름
- 새 NPU 줬을 때 혼자서 양자화 파이프라인 처음부터 구축
- 논문 읽고 구현
- quantization error의 수학적 bound 계산
- NPU MAC unit이 실제로 uint8 연산을 어떻게 처리하는지 설명
- per-channel vs per-tensor quantization을 직접 구현
- SmoothQuant, GPTQ, AWQ를 코드 레벨로 이해
- 양자화된 weight가 실제로 NPU에서 어떻게 메모리에 배치되는지 모름

## 1.3 핵심 문제

**도구가 해주는 것과 본인이 이해하는 것의 경계를 모른다.**

Acuity가 내부에서 뭘 하는지 모르는 상태에서 결과만 보고 판단하고 있음. 이건 "양자화를 하는 것"이 아니라 "양자화 도구를 쓰는 것"임.

비유: 자동차를 운전할 줄 아는 것(도구 사용자)과 엔진을 설계할 줄 아는 것(전문가)은 다르다. 지금은 운전은 잘 하는데 엔진 후드를 열어본 적이 없는 상태.

---

# 2. 양자화 전문가의 역량 정의

양자화 고수를 정의하면 이런 사람이다:

### Level 0: 도구 사용자 (현재)
- 주어진 툴체인으로 양자화 실행 가능
- 결과를 보고 성공/실패 판단 가능
- "이 모델은 안 되더라" 수준의 경험적 지식

### Level 1: 이해자
- 양자화 수학을 직접 유도할 수 있음
- 각 기법의 장단점을 수학적으로 설명할 수 있음
- "이 모델이 이 NPU에서 왜 안 되는지" 원인을 정확히 진단

### Level 2: 구현자
- 양자화 파이프라인을 처음부터 짤 수 있음
- 기존 기법을 코드로 재현할 수 있음
- 툴체인에 종속되지 않음

### Level 3: 설계자 (고수)
- 새로운 양자화 기법을 만들 수 있음
- 처음 보는 NPU + 모델 조합에서 30분 안에 판단 가능
- 논문을 쓸 수 있는 수준

**목표: Level 0 → Level 2까지 올리고, Level 3의 문을 여는 것.**

---

# 3. 로드맵

## Step 1: 양자화 수학을 손으로 해보기

### 3.1.1 목표
float → integer 변환 과정을 계산기로 직접 할 수 있을 것. .quantize 파일의 모든 숫자가 무엇을 의미하는지 이해할 것.

### 3.1.2 배경지식: 양자화란 정확히 뭔가

양자화는 연속적인 실수(float)를 이산적인 정수(integer)로 매핑하는 것이다.

```
실수 세계:   ... -1.234, -1.233, -1.232, ... 0.000, ... 3.141, 3.142 ...  (무한한 값)
정수 세계:   0, 1, 2, 3, 4, ..., 253, 254, 255                             (256개 값)
```

핵심은 **매핑 함수**를 어떻게 정의하느냐이다. 이 매핑 함수가 quantizer이고, 매핑의 파라미터가 scale과 zero_point(또는 fractional length)이다.

### 3.1.3 Asymmetric Affine Quantization (uint8)

T527 Acuity에서 uint8 양자화 시 사용하는 방식. `.quantize` 파일에서 `quantizer: asymmetric_affine`로 표시됨.

**공식:**

```
# Quantize (float → uint8)
scale = (max_value - min_value) / 255
zero_point = round(-min_value / scale)
q = clamp(round(x / scale) + zero_point, 0, 255)

# Dequantize (uint8 → float)
x_hat = (q - zero_point) * scale
```

**왜 asymmetric인가:**
- 실수 범위가 0 중심이 아닐 수 있음 (예: ReLU 출력은 항상 ≥ 0)
- min과 max를 독립적으로 매핑 → 비대칭(asymmetric)
- zero_point로 실수 0.0이 정수 몇에 매핑되는지 조절

**손계산 연습 1:**

```
입력 float 값: [-2.0, -1.5, -0.5, 0.0, 0.3, 1.0, 1.7, 2.5, 3.0, 3.5]
min_value = -2.0, max_value = 3.5

step 1: scale = (3.5 - (-2.0)) / 255 = 5.5 / 255 = 0.02157
step 2: zero_point = round(-(-2.0) / 0.02157) = round(92.72) = 93

step 3: 각 값 양자화
  -2.0 → round(-2.0 / 0.02157) + 93 = round(-92.72) + 93 = -93 + 93 = 0     ✓ (min → 0)
  -1.5 → round(-1.5 / 0.02157) + 93 = round(-69.54) + 93 = -70 + 93 = 23
  -0.5 → round(-0.5 / 0.02157) + 93 = round(-23.18) + 93 = -23 + 93 = 70
   0.0 → round(0.0 / 0.02157) + 93 = 0 + 93 = 93                             ✓ (0.0 → zp)
   0.3 → round(0.3 / 0.02157) + 93 = round(13.91) + 93 = 14 + 93 = 107
   1.0 → round(1.0 / 0.02157) + 93 = round(46.36) + 93 = 46 + 93 = 139
   1.7 → round(1.7 / 0.02157) + 93 = round(78.81) + 93 = 79 + 93 = 172
   2.5 → round(2.5 / 0.02157) + 93 = round(115.90) + 93 = 116 + 93 = 209
   3.0 → round(3.0 / 0.02157) + 93 = round(139.08) + 93 = 139 + 93 = 232
   3.5 → round(3.5 / 0.02157) + 93 = round(162.26) + 93 = 162 + 93 = 255    ✓ (max → 255)

step 4: 역양자화 (dequantize)로 오차 확인
  0   → (0 - 93) * 0.02157 = -93 * 0.02157 = -2.006    (원래 -2.0, 오차 0.006)
  23  → (23 - 93) * 0.02157 = -70 * 0.02157 = -1.510   (원래 -1.5, 오차 0.010)
  70  → (70 - 93) * 0.02157 = -23 * 0.02157 = -0.496   (원래 -0.5, 오차 0.004)
  93  → (93 - 93) * 0.02157 = 0 * 0.02157 = 0.000      (원래 0.0, 오차 0.000) ✓ 정확
  107 → (107 - 93) * 0.02157 = 14 * 0.02157 = 0.302     (원래 0.3, 오차 0.002)
  ...
```

**핵심 관찰:**
- 최대 오차 = scale / 2 = 0.02157 / 2 = 0.01079 (반올림 오차)
- range가 넓을수록 scale이 커지고 → 오차가 커짐
- **이것이 activation range가 양자화 성패를 결정하는 수학적 이유**

### 3.1.4 Symmetric Quantization (int8)

Per-channel quantization에서 주로 사용. `.quantize` 파일에서 `quantizer: perchannel_symmetric_affine`로 표시됨.

**공식:**

```
# Quantize (float → int8)
scale = max(|max_value|, |min_value|) / 127
q = clamp(round(x / scale), -128, 127)

# Dequantize (int8 → float)
x_hat = q * scale
```

**왜 symmetric인가:**
- 0.0이 항상 정수 0에 매핑됨 (zero_point 불필요)
- 계산이 단순하고 하드웨어 구현 효율적
- 단점: 분포가 한쪽으로 치우치면 한쪽 range가 낭비됨

**손계산 연습 2:**

```
같은 입력: [-2.0, -1.5, -0.5, 0.0, 0.3, 1.0, 1.7, 2.5, 3.0, 3.5]

step 1: scale = max(|-2.0|, |3.5|) / 127 = 3.5 / 127 = 0.02756

step 2: 양자화
  -2.0 → round(-2.0 / 0.02756) = round(-72.57) = -73
   0.0 → round(0.0 / 0.02756) = 0                      ✓ 항상 정확
   3.5 → round(3.5 / 0.02756) = round(127.0) = 127     ✓ max → 127
  -2.0 → clamp(-73, -128, 127) = -73                    (OK, -128까지 여유 있음)

주목: -2.0은 -73에 매핑되는데, -128까지 55칸이 남음.
      이 55칸은 실제 데이터가 없는 범위 → 낭비!
      asymmetric이었으면 이 낭비가 없었음.
```

**asymmetric vs symmetric 비교:**

```
asymmetric: scale = 5.5 / 255 = 0.02157  (더 정밀)
symmetric:  scale = 3.5 / 127 = 0.02756  (28% 덜 정밀)

이유: symmetric은 -3.5~+3.5 = 7.0 range를 256칸에 매핑
      asymmetric은 -2.0~+3.5 = 5.5 range를 256칸에 매핑
      → 같은 칸 수에 더 좁은 범위 → 더 정밀
```

**그러면 왜 symmetric을 쓰는가?**
- zero_point가 0으로 고정 → 곱셈에서 offset 보정 불필요
- NPU 하드웨어에서 더 빠름 (MAC 연산 단순화)
- weight 분포는 보통 0 중심 대칭이라 낭비가 적음

### 3.1.5 Dynamic Fixed Point (int16)

T527 Acuity에서 int16 양자화 시 사용. `.quantize` 파일에서 `quantizer: dynamic_fixed_point`, `fl: N`으로 표시됨.

**공식:**

```
# Quantize (float → int16)
q = clamp(round(x * 2^fl), -32768, 32767)

# Dequantize (int16 → float)
x_hat = q / 2^fl = q * 2^(-fl)
```

**fl (fractional length)란:**
- 고정소수점에서 소수부에 할당하는 비트 수
- fl=9이면 소수점 아래 9비트 → step size = 2^(-9) = 1/512 ≈ 0.00195
- int16 = 16bit = 1(부호) + (15-fl)(정수부) + fl(소수부)

**실제 .quantize 파일 예시:**

```yaml
'@Conv_/decoder_layers/decoder_layers.0/Conv_3:out0':
    qtype: i16
    quantizer: dynamic_fixed_point
    rounding: rtne
    max_value: 16.725765228271484
    min_value: -40.68741989135742
    fl: 9
```

**이 레이어를 분석해보자:**

```
fl = 9
step size = 2^(-9) = 0.00195
표현 가능 범위: -32768 / 512 ~ 32767 / 512 = -64.0 ~ +63.998
실제 데이터 범위: -40.687 ~ +16.726
→ 범위 안에 들어오므로 OK

유효 정밀도 계산:
  max_value 근처에서: 16.726 / 0.00195 ≈ 8577 단계로 표현
  min_value 근처에서: 40.687 / 0.00195 ≈ 20865 단계로 표현

uint8이었다면:
  scale = (16.726 - (-40.687)) / 255 = 57.413 / 255 = 0.2251
  → step size 0.2251 vs 0.00195 → int16이 115배 정밀

그런데 CER 차이는 10.02% vs 9.59% = 0.43%p밖에 안 됨.
→ 이 레이어에서는 0.2251 step이어도 충분하다는 뜻.
→ Conformer 아키텍처 덕분에 activation 분포가 양자화 친화적.
```

**손계산 연습 3:**

```
fl=9일 때, float 값 -40.687을 int16으로 양자화:
q = round(-40.687 * 512) = round(-20831.744) = -20832
dequantize: -20832 / 512 = -40.6875
오차: |-40.687 - (-40.6875)| = 0.0005

fl=9일 때, float 값 16.726을 int16으로 양자화:
q = round(16.726 * 512) = round(8563.712) = 8564
dequantize: 8564 / 512 = 16.7265625
오차: |16.726 - 16.7266| = 0.0006
```

### 3.1.6 fl은 어떻게 결정되는가

**Acuity가 자동으로 결정하는 로직:**

```
fl을 가능한 크게 하고 싶음 (정밀도↑)
하지만 max_value가 표현 가능 범위 안에 들어와야 함

int16 범위: [-2^15, 2^15 - 1] = [-32768, 32767]
max_value * 2^fl <= 32767
min_value * 2^fl >= -32768

max_abs = max(|max_value|, |min_value|)
fl = floor(log2(32767 / max_abs))

예: max_abs = 40.687
fl = floor(log2(32767 / 40.687)) = floor(log2(805.5)) = floor(9.65) = 9 ✓
```

**fl이 작으면:** 정수부 많고 소수부 적음 → 큰 값은 잘 표현하지만 작은 차이 구분 못 함
**fl이 크면:** 소수부 많고 정수부 적음 → 작은 차이까지 구분하지만 큰 값 표현 못 함

### 3.1.7 세 가지 quantizer 비교 종합

같은 값 `x = 1.234`를 양자화한다고 가정 (range: -5.0 ~ +5.0):

```
Asymmetric Affine (uint8):
  scale = 10.0/255 = 0.03922
  zp = round(5.0/0.03922) = 128 (우연히 정가운데)
  q = round(1.234/0.03922) + 128 = round(31.46) + 128 = 31 + 128 = 159
  x_hat = (159-128) * 0.03922 = 31 * 0.03922 = 1.21582
  오차 = 0.01818

Symmetric (int8):
  scale = 5.0/127 = 0.03937
  q = round(1.234/0.03937) = round(31.34) = 31
  x_hat = 31 * 0.03937 = 1.22047
  오차 = 0.01353

Dynamic Fixed Point (int16, fl=12):
  q = round(1.234 * 4096) = round(5054.464) = 5054
  x_hat = 5054 / 4096 = 1.233887
  오차 = 0.000113

비교:
  uint8 오차:  0.018   (step ≈ 0.039)
  int8 오차:   0.014   (step ≈ 0.039)
  int16 오차:  0.00011 (step ≈ 0.000244) → 160배 정밀
```

### 3.1.8 왜 이걸 알아야 하는가

**이걸 모르면:**
- .quantize 파일이 그냥 숫자 나열로 보임
- Acuity가 "알아서 해주겠지" 상태에서 벗어나지 못함
- 커스텀 양자화가 불가능

**이걸 알면:**
- .quantize 파일의 모든 숫자가 읽힘
- "이 레이어는 step이 0.2인데 logit margin이 0.18이니까 argmax 뒤집힐 수 있다" 계산 가능
- 특정 레이어만 fl을 바꾸거나 quantizer를 바꾸는 실험 가능

### 3.1.9 연습 과제

1. `.quantize` 파일에서 임의의 레이어 5개를 골라 step size 계산하기
2. 그 5개 레이어의 "표현 가능 범위" vs "실제 데이터 범위" 비교하기
3. 가장 range가 넓은 레이어를 찾아서 "이 레이어가 양자화에서 가장 취약할 것" 가설 세우기
4. Conformer의 logit 출력 레이어(LogSoftmax)의 step size와 margin 비교하기

### 예상 소요: 2~3시간
### 완료 기준: .quantize 파일의 모든 필드를 보고 "이 레이어는 step size가 X이고, 이 range에서 Y bit의 유효 정밀도를 가지며, 표현 가능 범위는 Z이다"라고 말할 수 있을 것.

---

## Step 2: 논문 3편 읽기

### 3.2.1 논문 1: 기본 중의 기본

**제목:** Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
**저자:** Jacob et al. (Google), 2018
**링크:** arXiv:1712.05877

**읽어야 하는 이유:**
- Asymmetric affine quantization의 원조
- TFLite, Acuity 등 모든 edge 양자화 툴의 기반
- uint8 행렬곱이 실제로 어떻게 수행되는지 설명

**핵심 내용 미리보기:**

```
실수 행렬곱: r3 = r1 * r2
양자화 행렬곱:
  r = S * (q - Z)  (S=scale, Z=zero_point)

  S3(q3 - Z3) = S1(q1 - Z1) * S2(q2 - Z2)
  q3 = Z3 + (S1*S2/S3) * (q1 - Z1) * (q2 - Z2)

  M = S1*S2/S3는 float인데, 이걸 fixed-point로 근사:
  M ≈ M0 * 2^(-n)  (M0는 int32, n은 shift amount)

  → 전체 연산이 정수 곱셈 + bit shift로 수행됨!
```

**이걸 이해하면:**
- NPU가 실제로 uint8 곱셈을 어떻게 하는지 알게 됨
- accumulator가 왜 int32여야 하는지 이해됨 (uint8 * uint8 = 최대 65025, uint16으로 부족)
- bias가 int32인 이유, batch norm folding이 왜 필요한지 등

**읽을 때 집중할 섹션:**
- Section 2: Quantized Inference (핵심 수학)
- Section 3: Training with Simulated Quantization (QAT의 기초)
- Appendix B: ARM NEON 구현 (하드웨어와 연결)

**읽은 후 자가 점검:**
- [ ] uint8 행렬곱에서 M0와 shift를 직접 계산할 수 있는가?
- [ ] STE(Straight-Through Estimator)가 왜 필요한지 설명할 수 있는가?
- [ ] batch norm folding을 직접 할 수 있는가?

---

### 3.2.2 논문 2: PTQ 종합 서베이

**제목:** A White Paper on Neural Network Quantization
**저자:** Nagel et al. (Qualcomm AI Research), 2021
**링크:** arXiv:2106.08295

**읽어야 하는 이유:**
- 양자화 기법의 분류 체계 (PTQ vs QAT, symmetric vs asymmetric, per-tensor vs per-channel)
- 각 기법의 수학적 근거를 한 곳에서 정리
- 실무에서 어떤 상황에 어떤 기법을 쓰는지 가이드

**핵심 내용 미리보기:**

```
PTQ 기법의 핵심 문제: 최적의 clipping range를 어떻게 찾는가?

1. MinMax: 그냥 min/max 사용
   - 장점: 단순
   - 단점: outlier 하나에 전체 range 왜곡

2. MSE 최소화: argmin_Δ E[(x - Q(x))^2]
   - Δ = clipping threshold
   - 양자화 오차의 기대값을 최소화하는 Δ를 찾음
   - 장점: 수학적으로 최적
   - 단점: 전수 탐색 필요

3. KL Divergence: argmin_Δ KL(P || Q)
   - P = 원래 분포, Q = 양자화된 분포
   - 정보 이론적으로 두 분포의 차이를 최소화
   - 장점: 분포의 모양을 고려
   - 단점: histogram 기반이라 bin 수에 민감

4. 학습 기반: scale/zp를 gradient descent로 학습
   - AdaRound, BRECQ, QDrop 등
   - 장점: 가장 좋은 성능
   - 단점: 학습 필요 (PTQ인데 학습?)
```

**읽을 때 집중할 섹션:**
- Section 3: Post-Training Quantization (4가지 방법 상세)
- Section 4: Quantization-Aware Training (QAT 수학)
- Section 5.1: Per-channel vs Per-tensor (granularity)
- Table 1, 2: 기법별 성능 비교 (숫자로 보는 차이)

**읽은 후 자가 점검:**
- [ ] MinMax, KL, MSE, Percentile의 차이를 수식으로 설명할 수 있는가?
- [ ] per-channel이 per-tensor보다 좋은 이유를 수학적으로 설명할 수 있는가?
- [ ] AdaRound가 왜 round-to-nearest보다 좋은지 설명할 수 있는가?
- [ ] 우리 T527 프로젝트에서 어떤 기법이 적용 가능하고 불가능한지 판단할 수 있는가?

---

### 3.2.3 논문 3: 최신 기법

**제목:** SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
**저자:** Xiao et al. (MIT, NVIDIA), 2023
**링크:** arXiv:2211.10438

**읽어야 하는 이유:**
- activation outlier 문제를 수학적으로 해결하는 기법
- 우리가 T527에서 겪은 문제(activation range가 넓어서 양자화 실패)와 직결
- W8A8 환경에서의 해결책을 제시

**핵심 아이디어:**

```
문제: Transformer의 activation에 outlier가 있음
  - 대부분의 값: [-1, 1]
  - 소수의 outlier: [-100, 100]
  - 이 outlier 때문에 scale이 커지고 → 대부분의 값이 뭉개짐

기존 접근: activation을 그냥 양자화 → 실패
SmoothQuant 접근: activation의 outlier를 weight 쪽으로 수학적으로 옮김

핵심 공식:
  Y = X * W  (X=activation, W=weight)
  Y = (X * diag(s)^-1) * (diag(s) * W)
  Y = X_smooth * W_smooth

  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)   (α ∈ [0, 1])

  X_smooth = X / s  → activation range 축소
  W_smooth = W * s  → weight range 확대 (weight는 양자화에 강건)

즉: activation이 감당 못 하는 양자화 부담을 weight에게 떠넘기는 것
```

**T527 프로젝트와의 연결:**
- wav2vec2 한국어의 activation range가 영어보다 5~50배 넓었음
- SmoothQuant를 적용하면 이 range를 줄일 수 있을까?
- 근데 Acuity가 SmoothQuant를 지원 안 함 → ONNX 레벨에서 직접 적용해야 함
- **커스텀 PTQ 파이프라인(Step 3)이 있으면 이걸 실험 가능**

**읽을 때 집중할 섹션:**
- Section 3: Method (핵심 공식 유도)
- Section 3.2: Migration Strength α (얼마나 옮길지 조절)
- Figure 3: smoothing 전후 activation 분포 변화 (시각적 이해)
- Table 3: W8A8 결과 (우리 T527과 같은 조건)

**읽은 후 자가 점검:**
- [ ] smoothing factor s를 직접 계산할 수 있는가?
- [ ] α를 어떻게 정하는지, 왜 0.5가 많이 쓰이는지 설명할 수 있는가?
- [ ] ONNX 모델에 SmoothQuant를 직접 적용하는 방법을 설계할 수 있는가?
- [ ] Conformer에 SmoothQuant를 적용하면 효과가 있을지 없을지 예측할 수 있는가?

---

### 3.2.4 보너스 논문 (여유 있으면)

| 논문 | 핵심 | 읽어야 하는 이유 |
|------|------|-----------------|
| Nagel et al. 2019 "Data-Free Quantization" | calibration 데이터 없이 양자화 | calibration data의 역할을 역으로 이해 |
| Li et al. 2021 "BRECQ" | block-wise reconstruction | PTQ의 최신 SOTA, layer별 최적화 |
| Lin et al. 2024 "AWQ" | activation-aware weight quantization | SmoothQuant의 진화형 |
| Gulati et al. 2020 "Conformer" | Conformer 원논문 | 아키텍처를 설계 관점에서 이해 |

### 예상 소요: 각 2~3일 (총 1~2주)
### 완료 기준: 각 논문의 핵심 공식을 직접 유도할 수 있고, T527 프로젝트에 적용 가능한지 근거를 들어 판단할 수 있을 것.

---

## Step 3: 커스텀 PTQ 파이프라인 직접 짜기

### 3.3.1 목표
Acuity의 quantize 단계를 우리 코드로 대체한다. `.quantize` 파일을 직접 생성하여, Acuity는 import와 export만 사용한다.

### 3.3.2 왜 이걸 해야 하는가

현재 파이프라인:
```
ONNX → [Acuity import] → .json + .data → [Acuity quantize] → .quantize → [Acuity export] → .nb
                                           ^^^^^^^^^^^^^^^^
                                           블랙박스. 내부에서 뭘 하는지 모름.
                                           KL인지 MinMax인지도 확실하지 않음.
                                           파라미터 튜닝 불가.
```

목표 파이프라인:
```
ONNX → [Acuity import] → .json + .data → [우리 코드] → .quantize → [Acuity export] → .nb
                                           ^^^^^^^^^^^^
                                           완전히 투명. 모든 레이어의 양자화를
                                           우리가 통제. 아무 알고리즘이나 실험 가능.
```

### 3.3.3 전체 구현 계획

**Phase 1: Activation 수집기 (1~2일)**

```python
# 의사코드
import onnxruntime as ort

session = ort.InferenceSession("conformer.onnx")

# 모든 중간 레이어를 출력으로 등록
all_nodes = get_all_intermediate_nodes(session)

# calibration 데이터로 추론
for audio in calibration_dataset:  # 100~500개
    mel = compute_mel(audio)
    outputs = session.run(all_nodes, {"input": mel})

    for node_name, output in zip(all_nodes, outputs):
        # 각 레이어의 min, max, histogram 수집
        stats[node_name].update(output)

# 결과: 레이어별 {min, max, histogram, percentiles}
```

**핵심 포인트:**
- ONNX Runtime의 `get_outputs()` 대신 중간 노드도 출력으로 받아야 함
- `onnx.helper`로 모든 중간 노드를 output으로 추가하는 전처리 필요
- 또는 ONNX Runtime의 profiling 기능 활용

**Phase 2: Calibration 알고리즘 4종 구현 (3~5일)**

각 알고리즘이 하는 일: 수집된 activation 분포에서 최적의 `(min_value, max_value)`를 결정.

```python
# 1. MinMax — 가장 단순
def calibrate_minmax(stats):
    return stats.min, stats.max

# 2. Percentile — outlier 제거
def calibrate_percentile(stats, percentile=99.99):
    return np.percentile(stats.all_values, 100 - percentile), \
           np.percentile(stats.all_values, percentile)

# 3. MSE — 양자화 오차 최소화
def calibrate_mse(stats, num_bits=8):
    best_loss = float('inf')
    best_range = None
    # 후보 threshold를 순회
    for percentile in np.arange(99.0, 100.0, 0.01):
        min_val = np.percentile(stats.all_values, 100 - percentile)
        max_val = np.percentile(stats.all_values, percentile)
        # 이 range로 양자화했을 때 MSE 계산
        scale = (max_val - min_val) / (2**num_bits - 1)
        q = np.clip(np.round(stats.all_values / scale), 0, 2**num_bits - 1)
        x_hat = q * scale + min_val
        mse = np.mean((stats.all_values - x_hat) ** 2)
        if mse < best_loss:
            best_loss = mse
            best_range = (min_val, max_val)
    return best_range

# 4. KL Divergence — 정보 손실 최소화
def calibrate_kl(stats, num_bits=8, num_bins=2048):
    histogram = np.histogram(stats.all_values, bins=num_bins)
    # TensorRT 방식:
    # 1. reference 분포 P (원래 histogram)
    # 2. 다양한 threshold로 양자화된 분포 Q 생성
    # 3. KL(P || Q)를 최소화하는 threshold 선택
    best_kl = float('inf')
    best_threshold = None
    for i in range(128, num_bins):
        # threshold = histogram의 i번째 bin edge
        # P = histogram[:i]를 정규화
        # Q = histogram[:i]를 num_bits 구간으로 양자화 후 복원
        # KL = sum(P * log(P / Q))
        ...
    return -best_threshold, best_threshold  # symmetric 가정
```

**Phase 3: .quantize YAML 생성기 (1~2일)**

```python
def generate_quantize_file(layer_ranges, quantizer_type="asymmetric_affine",
                           qtype="uint8", output_path="model.quantize"):
    lines = ["# !!!This file disallow TABs!!!", "version: 2", "quantize_parameters:"]

    for layer_name, (min_val, max_val) in layer_ranges.items():
        if quantizer_type == "asymmetric_affine":
            lines.append(f"  '{layer_name}':")
            lines.append(f"    qtype: u8")
            lines.append(f"    quantizer: asymmetric_affine")
            lines.append(f"    rounding: rtne")
            lines.append(f"    max_value: {max_val}")
            lines.append(f"    min_value: {min_val}")
            # scale과 zp는 Acuity가 min/max에서 계산

        elif quantizer_type == "dynamic_fixed_point":
            max_abs = max(abs(min_val), abs(max_val))
            fl = int(np.floor(np.log2(32767 / max_abs))) if max_abs > 0 else 15
            lines.append(f"  '{layer_name}':")
            lines.append(f"    qtype: i16")
            lines.append(f"    quantizer: dynamic_fixed_point")
            lines.append(f"    rounding: rtne")
            lines.append(f"    max_value: {max_val}")
            lines.append(f"    min_value: {min_val}")
            lines.append(f"    fl: {fl}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
```

**핵심 주의사항:**
- `.quantize` 파일에서 TAB 사용 금지 (첫 줄에 명시됨)
- 레이어 이름이 Acuity import 결과의 `.json` 파일과 정확히 일치해야 함
- `@` prefix와 `:out0` suffix 형식 주의

**Phase 4: 검증 및 비교 (2~3일)**

```
실험 매트릭스:
┌──────────────┬───────────┬───────────┬───────────┬───────────┐
│              │ MinMax    │ Percentile│ KL        │ MSE       │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ uint8        │ CER: ?    │ CER: ?    │ CER: ?    │ CER: ?    │
│ int16 DFP    │ CER: ?    │ CER: ?    │ CER: ?    │ CER: ?    │
│ PCQ (int8)   │ CER: ?    │ CER: ?    │ CER: ?    │ CER: ?    │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Acuity 기본  │ CER: 10.02% (uint8 KL) │ CER: 9.59% (int16) │
└──────────────┴───────────────────────┴─────────────────────────┘

비교 대상: Acuity 기본 결과와 우리 커스텀 결과
성공 기준: Acuity와 동일하거나 더 좋은 CER
```

### 3.3.4 고급 실험 (Phase 4 이후)

**Mixed Quantizer 실험:**
```yaml
# 레이어별로 다른 quantizer를 쓰면?
'@encoder_layer_0:out0':
    quantizer: asymmetric_affine     # range 넓은 레이어는 uint8
'@encoder_layer_5:out0':
    quantizer: dynamic_fixed_point   # 민감한 레이어는 int16
'@logit_output:out0':
    quantizer: asymmetric_affine     # 출력은 margin 크니까 uint8 OK
```

**Sensitivity 기반 자동 배정:**
```python
# 각 레이어를 하나씩 양자화하고 나머지는 FP32로 두고 CER 측정
# → 어떤 레이어가 양자화에 가장 민감한지 ranking
for layer in all_layers:
    quantize_only(layer)  # 이 레이어만 양자화
    cer = measure_cer()   # CER 측정
    sensitivity[layer] = cer - baseline_cer

# 민감한 레이어: int16, 덜 민감한 레이어: uint8 → .quantize 생성
```

**SmoothQuant 적용:**
```python
# ONNX 모델의 Linear 레이어 앞에 smoothing factor 삽입
# weight를 미리 s로 곱해서 저장
# activation은 추론 시 1/s로 나눠짐
# → activation range 축소, weight range 확대
```

### 3.3.5 이걸 하면 얻는 것

- 양자화 수학이 코드로 체화됨 (Step 1의 손계산이 프로그래밍으로 확장)
- Acuity에 종속되지 않는 독립적 양자화 역량
- 새로운 calibration 알고리즘을 자유롭게 실험 가능
- .quantize 파일 구조를 완전히 이해하여 레이어별 커스텀 양자화 가능
- **"내가 만든 양자화가 Acuity보다 낫다"** 를 증명할 수 있는 프레임워크

### 예상 소요: 2~3주
### 완료 기준:
- Acuity quantize 없이 .quantize 파일을 직접 생성하여 NB를 만들 수 있음
- 4가지 알고리즘의 CER 차이를 측정하고 왜 그런지 수학적으로 설명할 수 있음
- "Acuity가 내부에서 뭘 하는지"를 코드 레벨로 완전히 이해

---

## Step 4: 하드웨어 이해하기

### 3.4.1 목표
NPU가 양자화된 모델을 실제로 어떻게 실행하는지 이해한다. "소프트웨어에서의 양자화"와 "하드웨어에서의 양자화"의 차이를 안다.

### 3.4.2 NPU가 실제로 하는 일

```
CPU에서의 float 행렬곱:
  c[i][j] = Σ a[i][k] * b[k][j]    (float32 곱셈, float32 누적)
  → 1 곱셈 = ~4 clock cycles

NPU에서의 uint8 행렬곱:
  acc[i][j] = Σ (uint8)a[i][k] * (uint8)b[k][j]   (uint8 곱셈, int32 누적)
  result[i][j] = (uint8) requantize(acc[i][j])       (int32 → uint8 변환)
  → MAC array: 수백~수천 개의 곱셈기가 동시에 연산 (1 cycle에 수백 곱셈)
```

**T527 NPU (Vivante VIP9000NANOSI_PLUS) 스펙:**
- 2 TOPS (Tera Operations Per Second)
- = 초당 2조 번의 uint8 곱셈+덧셈
- MAC array 크기: 추정 ~512 units (공개 정보 제한적)
- SRAM: 추정 ~512KB (on-chip buffer)
- Bandwidth: DDR 대역폭에 의존

**왜 uint8이 빠른가:**
```
float32: 32bit × 32bit 곱셈기 → 면적 크고 전력 많이 먹음
uint8:   8bit × 8bit 곱셈기 → 면적 1/16, 전력 ~1/30
같은 칩 면적에 16배 많은 곱셈기 탑재 가능
→ 이론적 throughput 16배 (실제로는 메모리 대역폭 병목으로 ~4-8배)
```

**왜 FP16이 SW 에뮬레이션인가 (T527):**
```
T527 NPU에 FP16 곱셈기가 물리적으로 없음.
uint8 곱셈기만 있음.
FP16 연산을 하려면:
  1. FP16 → 정수 분해
  2. 정수 연산으로 FP16 곱셈을 시뮬레이션
  3. 결과를 다시 FP16으로 조합
  → 1 FP16 곱셈 = ~20-40 uint8 연산 → ~25배 느림 (실측: 42배)
```

### 3.4.3 Accumulator의 중요성

```
uint8 × uint8 = 최대 255 × 255 = 65,025
→ uint16 (max 65,535) 으로 1개 곱셈은 담을 수 있음

하지만 행렬곱에서는 K개를 누적:
  acc = Σ(k=0..K-1) a[k] * b[k]
  K=512 (Conformer d_model)일 때:
  최대값 = 512 × 65,025 = 33,292,800
  → int32 (max 2,147,483,647) 필요

이것이 NPU에 int32 accumulator가 필요한 이유.
accumulator overflow가 발생하면 결과가 완전히 깨짐.
```

**T527에서의 실제 흐름:**
```
1. SRAM에 uint8 weight tile 로드 (예: 64×64)
2. SRAM에 uint8 activation tile 로드 (예: 64×64)
3. MAC array에서 uint8×uint8 곱셈, int32 accumulator에 누적
4. Accumulator에서 requantize (int32 → uint8)
   - scale 적용 (fixed-point 곱셈 + shift)
   - bias 더하기 (int32)
   - activation function 적용 (LUT 또는 piecewise linear)
   - clamp to [0, 255]
5. 결과 uint8을 SRAM에 저장
6. 다음 tile로 이동
```

### 3.4.4 다른 NPU와의 비교

| 항목 | T527 (Vivante) | RK3588 (RKNN) | Jetson Orin (TensorRT) | Hexagon (QNN) |
|------|---------------|---------------|----------------------|---------------|
| INT8 HW | O | O | O | O |
| FP16 HW | **X** (SW emul) | **O** | **O** | **O** |
| Mixed Precision | **X** | O (layer별) | O (layer별) | O (layer별) |
| Activation 양자화 | **강제 uint8** | 선택 가능 | 선택 가능 | 선택 가능 |
| Per-channel | O | O | O | O |
| 양자화 유연성 | **최저** | 중간 | 최고 | 높음 |
| 성능 (TOPS) | 2 | 6 | 100+ | 15+ |

**핵심 인사이트:**
- T527이 양자화가 어려운 이유는 성능이 낮아서가 아님
- **FP16 HW 미지원 + Mixed Precision 불가** 때문
- 다른 NPU에서는 "이 레이어만 FP16"이 가능 → 양자화 실패율 급감
- T527에서는 전부 uint8이어야 함 → 모든 레이어가 uint8을 견뎌야 함 → 아키텍처가 결정적

### 3.4.5 SRAM과 Tiling

```
NPU의 on-chip SRAM은 제한적 (~512KB).
모델 weight (Conformer: 102MB) >> SRAM

→ weight를 tile 단위로 잘라서 DDR에서 SRAM으로 로드
→ 한 tile 연산 완료 → 다음 tile 로드 → 반복

Tiling이 중요한 이유:
  - tile 크기가 MAC array와 맞지 않으면 utilization 떨어짐
  - DDR → SRAM 대역폭이 병목이 되면 compute가 idle
  - NB 파일은 이 tiling 스케줄을 미리 계산해놓은 것

이것이 NB export를 Acuity가 해야 하는 이유.
NPU 하드웨어의 tiling 스케줄을 알아야 NB를 만들 수 있고,
이건 VeriSilicon만 알고 있음.
```

### 3.4.6 연습 과제

1. Conformer의 첫 번째 Conv 레이어의 MAC 연산 횟수를 계산해보기
   - input: [1, 80, 304] (mel 80채널, 304프레임)
   - kernel: [256, 80, 3, 3]
   - 총 곱셈 횟수 = ?
   - T527 2 TOPS로 몇 ms 걸리는지 이론값 계산

2. accumulator overflow가 발생하는 조건을 계산해보기
   - d_model=512, uint8 weight, uint8 activation
   - 최악의 경우 accumulator 값은?
   - int32로 충분한가?

### 예상 소요: 1주 (읽기 + 계산)
### 완료 기준: "NPU가 uint8 행렬곱을 어떻게 수행하는지" 화이트보드에 그리면서 설명할 수 있을 것.

---

## Step 5: 다른 NPU 해보기

### 3.5.1 목표
T527 전문가가 아니라 "양자화 전문가"가 된다. NPU마다 양자화 제약이 다르다는 것을 체감으로 알고, 범용적 판단력을 키운다.

### 3.5.2 추천: RK3588 (RKNN) 먼저

**추천 이유:**
- 이미 T527 프로젝트에서 RK3588 비교 데이터가 있음 (wav2vec2-xls-r-300m CER 11.78%)
- RKNN-Toolkit2가 오픈소스이고 Python API가 깔끔
- INT8 + FP16 mixed precision 지원 → T527과의 직접 비교 가능
- 보드 가격이 저렴 (~$60-100)

### 3.5.3 같은 모델로 비교 실험

```
실험: SungBeom Conformer를 두 NPU에서 양자화

T527 (Acuity):
  - ONNX → Acuity import → uint8 quantize → NB export
  - CER: 10.02%
  - 추론: 233ms

RK3588 (RKNN):
  - ONNX → rknn.load_onnx() → rknn.build(do_quantization=True) → .rknn export
  - CER: ?
  - 추론: ?

비교 포인트:
  1. 같은 모델, 같은 INT8인데 CER이 다른가?
  2. 다르다면 왜? calibration 알고리즘 차이? HW 연산 차이?
  3. RK3588에서 FP16 mixed를 쓰면 어떻게 달라지는가?
  4. wav2vec2는 RK3588에서 되고 T527에서 안 되는 이유를 하드웨어 차이로 설명 가능한가?
```

### 3.5.4 후보 NPU 상세

| NPU | 툴체인 | 특징 | 난이도 | 보드 |
|-----|--------|------|--------|------|
| **RK3588** (RKNN) | RKNN-Toolkit2 (Python) | INT8+FP16 mixed, 오픈소스, 6 TOPS | 쉬움 | Orange Pi 5, Rock 5B |
| **Jetson Orin** (TensorRT) | TensorRT (C++/Python) | INT8+FP16, 가장 문서화 잘 됨, 100+ TOPS | 중간 | Jetson Orin Nano |
| **Qualcomm** (QNN/Hexagon) | SNPE/QNN (C++) | 스마트폰 NPU, W8A8+W8A16 | 어려움 | Snapdragon Dev Kit |
| **ARM Ethos-U** | Vela compiler (Python) | MCU급, 극한 제약, 256KB SRAM | 어려움 | Corstone-300 FPGA |
| **MediaTek APU** | NeuroPilot (Android) | 스마트폰, 문서 적음 | 매우 어려움 | Dimensity Dev Kit |

### 3.5.5 각 NPU에서 배울 수 있는 것

**RK3588:** "mixed precision이 왜 중요한지" 체감. 같은 모델에서 INT8 only vs INT8+FP16 mixed의 차이를 직접 측정.

**TensorRT:** "calibration 알고리즘이 가장 발전한 환경" 경험. INT8 calibration의 교과서적 구현 (NVIDIA가 KL calibration을 대중화함). Profiling 도구가 가장 좋아서 레이어별 분석이 쉬움.

**QNN/Hexagon:** "실제 상용 제품(스마트폰)에서의 양자화" 경험. 모든 스마트폰 음성 비서가 여기서 돌아감. 가장 실무적.

**Ethos-U:** "극한 제약에서의 양자화" 경험. SRAM 256KB에서 모델을 돌려야 하는 상황. 양자화 + 모델 경량화 + tiling 최적화를 동시에 해야 함.

### 예상 소요: 2~4주 (하나만)
### 완료 기준: 처음 보는 NPU + 처음 보는 모델 조합에서 "될까? 안 되면 왜?" 를 30분 안에 초기 판단할 수 있을 것.

---

## Step 6: 자기만의 기법 만들기 (고수 진입)

### 3.6.1 언제
Step 1~5를 마친 후. 수학, 구현, 하드웨어, 다양한 NPU 경험이 있어야 자기 기법에 의미가 있음.

### 3.6.2 후보 아이디어

**아이디어 1: T527 디바이스 정합 FakeQuantize**

```
문제: Acuity 시뮬레이션과 T527 디바이스의 argmax 일치율이 31.5%
원인: Acuity 시뮬레이션은 이상적 uint8 연산을 가정하지만,
      실제 NPU는 tiling, accumulator overflow handling,
      activation function LUT 근사 등에서 미세한 차이 발생

접근:
  1. 같은 입력에 대해 Acuity 시뮬레이션 출력과 T527 디바이스 출력을 수백 개 수집
  2. 레이어별로 어디서 차이가 발생하는지 분석
  3. 차이 패턴을 모델링하여 FakeQuantize에 반영
  4. 이 "device-accurate FakeQuantize"로 QAT 학습
  5. 디바이스 정합률 31.5% → ?% 개선

기대 효과: QAT 학습이 실제 디바이스 동작에 더 가까워짐 → CER 개선
논문 가치: "Device-Accurate Quantization-Aware Training for Edge NPU" — 실용적 기여
```

**아이디어 2: STT 특화 Calibration Data Sampling**

```
문제: 양자화 calibration 데이터를 랜덤 샘플링하면 최적이 아닐 수 있음
관찰: 음성 데이터는 특성이 다양 — silence, onset, sustained, noise

접근:
  1. 음성 데이터를 특성별로 클러스터링 (에너지, pitch, SNR, duration)
  2. 각 클러스터에서 대표 샘플 추출 (stratified sampling)
  3. 이 대표 샘플로 calibration vs 랜덤 샘플 calibration 비교
  4. activation 분포의 coverage를 정량화하는 메트릭 정의

기대 효과: 적은 calibration 데이터(50개)로 랜덤 500개와 동등한 성능
논문 가치: "Representative Calibration Sampling for Speech Model Quantization"
```

**아이디어 3: 레이어별 Adaptive Quantization**

```
문제: 현재 .quantize 파일은 모든 레이어에 같은 quantizer를 적용
관찰: 레이어마다 양자화 민감도가 다름

접근:
  1. 레이어 하나씩 양자화하고 나머지는 FP32 → CER 측정 (sensitivity analysis)
  2. 민감한 레이어 ranking 생성
  3. T527 제약 하에서 최적의 mixed precision 배정:
     - 민감한 레이어: int16 DFP (정밀도↑, 속도↓)
     - 둔감한 레이어: uint8 (속도↑)
  4. .quantize 파일에 레이어별 다른 quantizer 배정
  5. CER vs 추론 속도 pareto front 그리기

기대 효과: uint8 전체보다 CER 개선되면서, int16 전체보다 빠름
논문 가치: "Layer-Adaptive Mixed-Precision Quantization for Edge STT"
```

**아이디어 4: Activation Range 예측기**

```
문제: 새 모델이 주어졌을 때 양자화 가능 여부를 판단하려면 실제로 돌려봐야 함
      → 시간 + 디바이스 필요

접근:
  1. 지금까지의 15+ 모델 실험 데이터를 학습 데이터로 사용
  2. ONNX 그래프의 구조적 특징 추출:
     - op 종류별 개수, 연결 패턴
     - 각 op의 weight 분포 통계
     - 그래프 깊이, 브랜칭 패턴
  3. 이 특징으로 "레이어별 activation range 예측" 또는 "양자화 성공/실패 분류"
  4. 새 모델을 돌려보기 전에 "이 모델은 uint8에서 CER ~X% 예상" 판단

기대 효과: 모델 선택 시간 단축 (2주 → 30분)
논문 가치: "Predicting Quantization Success from Neural Network Graph Structure"
```

### 3.6.3 완료 기준
기존 기법보다 **측정 가능한 개선**을 달성하고, 그 이유를 **수학적으로 설명**할 수 있을 것. 이 결과를 논문으로 쓸 수 있을 것.

---

# 4. 마일스톤 체크리스트

| # | 마일스톤 | 상태 | 예상 기간 | 완료 시 나는... |
|---|---------|------|----------|---------------|
| 1 | 양자화 수학 손계산 | [ ] | 2~3시간 | scale/zp/fl을 직접 계산할 수 있다 |
| 2 | 논문 3편 읽기 | [ ] | 1~2주 | 기법들의 수학적 근거를 설명할 수 있다 |
| 3 | 커스텀 PTQ 파이프라인 | [ ] | 2~3주 | Acuity 없이 .quantize를 만들 수 있다 |
| 4 | 하드웨어 이해 | [ ] | 1주 | NPU 동작 원리를 설명할 수 있다 |
| 5 | 다른 NPU 1개 경험 | [ ] | 2~4주 | NPU 비교 판단이 가능하다 |
| 6 | 자기만의 기법 | [ ] | 4~8주 | 양자화 고수다 |

**총 예상: 약 3~4개월** (회사 업무 병행 시)

---

# 5. 현실적 우선순위

지금 당장은 **QAT 학습이 급함** (회사 업무). 하지만 병행 가능:

### 이번 주
- QAT 학습 실행 (서버에서)
- **QAT 돌아가는 동안 Step 1 (수학 손계산)** — .quantize 파일 뜯으면서

### 다음 주
- QAT 결과 정리 + T527 테스트
- **논문 1편 시작 (Jacob 2018)** — 출퇴근 시간에 읽기

### 2~3주 후
- **커스텀 PTQ Phase 1 (activation 수집기)** 시작
- 논문 2편 (Nagel 2021)

### 1달 후
- 커스텀 PTQ 완성 + Acuity 결과와 비교
- **"내가 만든 양자화 vs Acuity 양자화" 첫 대결**

### 2달 후
- RK3588 실험 시작
- 논문 3편 (SmoothQuant)

### 3달 후
- 자기만의 기법 구상 + 초기 실험

**핵심: 학습 돌려놓고 노는 시간에 수학 하나씩 해보면 됨. 어차피 GPU 돌아가는 동안 할 거 없으니까.**

---

# 6. 참고 자료

### 필수 논문
1. Jacob et al. 2018 — "Quantization and Training of Neural Networks..." (arXiv:1712.05877)
2. Nagel et al. 2021 — "A White Paper on Neural Network Quantization" (arXiv:2106.08295)
3. Xiao et al. 2023 — "SmoothQuant" (arXiv:2211.10438)

### 추가 논문
4. Nagel et al. 2019 — "Data-Free Quantization Through Weight Equalization..." (arXiv:1906.04721)
5. Li et al. 2021 — "BRECQ: Pushing the Limit of PTQ" (arXiv:2102.05426)
6. Lin et al. 2024 — "AWQ: Activation-aware Weight Quantization" (arXiv:2306.00978)
7. Gulati et al. 2020 — "Conformer" (arXiv:2005.08100)

### 온라인 자료
- TensorRT Quantization Guide (NVIDIA) — calibration 알고리즘 설명이 가장 잘 되어있음
- ONNX Runtime Quantization Documentation — ONNX 모델 양자화 실습
- PyTorch Quantization Documentation — QAT 구현 참고

### 코드 참고
- `/home1/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom/sb_int16.quantize` — 실제 .quantize 파일 (13,227줄)
- `/home1/nsbb/travail/claude/T527/ai-sdk/scripts/pegasus_quantize.sh` — Acuity quantize 호출 스크립트
- `/home1/nsbb/travail/claude/T527/T527-STT/conformer/` — Conformer 양자화 실험 기록
