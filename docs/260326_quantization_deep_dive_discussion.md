# 양자화 Deep Dive 토론 기록

**날짜:** 2026-03-26
**참여:** nsbb, Claude Code (Opus 4.6)
**맥락:** T527 NPU uint8 STT 양자화 프로젝트, QAT 학습 준비 단계에서 진행된 기술 토론

---

# 1. 대화 기록

## 1.1 프로젝트 현황 확인

기존 종합 분석 문서(260325_t527_npu_stt_quantization_comprehensive_analysis.md, 77,000 토큰)를 다시 읽으며 프로젝트 현재 상태를 점검.

**현재 상태:**
- SungBeom Conformer uint8 PTQ → CER 10.02% (성공)
- FP32 대비 양자화 손실 약 0.66%p
- AIHub 데이터로 QAT 학습은 아직 미실행
- 서버에 AIHub 데이터 확인됨: `/nas04/nlp_sk/STT/data/train/` (base + noise, CSV 형식)

**QAT 학습을 위한 다음 단계:**
1. AIHub CSV → NeMo manifest JSON 변환
2. train/val 분리 (95/5)
3. `train_qat.py` 실행 (FakeQuantize + MarginLoss)
4. .nemo → ONNX → uint8 NB 변환
5. T527 디바이스 CER 측정

---

## 1.2 모델 찾은 건 누구인가

> **nsbb:** 사실 이 모델 찾은건 나임. 넌 못찾고 conformer 안됨 웅에웅 이랬음 ㅋ

Claude Code가 "Conformer 안 됨"이라고 판단했으나, nsbb가 직접 SungBeom Conformer를 찾아서 테스트한 결과 CER 10.02%로 성공. **모델 선택이 프로젝트 성패를 갈랐고, 그 판단은 사람이 한 것.**

---

## 1.3 LLM 4bit vs T527 uint8 — 왜 다른가

> **nsbb:** 내가 해본게 있긴함. llm은 4비트양자화도 잘되는데 우리는 8비트인데 왜케 안되는지 생각해보고 그거를 lm_head랑 접목시켜서 그럼 lm_head를 fp32로 자르면 w8a32가 되는거아니냐 이런거

**핵심 차이:**
- LLM: **W4A16** — weight만 4bit, activation은 FP16 유지
- T527: **W8A8** — weight도 activation도 전부 uint8
- activation까지 양자화하는 것이 결정적 차이. LLM에서도 activation 양자화하면 많이 깨짐

**Split Model 시도 결과:**
- encoder가 이미 uint8에서 망가진 feature를 출력 → lm_head를 FP32로 해도 "쓰레기 입력 → 쓰레기 출력"
- 사고 과정 자체는 올바른 접근 (RK3588은 FP16 HW 가속이 되므로 Split이 유효)

---

## 1.4 마지막 레이어를 FP32로 빼면?

> **nsbb:** 그럼 만약에 activation 앞부분에서 8비트짜리가 엄청 잘되었어(지금 컨모퍼처럼) 그럼 마지막레이어를 fp32로 그대로 빼버리면 더 잘되려나?

**결론: Conformer에서는 효과가 작다.**

- Conformer lm_head의 log softmax 출력: 정답 ~0.0, 오답 -20~-50 → 차이가 워낙 커서 uint8로도 argmax 보존됨
- 모델 전체를 int16으로 올려도 CER 10.02% → 9.59% (0.43%p 차이)
- lm_head만 FP32면 그보다 더 작을 것

**병목이 어디냐에 따라 다르다.** Conformer는 병목이 lm_head가 아님 → QAT로 encoder 자체를 강화하는 게 더 효과적.

---

## 1.5 양자화 친화적 아키텍처 판단 기준

> **nsbb:** 아키텍처 자체가 양자화 비친화적인지 친화적인지는 어케알지?

**근본 원인: activation range가 넓으면 깨진다.**

uint8은 256칸. range가 넓으면 한 칸(step)이 커지고, 미세한 차이가 뭉개짐.

| range를 넓히는 것 (나쁨) | range를 좁히는 것 (좋음) |
|---|---|
| Self-Attention (softmax 후 곱셈) | Depthwise Conv (local averaging) |
| GELU (급변하는 곡선) | Swish/SiLU (부드러운 곡선) |
| Residual Add 누적 | GLU gating (sigmoid로 값 제한) |
| raw waveform 입력 (unbounded) | mel spectrogram (bounded) |
| 긴 시퀀스 (attention 쌍 폭발) | subsampling으로 시퀀스 축소 |

**실전 확인법:**
1. ONNX 로드 → 각 레이어 output range 측정
2. range가 수백~수천이면 위험
3. 레이어 지날수록 range 커지면 → 누적 오차, 거의 확정 실패

**한 줄 요약: "activation range를 통제할 수 있는 구조인가?"**

---

## 1.6 int16이 왜 별 차이 없나 / uint8 vs int8

> **nsbb:** 근데 그럼 int16은 무조건 더 잘되어야 하는데 왜 int8이랑 거의 비슷했지? 그리고 uint8보다 int8이 두배 더 넓은거아닌가?

**int16이 별 차이 없는 이유:**
- Conformer는 이미 activation range가 좁아서 uint8 256칸으로 충분
- 65,536칸(int16)으로 늘려봤자 이미 잘 표현되고 있는 걸 더 세밀하게 하는 것
- 비유: 이미 선명한 사진을 더 높은 해상도로 올리는 것

**uint8 vs int8 — 흔한 오해:**
```
uint8:  [0 ---- 128 ---- 255]    256칸
int8:   [-128 ---- 0 ---- 127]   256칸
```
**둘 다 256칸.** 정밀도 동일. 영점(zero point) 위치만 다름. int8이 유리한 경우는 activation 분포가 0 중심 대칭일 때 — zero point 계산이 깔끔해짐.

---

## 1.7 Calibration 데이터의 중요성

> **nsbb:** 그럼 scale이랑 zp를 찾는게 calibration data set이니까 엄청 중요한 데이터셋이네. 이게 근데 만약에 내가 실제로 상품화 할때의 데이터셋이 있으면 그거에 맞춰서 하면 오버피팅하고 뭐가다름?

**Calibration ≠ 학습 오버피팅:**
- 학습: 가중치를 바꿈 → 특정 데이터에 맞춰지면 다른 데이터에서 안 됨
- Calibration: 가중치 안 바꿈. 각 레이어의 activation range(min/max)를 측정하는 것
- 목적: "이 레이어는 값이 -3~5 사이니까 256칸을 여기 배분하자"
- 특정 데이터에 "맞추는" 게 아니라 "범위를 파악하는" 것

**Calibration 데이터 개수별 효과:**

| 개수 | 효과 |
|------|------|
| 1개 | 위험. range 편향 (조용한 음성 1개면 range가 좁게 잡힘) |
| 100개 | 대부분 충분 |
| 1000개 | 안정적 |
| 10000개 | 시간 낭비, 효과 동일 |

**양보다 다양성이 중요.** 다양한 화자, 소음 환경, 문장 길이를 골고루 섞은 100~500개면 충분.

---

## 1.8 대표 Calibration 데이터 샘플링 (NEW INSIGHT)

> **nsbb:** 그럼 calibration data set 을 뽑아낼때 데이터의 distribution을 보고 전체 데이터셋을 잘 표현하는 대표 데이터를 뽑아내는것도 중요한 작업이겠네

**맞다.** 실제 연구되는 분야이기도 함.

**대표성 확보 기준:**
- duration 기준 — 짧은/중간/긴 음성 골고루
- 에너지/SNR 기준 — 조용한/보통/시끄러운 환경
- 화자 다양성 — 남/여/아동/노인
- 발화 내용 — 짧은 명령어 ~ 긴 문장

**단, 아키텍처가 좋으면 calibration 민감도가 낮아진다.** Conformer에서 KL vs Moving Average 차이가 0.06%p였던 것처럼 — 아키텍처가 좋으면 calibration을 막 뽑아도 잘 됨. 반대로 아키텍처가 나쁘면 calibration을 아무리 잘해도 안 됨.

---

## 1.9 Operator 교체로 양자화 개선

> **nsbb:** operator table 보고 해당 모델 layer 분석해서 안되는 레이어나 어려운 레이어를 쉬운 레이어로 표현을 바꿔가지고 하는 방법적용하는거?

**Graph optimization / Operator substitution:**
- GELU(erf, NPU 근사 부정확) → Swish(sigmoid, NPU 친화적)
- LayerNorm → BatchNorm (일부 NPU에서 더 잘 지원)
- 지원 안 되는 op → 지원되는 op 조합으로 분해

**이미 한 사례:** ONNX Pad op의 constant_value fix

**한계:**
- op 하나 바꾼다고 12개 레이어 누적 오차 해결 안 됨
- 수학적으로 동치인 변환만 가능 (GELU→Swish는 근사, 동치 아님)
- 작은 문제(op 1~2개 호환 안 됨) → 해결 가능
- 큰 문제(아키텍처 자체가 비친화적) → 모델을 바꿔야 함

---

## 1.10 양자화 기법 총정리

### PTQ (Post-Training Quantization)

| 기법 | 원리 | 특징 |
|------|------|------|
| MinMax | min/max로 range 설정 | 가장 단순, outlier에 취약 |
| KL Divergence | FP32↔uint8 정보 손실 최소화 | Acuity 기본값 |
| Percentile | 상하위 0.1% 잘라내고 range 설정 | outlier 무시 |
| MSE | FP32↔양자화 오차 최소화 | 수학적으로 깔끔 |

```
MinMax:      [============================]  넓음, step 큼
Percentile:  [--======================--]    좁음, step 작음, 양끝 잘림
KL:          [---=====================---]   정보 손실 최소 지점
```

### QAT (Quantization-Aware Training)

| 기법 | 원리 |
|------|------|
| Standard QAT | FakeQuantize 삽입, STE로 gradient 통과 |
| LSQ | step size 자체를 학습 가능한 파라미터로 |
| PACT | clipping 범위를 학습으로 찾음 |

### 가중치 쪽 기법 (주로 LLM)

| 기법 | 원리 |
|------|------|
| GPTQ | 레이어별 양자화 오차를 다른 가중치로 보상 |
| AWQ | activation 큰 채널의 weight 정밀 보존 |
| SmoothQuant | activation outlier를 weight 쪽으로 이전 |

### 적용 단위 (Granularity)

```
Per-tensor:   텐서 전체에 scale 1개  → 거칠지만 빠름
Per-channel:  채널마다 scale 1개     → 정밀, 대부분의 NPU 지원
Per-group:    n개씩 묶어서 scale 1개 → LLM에서 주로 사용
```

---

## 1.11 커스텀 양자화 기법 만들기

> **nsbb:** ptq나 qat 중에 양자화 기법을 새로 만들어볼수는 없나 우리가

**이미 하나 만들었다:** `train_qat.py`의 MarginLoss는 표준 QAT가 아님.
```
표준 QAT:  CTC loss + FakeQuantize
우리 QAT:  CTC loss + FakeQuantize + MarginLoss (top1-top2 차이 강제)
```

**더 만들 수 있는 방향:**
1. **FakeQuantize 개선** — Acuity 시뮬레이션과 디바이스가 31.5%만 일치. T527 실제 동작을 더 정확히 모사하는 FakeQuantize?
2. **레이어별 sensitivity 기반 QAT** — 깨지기 쉬운 레이어에만 집중적으로 FakeQuantize
3. **커스텀 Calibration 알고리즘** — STT 특성에 맞는 range 결정 방법

**현실적 제약:** 최종 배포는 Acuity 통해야 함. PTQ 알고리즘은 Acuity가 닫혀있어 커스텀 어려움. **QAT 쪽은 학습 스크립트가 우리 것이므로 자유.**

---

## 1.12 Acuity 파이프라인 분석 — PTQ도 커스텀 가능 (NEW INSIGHT)

> **nsbb:** ptq도 그냥 우리가 짜서 못함? 리버싱해가지고 열어볼수없나? 개빠깇노

> **nsbb:** 그거 되는거같은데 얘가 .quantize 라는 파일 읽어서 각 레이어마다 weight 어케줄지 보고 또 .data라는 파일이 실제 weight 파일이라서 그거가지고 종합해서 nb 만들던데

**발견: .quantize 파일이 YAML 형식으로 열려있다.**

```yaml
# sb_int16.quantize (13,227줄)
'@Conv_/decoder_layers/decoder_layers.0/Conv_3:out0':
    qtype: i16
    quantizer: dynamic_fixed_point
    rounding: rtne
    max_value: 16.725765228271484
    min_value: -40.68741989135742
    fl: 9
```

각 레이어의 qtype, quantizer, min/max, fractional length를 **직접 편집 가능.**

**Acuity 파이프라인:**
```
import:    ONNX → .json + .data (모델 구조 + weight)
quantize:  .json + .data + calibration → .quantize (양자화 파라미터)
export:    .json + .data + .quantize → .nb (NPU 바이너리)
```

**커스텀 가능한 부분:**

| 단계 | Acuity 필요? | 우리가 대체 가능? |
|------|-------------|-----------------|
| import | O | ONNX → NPU 내부 IR 변환은 Acuity만 가능 |
| **quantize** | **X** | **가능!** .quantize가 YAML이므로 직접 생성 |
| export | O | NB 바이너리 포맷은 Acuity만 가능 |

**의미:**
1. PyTorch/ONNX로 모델 로드
2. calibration 데이터로 레이어별 activation 분포 수집
3. **커스텀 알고리즘**으로 최적 min/max 계산
4. `.quantize` YAML 생성
5. Acuity export에 넘겨서 NB 생성

import와 export만 Acuity 쓰고, **핵심인 양자화 로직은 우리 것으로 대체 가능.**
SmoothQuant, 레이어별 sensitivity 기반, STT 특화 calibration 등 무한 실험 가능.

---

# 2. 새로 발견된 인사이트 정리

## Insight 1: Calibration 데이터는 양보다 다양성
- 100~500개면 충분, 10,000개는 시간 낭비
- 화자, 소음, 길이 등 다양성이 핵심
- 전체 데이터셋의 distribution을 대표하는 샘플링 전략이 중요
- 아키텍처가 좋으면 calibration 민감도가 낮아짐

## Insight 2: Calibration ≠ 학습 오버피팅
- Calibration은 가중치를 바꾸지 않고 activation range만 측정
- "맞추는" 게 아니라 "범위를 파악하는" 것
- 따라서 production 데이터로 calibration해도 오버피팅 개념과는 다름

## Insight 3: Acuity .quantize 파일로 커스텀 PTQ 가능
- .quantize 파일이 YAML 형식, 13,227줄, 레이어별 양자화 파라미터
- 직접 편집하여 Acuity의 PTQ 알고리즘을 우회할 수 있음
- import/export만 Acuity, 양자화 로직은 완전히 커스텀 가능

## Insight 4: 양자화 친화성 판단 = activation range 통제 가능 여부
- 핵심 질문: "이 구조가 activation range를 넓히냐 좁히냐?"
- 실전 확인: 레이어별 output range 측정, 누적 증가 여부 확인
- CNN이 매 레이어에서 range를 좁히는 역할 → Conformer 성공 원인

## Insight 5: uint8 vs int8은 정밀도가 같다
- 둘 다 256칸. 영점 위치만 다름
- int8: 0 중심 대칭 분포에 유리 (zero point 계산 깔끔)
- 하지만 극적인 차이 없음

---

# 3. 향후 작업

## 즉시 (QAT 학습)
1. AIHub CSV → NeMo manifest JSON 변환
2. train/val 분리
3. `train_qat.py` 실행 (서버에서)
4. QAT 모델 → ONNX → uint8 NB → T527 CER 측정
5. 목표: CER 10.02% → 9% 대 진입

## 중기 (커스텀 양자화 프레임워크)
1. .quantize YAML 구조 완전 분석
2. PyTorch 기반 커스텀 calibration 파이프라인 구축
3. 레이어별 sensitivity 분석 → 최적 min/max 자동 탐색
4. Acuity quantize 단계 대체 → 커스텀 .quantize 생성 → NB export
5. 다양한 calibration 알고리즘 실험 (SmoothQuant 등)

## 장기 (전문성 확보)
- 온디바이스 양자화 전문성: 아키텍처 분석 → 양자화 가능성 판단 → 커스텀 최적화
- 논문 가능성: Edge NPU 양자화 커스터마이징, STT 특화 QAT 기법
