# Wav2Vec2 Base Korean — 한국어 (T527 NPU)

Kkonjeong/wav2vec2-base-korean. 한국어 STT, T527 NPU 양자화. **전부 실패.**

## 결과: 60종+ 시도, 21종 NPU 실측, 전부 실패

| 카테고리 | 시도 수 | 결과 |
|---------|--------|------|
| uint8 동작, garbled 출력 | 12종 | ALL PAD 또는 ㅇ 토큰 과다 |
| int16/DFP, NPU status=-1 | 3종 | NB 생성 성공, **NPU 실행 거부** (HANG 아님) |
| NPU status=-1 (실행 거부) | 4종 | CNN-only, split, combo 등 |
| NB 생성 실패 | 6종 | bf16, PCQ, symmetric 등 |
| 시뮬레이션만 | 10종+ | SmoothQuant v1-v3, AttnClip, RangeClip 등 |

## 모델 스펙

| 항목 | 값 |
|------|-----|
| 원본 | Kkonjeong/wav2vec2-base-korean (HuggingFace) |
| 파라미터 | 94.4M, 12 Transformer layers |
| NB 크기 | 72MB (uint8) / 153MB (int16) |
| 입력 | `[1, 48000]` raw waveform (3초, 16kHz) |
| Vocab | 56 (한글 자모 분리) |
| 추론시간 | 425ms / 3초 (uint8, 동작하지만 출력 쓰레기) |
| RTF | 0.142 |

## 실패 원인 분석 (심층)

### 핵심: 한국어 모델의 activation 동적 범위가 영어 대비 2-5배 넓음

Pegasus 시뮬레이션에서 layer-by-layer 분석 결과:

| 지표 | 한국어 모델 | 영어 모델 | 비율 |
|------|-----------|----------|------|
| attention MatMul avg range | 37.55 | 19.48 | **1.9x** |
| attention MatMul max range | 276.7 | 49.1 (L11=1050) | — |
| softmax avg range | 0.82 | 0.03 | **27x** |
| FFN output range | 136-254 | 30-69 | **3-4x** |
| CNN Div range | 454-960 | 129-192 | **3-5x** |
| uint8 scale > 0.5인 텐서 수 | 25개 | 3개 | — |
| Pegasus uint8 argmax agree (vs FP32) | **46.3%** | ~85% | — |

### 왜 한국어가 더 넓은가?

1. **attention 패턴 차이**: 영어 모델은 특정 위치에 sharp attention (near one-hot) → 좁은 softmax range. 한국어는 distributed attention (여러 위치에 분산) → 넓은 softmax range
2. **Q@K^T 범위**: 한국어 모든 layer에서 138-277 (영어는 대부분 < 50)
3. **k_proj bias**: 영어 L11 mean_abs=15.11, 한국어 mean_abs=0.14 — 그러나 이것은 원인이 아닌 상관관계 (softmax가 shift-invariant이므로 bias는 attention에 영향 없음)

### 모델 수정 시도 (전부 실패)

| 방법 | FP32 영향 | uint8 영향 | 결론 |
|------|----------|-----------|------|
| k_proj bias 이식 (영어→한국어) | 변화 없음 | — | softmax shift-invariance |
| k_proj bias ×100 | 변화 없음 | — | 같은 이유 |
| K weight ×0.25 (SmoothQuant식) | gap 2.09→0.82 | — | FP32 파괴 |
| Temperature T=4 (Q/K ×0.5) | gap 3.22→1.02 | — | FP32 파괴 |
| Temperature T=16 (Q/K ×0.25) | gap 3.22→0.67 | — | FP32 파괴 |
| Weight clipping (99.99th percentile) | gap 2.09→0.71 | — | 0.02%만 clip해도 FP32 파괴 |
| SmoothQuant v1 (α=0.3-0.9) | gap→0.06 | — | **완전 파괴 (잔차 연결 버그)** |
| **SmoothQuant v3 (α=0.3-0.9)** | **100% 보존** | ORT uint8: 5.4% | **FP32 최초 보존 성공, uint8 개선 없음** |
| SQv3 + AttnClip30 결합 | 81.9% agree | ORT uint8: 3.4% | 결합해도 개선 없음 |
| Q/K vector clamp=3.0 | gap 2.09→1.94 | — | FP32 보존하지만... |
| **Pegasus AttnClip100** | — | **agree 40.9%** | **원본(46.3%)보다 악화** |
| **Pegasus AttnClip50** | — | **agree 34.9%** | **더 악화** |
| **Pegasus RangeClip200** | — | **agree 11.4%** | **완전 파괴** |

**결정적 발견**: attention score의 wide range는 **노이즈가 아니라 필수 신호**. 범위를 줄이면 outlier가 saturation되어 정보 손실이 resolution 개선보다 큼.

### SmoothQuant 심층 분석

**v1/v2 실패 원인**: wav2vec2의 잔차 연결 구조가 일반적인 pre-LN Transformer와 다름.

```
일반 pre-LN Transformer:
  x → LayerNorm(x) → Q/K/V → Attention → out_proj
  x ─────────────────────────────────────────→ Add(x, out_proj)
  (잔차는 LN 이전의 x를 사용)

wav2vec2 (Kkonjeong/wav2vec2-base-korean):
  x → LayerNorm(x) → Q/K/V → Attention → out_proj
  x → LayerNorm(x) ─────────────────────────→ Add(LN(x), out_proj)
  (잔차가 LN 출력을 직접 사용!)
```

SmoothQuant v1/v2는 LayerNorm γ/β를 `/s`로 수정 → LN 출력 전체가 `/s` → 잔차 경로도 `/s`되어 보상 없이 FP32 파괴.

**v3 해결**: γ/β를 수정하지 않고 MatMul 입력 전에만 Div(s) 노드 삽입. 잔차 경로는 원래 LN 출력 사용 → FP32 100% 보존.

**v3 한계**: SmoothQuant은 Linear layer 입력의 activation range만 줄임. 모델 전체의 양자화 병목은:
1. attention Q@K^T score (range 138-277) — Linear이 아닌 MatMul
2. softmax 출력 (range 0.82) — 비선형 연산
3. CNN feature extractor (range 454-960) — Conv 연산

이들은 SmoothQuant의 적용 대상이 아님 → uint8 전체 양자화에서 개선 미미 (4.7% → 5.4%).

### 기존 분석 (유지)

1. **uint8 양자화 = argmax agreement 46%** — 12-layer Transformer에서 오류 누적
2. **int16 DFP = Pegasus 시뮬에서 한국어 출력 보존** — 그러나 T527 NPU가 **status=-1** (실행 거부, HANG 아님)
3. **CNN-only/split/combo = status=-1** — NPU가 소형/수정 모델 실행 거부

### int16 DFP 검증 결과 (2026-03-16)

- NB 변환: **성공** (153MB, `wav2vec2_ko_base_3s_nopad10_opset12_sim_int16_nbg_unify/`)
- Pegasus int16 시뮬레이션: 한국어 출력 `ㄸㅓㄴ ㅌㅔ ㅇㅣ ㅂㅇㅡㄹ` (FP32과 유사)
- 양자화 파라미터: 입력 i16 fl=15 (×32768), 출력 i16 fl=11 (×2048)
- **디바이스 실행: `fail to run network, status=-1`** — NPU가 int16 DFP 모델 실행을 거부
- 앱에서 PAD만 출력: status=-1 → 출력 버퍼 초기화 안 됨 → 0 = PAD 토큰

## Acuity Toolkit 지원 데이터 타입

| 데이터 타입 | Pegasus quantizer | NB 크기 | 정밀도 | 비고 |
|------------|-------------------|---------|--------|------|
| **uint8** | `asymmetric_affine` | ~72MB | 256단계 | T527 NPU 실행 가능. 단, 이 모델은 양자화 열화로 쓰레기 출력 |
| **int8 (PCQ)** | `perchannel_symmetric_affine` | ~72MB | 채널별 256단계 | NB 생성 실패 (Reshape tensor error / segfault) |
| **int16 (DFP)** | `dynamic_fixed_point` | ~153MB | 65536단계 | NB 생성 성공, **T527 NPU가 실행 거부** (status=-1) |
| **bf16** | `qbfloat16` | ~181MB | bfloat16 | NB 생성 실패 (gen_nbg segfault, 0 bytes) |
| **fp16** | 없음 (`export --dtype float`) | ~182MB | IEEE float16 | **NB 생성 성공 (182MB). 디바이스 미검증** |
| **fp32** | 없음 | ~362MB | float32 | SRAM 부족으로 NPU 실행 불가 |

### 각 타입별 calibration/알고리즘 조합 시도

#### uint8 (asymmetric_affine) — 12종 NB, 전부 garbled

| # | calibration | 알고리즘 | ONNX 변형 | NB 생성 | NPU 실행 | 출력 |
|---|-------------|---------|----------|---------|---------|------|
| 1 | 기본 (1 sample) | default | 원본 | O (72MB) | O | ALL PAD |
| 2 | 50 samples | moving_average (w=0.004) | 원본 | O (73MB) | O | ㅇ 토큰 과다 |
| 3 | 50 samples | moving_average v2 | 원본 | O (73MB) | O | garbled |
| 4 | 50 samples | KL divergence | 원본 | O (73MB) | O | garbled |
| 5 | 50 samples | entropy | 원본 | O (73MB) | O | garbled |
| 6 | 50 samples | moving_average | nopad10 | O (73MB) | O | garbled |
| 7 | 50 samples | KL divergence | nopad10 | O (73MB) | O | garbled |
| 8 | 50 samples | moving_average | opset12+sim | O (72MB) | O | garbled |
| 9 | 50 samples | default | opset12+sim | O (72MB) | O | garbled |
| 10 | 50 samples | moving_average | 2L pruned | O (15MB) | status=-1 | — |
| 11 | 50 samples | moving_average | 6L pruned | O (39MB) | status=-1 | — |
| 12 | 50 samples | moving_average | 6L+ReLU | O (39MB) | status=-1 | — |

#### int8 PCQ (perchannel_symmetric_affine) — NB 생성 실패

| # | calibration | ONNX 변형 | 결과 |
|---|-------------|----------|------|
| 1 | 50 samples | 원본 | Reshape tensor error → segfault |
| 2 | 200 samples | nopad5 | .quantize 생성, gen_nbg segfault |
| 3 | 200 samples | nopad8 | .quantize 생성, gen_nbg segfault |

> PCQ는 per-channel scale/zp로 정밀도가 높지만, Acuity gen_nbg가 wav2vec2 구조에서 크래시.

#### int16 DFP (dynamic_fixed_point) — NB 성공, NPU 거부

| # | ONNX 변형 | NB 크기 | NPU 결과 | Pegasus 시뮬 |
|---|----------|---------|---------|-------------|
| 1 | 원본 | 72MB | status=-1 | — |
| 2 | nopad10+opset12+sim | 153MB | status=-1 | **한국어 정상 출력** |
| 3 | clip3s+nopad10+opset12+sim | 175MB | status=-1 | — |

> Pegasus FP32 시뮬에서 int16은 `ㄸㅓㄴ ㅌㅔ ㅇㅣ ㅂㅇㅡㄹ` 등 정상 출력. T527 NPU VIPLite 드라이버가 int16 DFP 실행을 지원하지 않는 것으로 추정.

#### bf16 (qbfloat16) — NB 생성 실패

| # | 도구 버전 | 결과 |
|---|----------|------|
| 1 | Acuity 6.12 | .quantize 성공 (35KB), gen_nbg segfault → NB 0 bytes |
| 2 | Acuity 6.21 | .quantize 성공, gen_nbg segfault → NB 0 bytes |

#### fp16 (export --dtype float) — **미검증, NB 존재**

| # | 방법 | NB 크기 | I/O dtype | 디바이스 테스트 |
|---|------|---------|----------|-------------|
| 1 | FP32 모델 → `export --dtype float` | **182MB** | float16 in/out | **미수행** |
| 2 | uint8 quantize → `export --dtype float` | **182MB** | float16 in/out | **미수행** |

> fp16은 양자화가 아니라 FP32→FP16 다운캐스트. 정밀도 손실이 최소. **T527 NPU에서 실행 가능한지 확인 필요.**

#### 구조 변경 시도

| 방법 | NB 생성 | NPU 결과 |
|------|---------|---------|
| CNN-only (feature extractor만) | O | status=-1 |
| split: CNN(uint8) → Transformer(int16) | O | Part B HANG |
| hybrid `--hybrid` (CNN uint8 + Transformer int16) | O (72MB) | HANG |
| combo (ReLU + 6L + nopad) | O | status=-1 |

#### Post-training 모델 수정 시도 (Pegasus 시뮬레이션)

| 방법 | FP32 영향 | uint8 영향 | 비고 |
|------|----------|-----------|------|
| SmoothQuant v1 (γ/β 수정) | 파괴 | — | 잔차 연결 버그 |
| SmoothQuant v3 (Div 노드) | **100% 보존** | 개선 없음 | FP32는 최초 성공, uint8은 미미 |
| SQv3 + AttnClip30 | 81.9% agree | 3.4% | 결합해도 개선 없음 |
| temperature scaling (T=4,16) | 파괴 | — | gap 축소되나 FP32 손상 |
| weight clipping (99.99~99.5%) | 파괴 | — | 0.02%만 clip해도 FP32 파괴 |
| k_proj bias 이식/스케일 | 없음 | — | softmax shift-invariance |
| AttnClip100 | — | 40.9% | 원본(46.3%)보다 악화 |
| AttnClip50 | — | 34.9% | 더 악화 |
| RangeClip200 | — | 11.4% | 완전 파괴 |
| Q/K vector clamping | 열화 | — | FP32 보존 안 됨 |

## 파일 구조

```
base-korean/
├── README.md
├── download_and_convert.py       # HuggingFace → ONNX 변환
├── decode_ko_output.py           # NPU 출력 → 한국어 텍스트
├── prepare_ko_test_input.py      # 테스트 입력 준비
├── create_cnn_only_model.py      # CNN-only ONNX 추출
├── test_all_nbs.sh               # 전체 NB 일괄 테스트
├── test_split_model.sh           # CNN(uint8) → Transformer(int16) 분리 테스트
├── test_priority_nbs.sh          # 우선순위 NB 테스트
├── auto_test_on_connect.sh       # 디바이스 연결 시 자동 테스트
├── vocab.json                    # 56 자모 vocab
└── config.json                   # HuggingFace 모델 config
```

## 도메인 미스매치 검증 (2026-03-17)

NPU 양자화 실패와 별개로, ONNX FP32 모델 자체의 성능을 실제 월패드 음성으로 검증.

### 월패드 테스트셋 결과 (ONNX FP32, 양자화 없음)

| 테스트셋 | 샘플 수 | CER |
|---------|--------|-----|
| 7F_KSK | 108 | **140.1%** |
| modelhouse_2m_noheater | 51 | **132.1%** |
| modelhouse_2m | 51 | **184.0%** |
| 7F_HJY | 107 | **153.8%** |
| modelhouse_3m | 51 | **209.7%** |
| worst30 (전체) | 330 | **168.5%** |

> CER > 100%는 모델이 정답보다 더 많은 글자를 생성함을 의미 (과다 삽입).

### Zeroth-Korean 테스트셋 결과 (학습 도메인)

동일 모델을 학습 데이터와 같은 도메인(Zeroth-Korean test, 457개 낭독체)으로 테스트.
PyTorch FP32, 가변 길이 입력.

| 항목 | 결과 |
|------|------|
| 평균 CER | **9.5%** (100개 샘플) |
| CER < 10% | 66/100 |
| CER = 0% (완벽) | 10/100 |

→ 학습 도메인에서는 정상 동작. **NPU 양자화 실패와 도메인 미스매치는 독립적인 두 문제.**

### 실패 원인: 도메인 미스매치

| 항목 | 학습 데이터 (Zeroth-Korean) | 월패드 테스트 데이터 |
|------|---------------------------|-------------------|
| **데이터셋** | Zeroth-Korean (51시간) | 실제 월패드 녹음 |
| **음성 유형** | 뉴스/책 낭독 (6~20초) | 짧은 명령 (1~3초) |
| **녹음 환경** | 조용한 스튜디오 | 실내 반향 + 생활소음 |
| **어휘** | 일반 한국어 (뉴스, 소설) | 도메인 특화 (세대소독, 알림음, 가스사용량) |
| **화자** | 105명 낭독자 | 다양한 일반인 |
| **Base 모델** | facebook/wav2vec2-base (영어 960시간 pretrain) | — |

### Fine-tuning 방안

모델 아키텍처는 유효함 (Zeroth-Korean CER 9.5%). 월패드 데이터로 fine-tuning하면 개선 가능.

```
[Option A] 월패드 데이터만으로 추가 fine-tuning
  - Kkonjeong/wav2vec2-base-korean 체크포인트에서 시작
  - 월패드 녹음 데이터로 추가 학습 (5~20 epoch)
  - 장점: 빠름, 94.4M params로 T527 NPU uint8 배포 가능성 있음
  - 단점: 일반 한국어 성능 저하 가능 (catastrophic forgetting)

[Option B] Zeroth-Korean + 월패드 혼합 학습
  - facebook/wav2vec2-base 또는 Kkonjeong 체크포인트에서 시작
  - Zeroth-Korean 51시간 + 월패드 데이터 혼합
  - 장점: 일반 + 도메인 성능 균형
  - 단점: 학습 시간 증가
```

> **주의**: fine-tuning으로 ONNX FP32 성능이 개선되더라도, T527 NPU uint8 양자화에서 동작할 보장은 없음. 영어 모델이 uint8에서 성공한 이유는 activation range가 좁았기 때문이며, 한국어 모델의 wide activation range(본 문서 "실패 원인 분석" 참조)가 fine-tuning으로 개선되는지 확인 필요.

---

## 결론

### 1. NPU 양자화: T527 uint8로는 동작 불가능

- uint8만 NPU에서 실행 가능하나 한국어 모델은 양자화 열화로 출력이 파괴됨
- int8 PCQ: gen_nbg segfault (NB 생성 불가)
- int16 DFP: NB 생성 성공, NPU status=-1 (실행 거부)
- bf16: gen_nbg segfault (NB 0 bytes)
- **fp16: NB 생성 성공 (182MB), 디바이스 미검증** ← 유일한 미확인 경로
- fp32: SRAM 부족 (362MB)
- 60종+ 양자화 시도, 21종 NPU 실측, 전부 실패

### 2. 도메인 미스매치: FP32에서도 월패드 음성 인식 불가

- Zeroth-Korean(낭독체) CER 9.5% vs 월패드 CER 132~210%
- 학습 데이터(51시간 낭독체)와 타겟 도메인(월패드 명령어)의 완전한 불일치

### 3. 해결 경로

1. **fp16 NB 디바이스 검증** → 182MB NB가 T527에서 실행되는지 확인 (미검증 상태)
2. fp16 성공 시 → Zeroth-Korean 데이터로 CER 측정 → 월패드 fine-tuning → 배포
3. fp16 실패 시 → **월패드 데이터로 fine-tuning** → uint8 양자화 재검증 (activation range 변화 확인)
4. 모두 실패 시 → CNN 기반 모델(KoCitrinet)이 유일한 대안

### 시도한 모든 post-training 기법이 실패한 이유

uint8 per-tensor 양자화의 한계는 **모델 전체에 분산된 wide activation range** 때문:

| 병목 위치 | 범위 | SmoothQuant | Clip | 결과 |
|----------|------|-------------|------|------|
| LayerNorm → Linear 입력 | 2-58 | ✓ 축소 가능 | — | uint8 개선 미미 |
| Q@K^T attention score | 138-277 | ✗ 적용 불가 | ✗ 정보 손실 | 개선 불가 |
| Softmax 출력 | 0.82 | ✗ 비선형 | — | 개선 불가 |
| CNN feature extractor | 454-960 | ✗ Conv | — | 개선 불가 |

→ SmoothQuant는 병목 중 하나만 해결. 나머지가 여전히 uint8을 파괴.
→ 해결 가능 경로: (1) fp16 NB 실행 확인, (2) QAT(학습 기반), (3) int16/bf16 NPU 하드웨어.
