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

## 시도한 양자화 방식

| 방식 | NPU 결과 |
|------|---------|
| uint8 asymmetric_affine | garbled (ALL PAD) |
| uint8 + moving_average | garbled |
| uint8 + KL divergence | garbled |
| uint8 + entropy | garbled |
| int8 symmetric | NB 생성 실패 |
| int16 DFP | **NPU status=-1** (실행 거부, NB 생성은 성공) |
| bf16 | NB 생성 실패 (segfault) |
| PCQ (per-channel) | NB 생성 실패 |
| hybrid (CNN uint8 + Transformer int16) | NPU HANG |
| opset12 + onnxsim | garbled (입력 범위 수정됨, 출력 여전히 실패) |
| CNN-only (feature extractor만) | NPU status=-1 |
| split (CNN uint8 → Transformer int16) | Part B NPU HANG |
| layer pruning (6L, 8L, 10L) | 시뮬레이션만, NPU status=-1 |
| SmoothQuant v1 (γ/β 수정) | FP32 파괴 (잔차 연결 버그) |
| **SmoothQuant v3 (Div노드)** | **FP32 보존, uint8 개선 없음** |
| temperature scaling (T=4, T=16) | FP32 파괴 (gap 3→1, ㅇ 과다) |
| weight clipping (99.99/99.9/99.5%) | FP32 파괴 |
| k_proj bias 이식/스케일 | 효과 없음 (softmax invariance) |
| attention score range clip | 시뮬레이션에서 **원본보다 악화** |
| Q/K vector clamping | FP32 열화 |
| combo (relu + 6L + nopad) | NPU status=-1 |

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

## 결론

T527 NPU에서 한국어 Wav2Vec2 (Transformer 기반)는 **동작 불가능**.
- uint8만 NPU에서 실행 가능하나 한국어 모델은 양자화 열화로 출력이 파괴됨
- int16/bf16은 NPU 하드웨어 미지원
- CNN 기반 모델(KoCitrinet)만이 유일한 대안

### 시도한 모든 post-training 기법이 실패한 이유

uint8 per-tensor 양자화의 한계는 **모델 전체에 분산된 wide activation range** 때문:

| 병목 위치 | 범위 | SmoothQuant | Clip | 결과 |
|----------|------|-------------|------|------|
| LayerNorm → Linear 입력 | 2-58 | ✓ 축소 가능 | — | uint8 개선 미미 |
| Q@K^T attention score | 138-277 | ✗ 적용 불가 | ✗ 정보 손실 | 개선 불가 |
| Softmax 출력 | 0.82 | ✗ 비선형 | — | 개선 불가 |
| CNN feature extractor | 454-960 | ✗ Conv | — | 개선 불가 |

→ SmoothQuant는 병목 중 하나만 해결. 나머지가 여전히 uint8을 파괴.
→ 해결 가능한 유일한 경로: QAT(학습 기반) 또는 int16/bf16 NPU 하드웨어.
