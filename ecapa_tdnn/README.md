# ECAPA-TDNN — 한국어 음성인식 (T527 NPU 후보)

## ECAPA-TDNN이란

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
= **1D CNN 기반** 음성 모델. Transformer 없음 → **uint8 양자화에 매우 유리**.

원래 화자 인식(Speaker Verification) 모델이지만, CTC head를 붙이면 **ASR(음성인식)에도 사용 가능**.

## 아키텍처

```
Input (mel spectrogram)
  ↓
1D Conv layers (TDNN blocks)
  ↓
SE-Res2Net blocks (Channel Attention)
  ↓
Attentive Statistics Pooling
  ↓
FC layer → CTC / Classification
```

### T527 양자화 적합성

| 연산 | 양자화 | ECAPA-TDNN | Transformer |
|------|--------|-----------|-------------|
| 1D Conv | **매우 좋음** | ✓ 핵심 연산 | 일부만 |
| Batch/Layer Norm | 보통 | ✓ | ✓ |
| Self-Attention | **나쁨** | ✗ 없음 | ✓ 핵심 |
| Softmax | **나쁨** | ✗ 없음 | ✓ |
| GELU | **나쁨** | ✗ 없음 | ✓ |
| ReLU | **좋음** | ✓ | 일부 |

**Self-Attention, Softmax, GELU가 없으므로 KoCitrinet처럼 uint8 양자화에 강할 것으로 예상.**

## KoCitrinet과의 비교

| | KoCitrinet | ECAPA-TDNN |
|---|---|---|
| 핵심 구조 | 1D Conv + SE | 1D Conv + SE + Res2Net |
| Attention | 없음 | **없음** (Channel Attention만) |
| 양자화 적합성 | **매우 좋음** | **매우 좋음** (유사 구조) |
| 한국어 ASR | NeMo 공식 지원 | 직접 학습 필요 |
| 파라미터 | ~10M (256) | ~6-20M |

## 한국어 모델 현황

HuggingFace/NeMo에 **한국어 ECAPA-TDNN ASR 모델은 없음**.
화자 인식 모델만 존재 (SpeechBrain 등).

### 활용 방법

1. **SpeechBrain ECAPA-TDNN** backbone 가져와서 한국어 CTC fine-tune
2. 또는 **KoCitrinet 개선에 집중** (이미 같은 CNN 계열이고 한국어 모델 있음)

## 결론

ECAPA-TDNN은 양자화에 유리하지만 **한국어 ASR 모델이 없어서 처음부터 학습해야 함**.
이미 KoCitrinet (같은 CNN 계열)이 CER 8.44%로 잘 동작하므로,
**Conformer를 먼저 시도하고, ECAPA-TDNN은 KoCitrinet 대안으로 보류**.

## 상태

**보류** — 한국어 ASR 학습 데이터/인프라 필요. Conformer 우선 시도.
