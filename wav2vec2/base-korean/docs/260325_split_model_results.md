# Split Model 실험 결과: Encoder(NPU uint8) + lm_head(CPU float32)

**날짜:** 2026-03-25
**상태:** 실험 완료 — **한국어 모델에서는 효과 없음 확정**
**목적:** T527 NPU의 W8A8 강제 제약을 우회하여, 마지막 레이어(lm_head)만 CPU float32로 실행

---

## 1. 핵심 아이디어

T527 NPU는 weight+activation 둘 다 uint8 강제(W8A8). lm_head의 logit이 uint8로 깎이면 argmax가 뒤집힘.
**모델을 둘로 쪼개서** encoder는 NPU uint8, lm_head는 CPU float32로 실행 → logit 정밀도 보존.

```
기존 (전부 NPU uint8):
  audio → [encoder + lm_head] → logits(uint8) → argmax ✗

Split:
  audio → [encoder](NPU uint8) → hidden(uint8) → dequantize → [lm_head](CPU fp32) → logits(fp32) → argmax ✓
```

OpenVINO NNCF도 wav2vec2에서 conv_layers 3개를 FP32로 복원하여 WER 회복 — 같은 원리.

---

## 2. 전체 실험 결과

### 2.1 영어 모델 (검증용)

**모델:** facebook/wav2vec2-base-960h (vocab 32)

| 방식 | CER | NPU 시간 | 결론 |
|------|-----|---------|------|
| ONNX FP32 | 9.74% | 서버 | 최상 |
| **Full uint8** | **17.52%** | **715ms** | **동작** |
| Split (enc uint8 + lm fp32) | 20.68% | 715ms + 5ms | full보다 약간 나쁨 |

→ 영어에서는 full uint8이 이미 잘 되므로 **Split 불필요**. Split이 약간 나쁜 이유: encoder 출력을 uint8→fp32 dequantize하면서 정밀도 손실.

### 2.2 한국어 base-korean (vocab 56, Kkonjeong)

**모델:** Kkonjeong/wav2vec2-base-korean (vocab 56, 자모)

| 방식 | CER | NPU 시간 | 결론 |
|------|-----|---------|------|
| ONNX FP32 | 30.22% (3s) | 서버 75ms | 서버 전용 |
| Full uint8 | 100.86% | 415ms | 완전 실패 |
| Split lm_head fp32 | 99.70% | 415ms + 5ms | **효과 없음** |
| Split L7 (L0-7 NPU + L8-11+lm fp32) | 99.26% | 320ms + CPU | **효과 없음** |
| CNN+lm fp32 (OpenVINO 방식) | 100.00% (전부 blank) | 285ms + 34ms | **오히려 악화** |

### 2.3 한국어 aihub 80k (vocab 1912, 음절)

**모델:** aihub 데이터 80000 step 학습 (vocab 1912, FP32 CER 9~18%)

| 방식 | CER | NPU 시간 | 결론 |
|------|-----|---------|------|
| ONNX FP32 | 9~18% (테스트셋별) | 서버 | 매우 좋음 |
| Full uint8 | 92.83% (3s) / 98.21% (full) | 415ms | 실패 |
| **Split lm_head fp32** | **92.65%** | **425ms + matmul** | **효과 없음** |

→ **FP32에서 CER 9~18%인 좋은 모델도 encoder uint8에서 깨짐**

---

## 3. 종합 비교 테이블

| 모델 | vocab | FP32 CER | Full uint8 CER | Split CER | 결론 |
|------|-------|---------|----------------|-----------|------|
| 영어 base-960h | 32 | 9.74% | **17.52%** | 20.68% | uint8 동작, Split 불필요 |
| 한국어 base-korean | 56 | 30.22% | 100.86% | 99.70% | 실패 |
| 한국어 base-korean | 56 | — | — | 99.26% (Split L7) | 실패 |
| 한국어 base-korean | 56 | — | — | 100.00% (CNN fp32) | 악화 |
| **한국어 aihub 80k** | **1912** | **9~18%** | **92.83%** | **92.65%** | **실패** |
| 한국어 fine-tune (attempt5) | 56 | WER 40.6% | 부분 성공 | — | **유일한 성공** |

---

## 4. 왜 안 되는가

### 4.1 근본 원인: Encoder uint8 양자화 품질

```
영어 encoder uint8:  hidden states cos(NPU, FP32) ≈ 0.97  → 정보 충분
한국어 encoder uint8: hidden states cos(NPU, FP32) ≈ 0.66  → 정보 손실 심각
```

lm_head를 fp32로 빼도 **입력(hidden states)이 이미 망가져 있으면 복구 불가**.

### 4.2 왜 한국어 encoder가 uint8에서 더 나쁜가

레이어별 분석 (530개 레이어 FP32 vs uint8 dump):

| Layer | 한국어 avg cos | 영어 avg cos | delta |
|-------|-------------|-------------|-------|
| 0~7 | 0.66~0.74 | 0.67~0.76 | 거의 동일 |
| **8** | 0.677 | 0.789 | **-0.11** |
| **9** | 0.646 | 0.797 | **-0.15** |
| **10** | 0.688 | 0.861 | **-0.17** |
| **11** | 0.678 | 0.848 | **-0.17** |

L8-11에서 한국어 모델의 양자화 품질이 급격히 악화. 원인: activation range가 영어 대비 5~10배 넓음.

### 4.3 Split이 안 되는 이유 정리

| Split 방식 | 왜 안 되나 |
|---|---|
| lm_head만 fp32 | Transformer 출력이 이미 망가져서 복구 불가 |
| L8-11+lm fp32 | CER 약간 개선되나 CPU에서 6.5분 → 사용 불가 |
| CNN+lm fp32 | CNN 출력을 uint8로 변환해서 Transformer에 넣으니 오히려 악화 |
| Transformer 전부 fp32 | CPU에서 6.5분 → NPU 쓰는 의미 없음 |

---

## 5. T527 ARM CPU 속도 실측

| 구간 | T527 시간 | 비고 |
|------|----------|------|
| NPU 전체 모델 (uint8) | 415ms | 기존 |
| NPU Transformer only | 285ms | CNN 제외 |
| **CPU naive matmul (L8-11 + lm_head)** | **391,854ms (6.5분)** | -O2 컴파일, NEON 미사용 |
| CPU ONNX Runtime (예상) | 4~40초 | NEON SIMD 최적화 시 |
| CPU lm_head matmul (768→56) | ~5ms | 무시 가능 |
| CPU lm_head matmul (768→1912) | ~50ms (예상) | vocab 1912일 때 |

---

## 6. 추가 발견: Acuity 시뮬레이션 ≠ T527 디바이스

| 비교 | argmax 일치율 |
|------|-------------|
| Sim uint8 vs FP32 | 67.1% |
| **Device vs Sim uint8** | **31.5%** |

시뮬레이션에서 NB_agree를 58% → 70.8%로 개선해도 디바이스 CER은 100% → 174%로 악화.
**시뮬레이션 기반 최적화는 디바이스 결과를 예측하지 못함.**

---

## 7. 결론 및 남은 경로

**Split model은 한국어 wav2vec2에서 효과 없음.**
- vocab 56이든 1912이든 무관
- FP32 CER 9%든 30%든 무관
- 어디서 잘라도 무관

**encoder weight의 activation 분포가 uint8 부적합** — 이건 모델 아키텍처나 후처리로 해결 불가.

**유일한 성공 경로:**
```
영어 pretrained encoder (uint8 friendly weight)
  → 한국어 fine-tune (attempt5: WER 40.6%)
    → QAT + margin loss (margin 0.037 → 0.099)
      → aihub 4356시간 대규모 학습
```

---

## 8. 참고

- [OpenVINO NNCF Wav2Vec2 양자화](https://docs.openvino.ai/2024/notebooks/speech-recognition-quantization-wav2vec2-with-output.html) — conv_layers 3개 FP32 복원으로 WER 회복
- LLM W4A16 (GPTQ/AWQ) — activation FP16 유지로 양자화 성공, T527은 W8A8 강제라 불가
- Acuity `pegasus_quantizer` 소스: W8A16 옵션 없음 (weight+activation 동일 qtype 강제)
