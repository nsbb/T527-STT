# 260325 Split Model 추가 실험 결과

**날짜:** 2026-03-25
**이전 문서:** [260324_split_model_approach.md](260324_split_model_approach.md) — Split 개념, 구현 방법, 영어/한국어 base-korean 실험

---

## 1. aihub 80k 모델 (vocab 1912 음절) Split 실험

### 1.1 모델 정보

| 항목 | 값 |
|------|-----|
| 위치 | `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_korean/` |
| 학습 | aihub 데이터, 80000 step |
| vocab | 1912 (음절 단위) |
| 아키텍처 | wav2vec2-base (12L, 768H, 94.4M) |
| ONNX | `wav2vec2_base_korean_80k.onnx` (378MB) |
| NB (full) | `wksp/bk_uint8_nbg_unify/network_binary.nb` (77MB) |

### 1.2 FP32 성능 (eval_results)

| 테스트셋 | CER | 추론속도 |
|----------|-----|---------|
| 저음질 전화망 (007) | **10.23%** | 0.03s |
| 상담음성 (012) | **9.69%** | 0.03s |
| 한국어 강의 (009) | 16.57% | 0.03s |
| 회의음성 (010) | 16.75% | 0.04s |
| 7F_KSK | 9.48% | 0.01s |
| 7F_HJY | 28.06% | 0.01s |
| modelhouse 3m | 35.76% | 0.01s |

### 1.3 Full uint8 (기존 NB)

| CER (3s trunc) | CER (full GT) | non-blank | 비고 |
|---|---|---|---|
| 92.83% | 98.21% | 0~4개/149 | 거의 blank |

→ FP32 CER 9~18%인 좋은 모델이 uint8에서 92~98% — 완전 실패.

### 1.4 Split (encoder NPU uint8 + lm_head CPU fp32)

**방법:**
- ONNX에서 lm_head (MatMul+Add) 제거 → encoder_only.onnx
- lm_head weight (768×1912, fp32) + bias (1912, fp32) 별도 저장
- encoder NB: 76MB, 입력 scale=0.05061302 zp=119, 출력 scale=0.03400786 zp=127
- 기존 테스트 입력 재사용 (input scale/zp 동일)

**결과 (20 samples):**

| # | CER | nb | Split 출력 | GT |
|---|---|---|---|---|
| 3 | 81.8% | 2 | 그고 | 그리고 이 나무는 태즈메이 |
| 6 | 80.0% | 6 | 야고 여이 | 야권은 여당에서 분명한 |
| 11 | 83.3% | 4 | 하만 | 하지만 싫증 나면 버리면 그만 |
| 17 | 70.0% | 4 | 플이장에 | 애플 입장에선 이천 십 이 |
| 0 | 100% | 0 | (blank) | 몬터규는 자녀들이 사랑을 |

**CER: 92.65%** — full uint8 (92.83%)과 동일. **개선 없음.**

---

## 2. 전체 비교 (260324 + 260325 합산)

| 모델 | vocab | FP32 CER | Full uint8 | Split lm fp32 | Split L7 | CNN+lm fp32 |
|------|-------|---------|------------|---------------|----------|-------------|
| 영어 base-960h | 32 | 9.74% | **17.52%** | 20.68% | — | — |
| 한국어 base-korean | 56 | 30.22% | 100.86% | 99.70% | 99.26% | 100.00% |
| **한국어 aihub 80k** | **1912** | **9~18%** | **92.83%** | **92.65%** | — | — |
| 한국어 fine-tune (attempt5) | 56 | WER 40.6% | 부분 성공 | — | — | — |

---

## 3. 결론

**FP32 성능이 좋아도 (CER 9~18%) encoder uint8 양자화에서 깨지면 Split으로 복구 불가.**

- vocab 크기 (56 vs 1912) 무관
- FP32 성능 (9% vs 30%) 무관
- Split 지점 (lm_head / L7 / CNN) 무관

**encoder weight의 activation 분포가 uint8 부적합 — 이건 후처리로 해결 불가, 학습으로만 해결 가능.**

유일한 경로: 영어 pretrained encoder (uint8 friendly weight) → 한국어 fine-tune + QAT + aihub 대규모 데이터.
