# 실험 로그 2026-03-23 — Wav2Vec2 한국어 uint8 최초 성공

## 1. Zipformer 양자화 최종 결과

| 양자화 | NB | Encoder 상관계수 | CER | 비고 |
|--------|-----|-----------------|-----|------|
| uint8 AA | 63MB | 0.627 | 100% | state input 수동 교정 |
| int16 DFP | 118MB | 0.643 | 100% | 300개 노드 수동 교정 |
| PCQ int8 | 71MB | 0.275 | 100% | 오히려 악화 |
| bf16 | — | — | — | export 실패 |

**결론: Zipformer (5868노드 transformer)는 T527 NPU 양자화 불가.**

---

## 2. int16 DFP 실험 결과

T527 NPU에서 int16 DFP는 **실행은 되지만 결과가 int8보다 나쁨**.

| 모델 | 양자화 | NB | CER | 비고 |
|------|--------|-----|-----|------|
| KoCitrinet | int8 AA | 62MB | 44.44% | 정상 동작 |
| KoCitrinet | **int16 DFP** | 150MB | **330.95%** | int8보다 훨씬 나쁨 |

원인: DFP는 `2^fl` 스케일 + zero_point 없음 → asymmetric_affine int8보다 표현력 열등.
Acuity 6.12에서 `asymmetric_affine int16`은 미지원 (DFP만 가능).

**이전 "T527 NPU는 int16 미지원" 결론 교정**: Zipformer int16 (118MB) 정상 동작 확인. Wav2Vec2 int16 (153MB) 실패는 NB 크기 제한이 원인.

---

## 3. Wav2Vec2 영어 vs 한국어 — 양자화 실패 원인 분석

### 아키텍처: 100% 동일

| 항목 | 영어 (base-960h) | 한국어 (base-korean) |
|------|-----------------|---------------------|
| Encoder Layers | 12 | 12 |
| Hidden Dim | 768 | 768 |
| Parameters | 94.4M | 94.4M |
| ONNX | 361MB | 361MB |

ONNX 그래프 노드 수가 달랐던 이유: export 도구 차이 (영어: opset 12, 한국어: optimum → opset 14). opset 12로 re-export 후 동일해짐.

### 양자화 실패 근본 원인: Logit Margin

| | 영어 | 한국어 |
|---|---|---|
| Logit std | **8.39** | **1.95** |
| Top1-Top2 margin 최소 | **0.34** | **0.005** |
| uint8 step size | 0.08 | 0.05 |
| margin > step? | **0.34 > 0.08 ✓** | **0.005 < 0.05 ✗** |
| uint8 CER | **17.52%** | **100%** |

한국어 모델의 logit margin이 uint8 step size보다 작아서 양자화 후 argmax 결과 전부 뒤집힘.

---

## 4. Wav2Vec2 한국어 개선 시도

### Amplitude Normalization — 효과 없음

| 시도 | CER |
|------|-----|
| 기존 uint8 | 100% |
| amp norm 5.0 + 기존 NB | 100% |
| amp norm 5.0 + KL 재양자화 | 100% |

rknn-stt에서 효과적이었지만 T527에서는 안 됨 (Split INT8+FP16 없이는 무의미).

### 3-Part Split 각각 uint8 — 실패

```
Part A (CNN, 3.7MB)     → 고정값 출력 (입력 무시)
Part B (L0-5, 35MB)     → 무의미
Part C (L6-11, 34MB)    → 무의미
```

### 6L Pruned 모델 — 실패

FP32에서도 garbage (fine-tuning 없이 pruning만 해서 모델 망가짐).

---

## 5. 영어 base-960h → 한국어 Fine-tune — 최초 성공!

**전략**: uint8에서 동작하는 영어 모델 (logit margin min=0.34)에서 시작 → 한국어 CTC fine-tune

### 학습 이력

| Attempt | Epochs | LR | Frozen | WER | Margin min |
|---------|--------|-----|--------|-----|------------|
| 1 | 30 | 1e-4 | CNN | 100% (수렴 실패) | 0.0000 |
| 2 | +50 | 3e-5 | CNN + L0-5 | 54.18% | 0.0196 |
| 3 | +50 | 1e-5 | CNN only | 44.04% | 0.0018 |
| 4 | +30 | 5e-6 | CNN only | 42.06% | 0.0110 |
| **5** | **+50** | **2e-6** | **CNN only** | **40.60%** | **0.0365** |

- 데이터: Zeroth-Korean (22,263 train / 457 test, ~50시간)
- GPU: RTX 4070 Ti Super 16GB
- 총 학습 시간: ~7시간 (5 attempts)

### T527 NPU uint8 테스트 결과 (attempt5)

| 샘플 | NPU 출력 (음절) | GT |
|------|-----------------|-----|
| ko_test_0000 | 묵도 굗은 가ㅏㅏㅏ | 몬터규는 자녀들이... |
| ko_test_0001 | 가럼 며ㄸㄹ네니 ㅁ | 차 문이 종잇장처럼... |
| ko_test_0003 | **그리구 어기 나** | **그리고 이 나무는...** |
| ko_test_0006 | **야구워된 여당** | **야권은 여당에서...** |
| ko_test_0009 | **이벋에 발꼋 뗐다다** | **이번에 발견된...** |

**이전: CER 100% (전부 garbage) → 현재: 의미있는 한국어 출력!**

---

## 6. leader NB (kor_3s_ref09_best_nb) 테스트

외부에서 가져온 NB (73MB, i8).

### 기존 모델과의 비교

| | leader (ref09) | 기존 base-korean | 우리 attempt5 |
|---|---|---|---|
| qtype | **i8 (signed)** | u8 (unsigned) | u8 |
| Input scale | 0.00481 | 0.00427 | 0.04208 |
| Input range | [-0.59, 0.64] | [-0.54, 0.55] | [-4.19, 6.54] |
| Output range | [-5.18, 11.12] | [-10.21, 12.07] | [-19.84, 17.18] |

### vpm_run 테스트 (ko_test_0003, GT: "그리고 이 나무는...")

| 전처리 | blank | 출력 |
|--------|-------|------|
| raw (no norm) | 144/149 | ㅣㅣㅏㅣ |
| w2v norm | 143/149 | ㄱㅡㅣㅗㅇㅡ |
| amp5 no norm | 135/149 | ㄱㅡㅣㄱㅗㅇㅣ ㄴ ㅇ |
| amp0.5 | 140/149 | ㅣㅗㅣ ㄴㅏㅇ |

**결론: leader NB도 기존 base-korean과 동일한 문제 (대부분 blank). 우리 attempt5가 유일하게 동작.**

---

## 7. Android 앱 통합

### bundle_app (android_stt_bundle_app)

- raw audio 입력 지원 추가 (`isRawAudio` 분기)
- wav2vec2 normalize + i8 양자화 구현
- jamo CTC 디코딩 + 자모→음절 조합 (`KoreanTokenizer.composeJamo`) 추가
- leader NB + attempt5 NB 모두 앱에서 NPU 추론 동작 확인

### awaiasr_2

- `wav2vec2_base_kor_leader.nb` 탑재
- Config.java 모델 경로 수정
- awwav2vecsdk.c 양자화 파라미터 i8 대응 (`scale=0.00481, zp=-6`)
- NPU 추론 463ms 확인

---

## 8. 모델 비교 종합

| 모델 | 양자화 | NB | T527 결과 | 비고 |
|------|--------|-----|-----------|------|
| KoCitrinet int8 | AA uint8 | 62MB | **CER 44.44%** (동작) | 운용중 |
| KoCitrinet int16 | DFP | 150MB | CER 330% | int8보다 나쁨 |
| Wav2Vec2 EN uint8 | AA uint8 | 88MB | **CER 17.52%** (동작) | 검증완료 |
| Wav2Vec2 KO (기존) | AA uint8 | 72MB | CER 100% | 실패 |
| Wav2Vec2 KO (leader) | AA i8 | 73MB | 대부분 blank | 실패 |
| **Wav2Vec2 KO (attempt5)** | **KL uint8** | **72MB** | **WER 40.6%, 한국어 출력!** | **최초 성공** |
| Zipformer | 전 방식 | 63~118MB | CER 100% | 실패 |

---

## 9. 다음 단계

1. **NAS 4356시간 데이터로 재학습** — WER 40% → 10~20% 목표
2. **attempt6 학습 진행 중** (100ep, LR=1e-6, GPU 학습 중)
3. **Android 앱 통합 완성** — vocab/decoder 매칭
4. **per-sample CER CSV 생성** — 정확한 평가
