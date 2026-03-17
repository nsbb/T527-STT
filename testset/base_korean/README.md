# Wav2Vec2 Korean Test Set

Zeroth-Korean test split에서 추출한 한국어 음성 100개.

## 스펙

| 항목 | 값 |
|------|-----|
| 출처 | kresnik/zeroth_korean (HuggingFace) test split |
| 샘플 수 | 100 |
| 샘플링 레이트 | 16kHz mono |
| 음성 유형 | 뉴스/책 낭독체 |
| 포맷 | WAV (PCM 16bit) |

## 파일

- `ko_test_0000.wav` ~ `ko_test_0099.wav` — 음성 파일
- [ground_truth.txt](ground_truth.txt) — 정답 텍스트 (filename, GT, duration)

## 평가 결과

| 모델 | CER |
|------|-----|
| PyTorch FP32 | 9.5% |
| T527 NPU uint8 | 전부 실패 (garbled) |

결과 CSV: [../../wav2vec2/base-korean/test_results_zeroth_korean_pytorch_fp32.csv](../../wav2vec2/base-korean/test_results_zeroth_korean_pytorch_fp32.csv)
