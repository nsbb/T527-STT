# Wav2Vec2 English Test Set

LibriSpeech test-clean에서 추출한 영어 음성 50개.

## 스펙

| 항목 | 값 |
|------|-----|
| 출처 | LibriSpeech test-clean |
| 샘플 수 | 50 |
| 샘플링 레이트 | 16kHz mono |
| 길이 | 1.8~7.4초 |
| 포맷 | WAV (PCM 16bit) |

## 파일

- `en_test_0000.wav` ~ `en_test_0049.wav` — 음성 파일
- [ground_truth.txt](ground_truth.txt) — 정답 텍스트 (filename, GT, duration)

## 평가 결과

| 모델 | CER | WER |
|------|-----|-----|
| ONNX FP32 | 9.74% | — |
| T527 NPU uint8 | 17.52% | 27.38% |

결과 CSV: [../base-960h-en/test_results_librispeech.csv](../../wav2vec2/base-960h-en/test_results_librispeech.csv)
