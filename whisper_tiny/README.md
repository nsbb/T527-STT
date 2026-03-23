# Whisper tiny — T527 NPU 테스트

## 결과

| 항목 | 값 |
|------|-----|
| 모델 | openai/whisper-tiny (encoder only) |
| Params | 8.2M (encoder) |
| ONNX | 33MB, 211 nodes |
| NB | **117MB** (30초 고정 입력) |
| T527 추론 | **937ms** (30초 기준) |
| 구조 | Encoder-Decoder (CTC 아님) |

## 문제

Whisper는 Encoder-Decoder 구조라 Encoder만으로는 텍스트 출력 불가.
Decoder도 NPU에 올려야 하는데 autoregressive라 반복 실행 필요.

## 상태

**보류** — Encoder는 NB 성공, 하지만 Decoder 없이 사용 불가.
