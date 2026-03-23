# HuBERT Korean — T527 NPU 테스트

## 결과

| 항목 | 값 |
|------|-----|
| 모델 | HJOK/asr-hubert-base-ko |
| Params | 96M |
| ONNX | 384MB, 727 nodes |
| NB | **76MB** |
| T527 추론 | **423ms** |
| Vocab | 2145 |

## 문제

출력이 전부 동일 토큰 (2142) 반복. Wav2Vec2와 동일한 양자화 문제
(Transformer encoder의 logit margin 부족).

## 상태

**실패** — Wav2Vec2와 동일한 문제. Transformer 기반 한국어 모델은 uint8 양자화 비적합.
