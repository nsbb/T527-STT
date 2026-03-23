# Conformer — 한국어 음성인식 (T527 NPU 후보)

## 모델 후보

NeMo에서 제공하는 한국어 Conformer 계열 모델:

| 모델 | 아키텍처 | 크기 | CTC/RNN-T | 양자화 전망 |
|------|---------|------|-----------|-----------|
| stt_ko_conformer_ctc_small | Conformer CTC | ~15M | CTC | **유망** (작은 모델) |
| stt_ko_conformer_ctc_medium | Conformer CTC | ~30M | CTC | 유망 |
| stt_ko_conformer_ctc_large | Conformer CTC | ~120M | CTC | 어려울 수 있음 |
| stt_ko_squeezeformer_ctc_medium_ls | SqueezeFormer CTC | ~30M | CTC | **유망** (효율적) |
| stt_ko_fastconformer_ctc_large | FastConformer CTC | ~120M | CTC | 어려울 수 있음 |

## 왜 Conformer인가

Conformer = **CNN (depthwise conv) + Self-Attention** 하이브리드.

- KoCitrinet (순수 CNN): CER 8.44% — uint8 양자화 성공
- Wav2Vec2 (순수 Transformer): CER 100% — uint8 양자화 실패
- **Conformer: CNN 부분은 양자화에 강하고, Attention 부분은 제한적**

### KoCitrinet과의 비교

| | KoCitrinet | Conformer |
|---|---|---|
| 구조 | 1D Conv + SE | Conv + Self-Attention |
| 양자화 적합성 | **매우 좋음** (CNN only) | 중간 (하이브리드) |
| 정확도 | CER 8.44% (표준) | **더 좋을 것으로 기대** |
| 입력 | mel spectrogram | mel spectrogram |
| 디코딩 | CTC | CTC |

## T527 양자화 전략

1. **small 모델 우선** — 노드 수 적어서 에러 누적 적음
2. **CTC 모델만** — RNN-T는 Decoder/Joiner 추가 필요
3. **KL divergence 양자화** — Acuity `--algorithm kl_divergence`
4. **logit margin 확인** — 양자화 전 FP32 logit margin 측정

## 다운로드 방법

```python
import nemo.collections.asr as nemo_asr

# Small Conformer CTC
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_ko_conformer_ctc_small")

# ONNX export
model.export("stt_ko_conformer_ctc_small.onnx")
```

## 상태

**미시작** — 모델 다운로드 + ONNX export + Acuity 양자화 + T527 테스트 필요.
