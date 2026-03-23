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

---

## 진행 상황 (2026-03-23)

### ONNX Export 성공

```
stt_en_conformer_ctc_small (NeMo)
  → encoder+decoder only (mel 입력, STFT 제외)
  → opset 12 export
  → Where 48개 제거 (attention mask 상수 fold)
  → onnxsim: 1647 nodes
  → 53MB ONNX
```

- Input: `[1, 80, 301]` mel spectrogram
- Output: `[1, 76, 1025]` logprobs (영어 BPE 1025)
- 13.2M params

### Acuity Import 실패

`Pad` op에서 `IndexError: list index out of range`:
```
W /encoder/layers.0/self_attn/Pad : Pad input shape ['1x4x76x151', '8', '']
```

Conformer의 relative positional encoding에서 사용하는 Pad op의 세 번째 입력(constant_value)이 빈 텐서로, Acuity 6.12가 처리 못 함.

### 해결 필요

1. Pad op 16개의 constant_value를 명시적 0으로 채우기
2. 또는 Pad를 수동으로 다른 op(Concat + zeros)로 교체
3. 또는 Acuity 6.21 시도 (Pad 처리 개선 가능)

### Acuity Import 성공! (Pad fix 후)

Pad op의 빈 constant_value를 명시적 0으로 채운 후 import 성공.

### T527 NPU 테스트 결과

```
NB 크기:    14MB (KoCitrinet 62MB의 1/4!)
추론 시간:  74ms (KoCitrinet 120ms보다 빠름!)
출력:       blank=65/76, non-blank=11 (CTC 동작)
```

| | Conformer small | KoCitrinet 256 |
|---|---|---|
| NB 크기 | **14MB** | 62MB |
| 추론 시간 | **74ms** | 120ms |
| Params | 13.2M | ~10M |
| uint8 양자화 | **성공** | 성공 |

### 다음 단계

1. **한국어 fine-tune** (영어 Conformer → 한국어 CTC, wav2vec2와 같은 전략)
2. **FP32 logit margin 측정** (uint8 생존 가능성 확인)
3. **KoCitrinet과 CER 비교**

### ONNX 그래프 수술 이력

```
원본 NeMo export (opset 16, 3982 nodes)
  → mel 추출 분리 (STFT 제거, mel 입력으로 변경)
  → opset 12 export
  → Where 48개 제거 (attention mask 상수 fold)
  → Pad 16개 수정 (빈 constant_value → 0.0)
  → onnxsim: 1647 nodes, 53MB
  → Acuity uint8 KL: 14MB NB
```
