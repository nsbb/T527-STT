# Zipformer — 한국어 스트리밍 음성인식 (T527 NPU)

sherpa-onnx-streaming-zipformer-korean-2024-06-16 기반. RNN-Transducer 구조 (Encoder + Decoder + Joiner).

> **참고:** RK3588 NPU에서는 동일 모델로 CER 21.85% 달성 (INT8 KL + CumSum 패치). T527과의 차이 분석 → [RKNN vs T527 비교 문서](RKNN_COMPARISON.md)

## 상태

**양자화 실패 — uint8 / int16 / PCQ / bf16 전 방식 CER 100%**

ONNX float baseline CER: 16.2% (4 Korean test samples)

## 양자화 실험 결과

| 양자화 | NB 크기 | T527 NPU 실행 | Encoder 출력 상관계수 | CER | 비고 |
|--------|---------|-------------|---------------------|-----|------|
| uint8 asymmetric_affine | 63MB | **정상 동작** | 0.627 | **100%** | state input 수동 교정 |
| int16 dynamic_fixed_point | 118MB | **정상 동작** | 0.643 | **100%** | 300개 노드 수동 교정 |
| PCQ int8 perchannel_symmetric | 71MB | **정상 동작** | 0.275 | **100%** | 오히려 악화 |
| bf16 bfloat16 | — | — | — | — | export 실패 (error 64768) |

> **참고**: uint8/int16/PCQ 모두 T527 NPU에서 **정상 실행**됨 (crash/hang 없음). 출력이 나오긴 하나 양자화 에러 누적으로 의미없는 값. 특히 **int16 (118MB)이 T527 NPU에서 정상 동작**한 것은 이전 Wav2Vec2 int16 (153MB) 실패가 "int16 미지원"이 아니라 **NB 크기 제한** 때문이었음을 증명.

### 실패 원인

1. **양자화 에러 누적**: 5868개 노드의 sequential quantization으로 encoder 출력 상관계수 0.6 수준. Decoder/Joiner가 의미있는 토큰 생성 불가.
2. **Acuity multi-input 캘리브레이션 버그**: 31개 입력 모델에서 state 입력 calibration 데이터를 무시. 30개 state가 모두 scale=1.0/zp=0 또는 fl=300으로 설정됨.
3. **비트 수 증가 무효**: int16 (2배 정밀도)도 상관계수 개선 미미 (0.627→0.643).

### 동일 NPU 양자화 비교

| 모델 | 노드 수 | 구조 | uint8 CER |
|------|---------|------|-----------|
| KoCitrinet | ~200 | 1D Conv (CTC) | 44.44% (성공) |
| Wav2Vec2 base | ~2000 | 12L Transformer (CTC) | ~17.5% (성공) |
| **Zipformer** | **5868** | **5-stack Transformer (RNN-T)** | **100% (실패)** |

## 모델 구성

| 컴포넌트 | NB 크기 | 양자화 | 입력 | 출력 |
|----------|---------|--------|------|------|
| Encoder | 63MB | uint8 | `[1, 39, 80]` + 캐시 30개 | `[1, 8, 512]` + 캐시 30개 |
| Decoder | 2.8MB | int32 입력 | `[1, 2]` int32 (토큰 ID) | `[1, 512]` uint8 |
| Joiner | 1.9MB | float16 입력 | `[1, 512]` × 2 (enc+dec) | `[1, 5000]` uint8 |

## Encoder 입력 상세

주 입력:
- `x`: `[1, 39, 80]` — mel-spectrogram 프레임

캐시 상태 (30개, 초기값 0):
- `cached_avg_0~4`: 어텐션 평균 캐시
- `cached_key_0~4`: 어텐션 키 캐시
- `cached_val_0~4`: 어텐션 값 캐시
- `cached_val2_0~4`: 어텐션 값2 캐시
- `cached_conv1_0~4`: 컨볼루션 캐시1
- `cached_conv2_0~4`: 컨볼루션 캐시2

## 테스트셋

`test_wavs/` — 4개 한국어 WAV (KSS Dataset)

| 파일 | 텍스트 |
|---|---|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. |
| 3.wav | 주민등록증을 보여 주시겠어요? |

## 보캡

- **5,000 BPE 토큰** (`tokens.txt`)
- SentencePiece BPE 토크나이저 (`bpe.model`)

## 파일 구조

```
zipformer/
├── README.md
├── bpe.model              # SentencePiece BPE 토크나이저
├── tokens.txt             # 5,000 BPE 토큰
├── encoder/
│   ├── network_binary.nb  # 63MB, uint8 (CER 100%, 사용불가)
│   ├── nbg_meta.json
│   └── sample_zeros.txt
├── decoder/
│   ├── network_binary.nb  # 2.8MB
│   └── nbg_meta.json
└── joiner/
    ├── network_binary.nb  # 1.9MB
    └── nbg_meta.json
```

## 원본 모델

- **이름**: sherpa-onnx-streaming-zipformer-korean-2024-06-16
- **프레임워크**: k2-fsa/icefall (PyTorch → ONNX)
- **Encoder ONNX**: 280MB, 5868 nodes, 31 inputs, 31 outputs
- **테스트/양자화 스크립트**: `ai-sdk/models/zipformer/` 참조

## 결론

> T527 NPU (Acuity 6.12.0)로는 5868노드 transformer encoder 양자화 불가.
> CNN 기반(~200노드) 또는 중간 크기 transformer(~2000노드)까지만 사용 가능.
