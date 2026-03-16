# Zipformer — 한국어 스트리밍 음성인식 (T527 NPU)

sherpa-onnx-streaming-zipformer-korean-2024-06-16 기반. RNN-Transducer 구조 (Encoder + Decoder + Joiner).

## 상태

**NB 변환 완료, 디바이스 테스트 미완료.**

## 모델 구성

| 컴포넌트 | NB 크기 | 양자화 | 입력 | 출력 |
|----------|---------|--------|------|------|
| Encoder | 63MB | uint8 | `[1, 39, 80]` + 캐시 31개 | `[1, 8, 512]` |
| Decoder | 2.8MB | int32 입력 | `[1, 2]` int32 (토큰 ID) | `[1, 512]` uint8 |
| Joiner | 1.9MB | float16 입력 | `[1, 512]` × 2 (enc+dec) | `[1, 5000]` uint8 |
| **합계** | **68MB** | — | — | — |

## Encoder 입력 상세

주 입력:
- `x`: `[1, 39, 80]` uint8 — mel-spectrogram 프레임 (scale=0.02136, zp=125)

캐시 상태 (31개, 모두 uint8, 초기값 0):
- `cached_avg_0~4`: 어텐션 평균 캐시
- `cached_key_0~4`: 어텐션 키 캐시
- `cached_val_0~4`: 어텐션 값 캐시
- `cached_val2_0~4`: 어텐션 값2 캐시
- `cached_conv1_0~4`: 컨볼루션 캐시1
- `cached_conv2_0~4`: 컨볼루션 캐시2
- `cached_len` (int32): 프레임 카운터

## Decoder 입력

- `y`: `[1, 2]` **int32** — 직전 2개 토큰 ID (비양자화)

## Joiner 입력

- `encoder_out`: `[1, 512]` **float16** — Encoder 출력 (비양자화)
- `decoder_out`: `[1, 512]` **float16** — Decoder 출력 (비양자화)
- 출력: `[1, 5000]` uint8 → argmax → 토큰 ID

## 보캡

- **5,000 BPE 토큰** (`tokens.txt`)
- SentencePiece BPE 토크나이저 (`bpe.model`)

## 아키텍처

- **Zipformer**: 효율적인 Transformer 변형 (downsampling + bypass)
- **RNN-Transducer**: Encoder + Decoder + Joiner 3단 구조
- **스트리밍**: 청크 단위 추론 (39프레임 = ~390ms)
- **학습**: k2-fsa/icefall, 한국어 데이터

## 파일 구조

```
zipformer/
├── README.md
├── bpe.model              # SentencePiece BPE 토크나이저
├── tokens.txt             # 5,000 BPE 토큰
├── encoder/
│   ├── network_binary.nb  # 63MB, uint8
│   ├── nbg_meta.json
│   └── sample_zeros.txt   # 제로 입력 테스트용
├── decoder/
│   ├── network_binary.nb  # 2.8MB
│   └── nbg_meta.json
└── joiner/
    ├── network_binary.nb  # 1.9MB
    └── nbg_meta.json
```

## 과제

1. **Encoder 캐시 31개 입력** — vpm_run/awnn에서 다중 입력 처리 필요
2. **Joiner float16 입력** — Encoder/Decoder 출력을 float16으로 변환 후 전달
3. **RNN-T 디코딩 루프** — Encoder 1회 → Decoder/Joiner 반복 (greedy search)
4. **스트리밍 상태 관리** — 이전 청크의 캐시를 다음 청크에 전달

## 디바이스 테스트 방법 (예정)

```bash
# 1. Encoder 테스트 (제로 입력)
adb push encoder/network_binary.nb /data/local/tmp/zipformer/
# 32개 입력 파일 생성 필요 (x + 31 caches)

# 2. Decoder 테스트
adb push decoder/network_binary.nb /data/local/tmp/zipformer/
# int32 입력 [1, 2] 생성

# 3. Joiner 테스트
adb push joiner/network_binary.nb /data/local/tmp/zipformer/
# float16 입력 2개 [1, 512] 생성
```

## 원본 모델

- **이름**: sherpa-onnx-streaming-zipformer-korean-2024-06-16
- **프레임워크**: k2-fsa/icefall (PyTorch → ONNX)
- **원본 ONNX**: encoder-epoch-99-avg-1.int8.onnx (63MB), decoder-epoch-99-avg-1.onnx (2.8MB), joiner-epoch-99-avg-1.int8.onnx (1.9MB)
