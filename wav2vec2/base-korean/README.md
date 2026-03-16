# Wav2Vec2 Base Korean — 한국어 (T527 NPU)

Kkonjeong/wav2vec2-base-korean. 한국어 STT, T527 NPU 양자화. **전부 실패.**

## 결과: 50종+ 시도, 21종 NPU 실측, 전부 실패

| 카테고리 | 시도 수 | 결과 |
|---------|--------|------|
| uint8 동작, garbled 출력 | 12종 | ALL PAD 또는 ㅇ 토큰 과다 |
| int16/DFP, NPU HANG | 3종 | 물리적 전원 리셋 필요 |
| NPU status=-1 (실행 거부) | 4종 | CNN-only, split, combo 등 |
| NB 생성 실패 | 6종 | bf16, PCQ, symmetric 등 |
| 시뮬레이션만 | 5종+ | 6L, 8L, 10L, SmoothQuant 등 |

## 모델 스펙

| 항목 | 값 |
|------|-----|
| 원본 | Kkonjeong/wav2vec2-base-korean (HuggingFace) |
| 파라미터 | 94.4M, 12 Transformer layers |
| NB 크기 | 72MB (uint8) / 153MB (int16) |
| 입력 | `[1, 48000]` raw waveform (3초, 16kHz) |
| Vocab | 56 (한글 자모 분리) |
| 추론시간 | 425ms / 3초 (uint8, 동작하지만 출력 쓰레기) |
| RTF | 0.142 |

## 실패 원인 분석

1. **uint8 양자화 = logit 동적 범위 46% 손실** — 12-layer Transformer에서 오류 누적
2. **int16 DFP = FP32 = PyTorch 100% 동일** — 그러나 T527 NPU가 DFP 미지원 (HANG)
3. **내부 동적 범위 차이** — 영어 모델 L11 residual [-35,30] vs 한국어 [-4.2,3.4] (10x 차이)
   - 영어는 range 넓어서 uint8 양자화 노이즈 비중 작음
   - 한국어는 range 좁아서 uint8 양자화 노이즈가 신호 자체를 삼킴
4. **CNN-only/split/combo = status=-1** — NPU가 소형/수정 모델 실행 거부

## 시도한 양자화 방식

| 방식 | NPU 결과 |
|------|---------|
| uint8 asymmetric_affine | garbled (ALL PAD) |
| uint8 + moving_average | garbled |
| uint8 + KL divergence | garbled |
| uint8 + entropy | garbled |
| int8 symmetric | NB 생성 실패 |
| int16 DFP | NPU HANG |
| bf16 | NB 생성 실패 (segfault) |
| PCQ (per-channel) | NB 생성 실패 |
| hybrid (CNN uint8 + Transformer int16) | NPU HANG |
| opset12 + onnxsim | garbled (입력 범위 수정됨, 출력 여전히 실패) |
| CNN-only (feature extractor만) | NPU status=-1 |
| split (CNN uint8 → Transformer int16) | Part B NPU HANG |
| layer pruning (6L, 8L, 10L) | 시뮬레이션만, NPU status=-1 |
| SmoothQuant | 시뮬레이션만 |
| temperature scaling | garbled |
| combo (relu + 6L + nopad) | NPU status=-1 |

## 파일 구조

```
base-korean/
├── README.md
├── download_and_convert.py       # HuggingFace → ONNX 변환
├── decode_ko_output.py           # NPU 출력 → 한국어 텍스트
├── prepare_ko_test_input.py      # 테스트 입력 준비
├── create_cnn_only_model.py      # CNN-only ONNX 추출
├── test_all_nbs.sh               # 전체 NB 일괄 테스트
├── test_split_model.sh           # CNN(uint8) → Transformer(int16) 분리 테스트
├── test_priority_nbs.sh          # 우선순위 NB 테스트
├── auto_test_on_connect.sh       # 디바이스 연결 시 자동 테스트
├── vocab.json                    # 56 자모 vocab
└── config.json                   # HuggingFace 모델 config
```

## 결론

T527 NPU에서 한국어 Wav2Vec2 (Transformer 기반)는 **동작 불가능**.
- uint8만 NPU에서 실행 가능하나 한국어 모델은 양자화 열화로 출력이 파괴됨
- int16/bf16은 NPU 하드웨어 미지원
- CNN 기반 모델(KoCitrinet)만이 유일한 대안
