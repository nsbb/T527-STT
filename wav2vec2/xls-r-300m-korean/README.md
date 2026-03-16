# Wav2Vec2 XLS-R 300M Korean — 한국어 (T527 NPU)

kresnik/wav2vec2-large-xlsr-korean. 한국어 STT, T527 NPU 양자화. **전부 실패.**

## 결과

| 항목 | 값 |
|------|-----|
| NPU CER | ALL PAD (완전 실패) |
| 추론시간 | 1,098ms / 3초 |
| RTF | 0.366 |

8종+ 양자화 시도, 전부 실패 (garbled 또는 NPU HANG).

## 모델 스펙

| 항목 | 값 |
|------|-----|
| 원본 | kresnik/wav2vec2-large-xlsr-korean (HuggingFace) |
| 파라미터 | 300M, 24 Transformer layers |
| ONNX 크기 | 1.27GB |
| NB 크기 | 249MB (uint8) / 262MB (int16) |
| 입력 | `[1, 48000]` raw waveform (3초, 16kHz) |
| Vocab | 2,617 (한글 음절) |

## base-korean 대비 차이점

| 항목 | base-korean (12L) | XLS-R-300M (24L) |
|------|-------------------|------------------|
| 파라미터 | 94.4M | 300M |
| Layers | 12 | 24 |
| Hidden | 768 | 1024 |
| Vocab | 56 자모 | 2,617 음절 |
| NB 크기 | 72MB | 249MB |
| 추론시간 | 425ms | 1,098ms |
| 결과 | 실패 | 실패 |

24L 모델이라 양자화 열화가 더 심하고, NB도 249MB로 거대.
12L base-korean도 실패했으므로 더 큰 모델이 될 리 없음.

## 파일 구조

```
xls-r-300m-korean/
├── README.md
├── make_fixed_onnx.py            # 동적 → 고정 shape ONNX 변환
├── prune_layers.py               # Transformer 레이어 pruning (24L→12L)
├── decode_npu_output.py          # NPU 출력 → 한국어 텍스트
├── prepare_calib_data.py         # calibration 데이터 준비
├── prepare_test_input.py         # 테스트 입력 준비
├── compare_onnx_npu.py           # ONNX vs NPU 비교
├── analyze_output.py             # 출력 분석
├── run_pipeline.sh               # 전체 변환 파이프라인
├── test_xlsr_opset12_nb.sh       # opset12 NB 디바이스 테스트
└── vocab_xlsr.json               # 2,617 음절 vocab
```

## 결론

base-korean (12L)과 동일한 이유로 실패. T527 NPU uint8 양자화로는 Transformer 기반 한국어 STT 불가능.
