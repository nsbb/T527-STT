# Wav2Vec2 Base 960h — 영어 (T527 NPU)

facebook/wav2vec2-base-960h. 영어 STT, T527 NPU uint8 양자화. **동작 성공.**

## 성능

| 항목 | 값 |
|------|-----|
| CER | **17.52%** |
| WER | **27.38%** |
| exact match | 6/50 |
| 추론시간 | 715ms / 5초 음성 |
| RTF | 0.14 |
| ONNX FP32 CER | 9.74% |
| 양자화 열화 | +7.78%p |

## 모델 스펙

| 항목 | 값 |
|------|-----|
| 원본 | facebook/wav2vec2-base-960h (HuggingFace) |
| 파라미터 | 94.4M, 12 Transformer layers |
| NB 크기 | 87MB (uint8) |
| 입력 | `[1, 80000]` uint8 raw waveform (5초, 16kHz) |
| 출력 | `[1, 249, 32]` uint8 logits |
| 입력 scale/zp | 0.002860 / 137 |
| 출력 scale/zp | 0.150270 / 186 |
| Vocab | 32 (26 영문자 + space + apostrophe + blank + special) |
| 디코딩 | CTC greedy |
| ONNX opset | 12 (동적 연산 제거, onnxsim 적용) |

## 파일 구조

```
base-960h-en/
├── README.md
├── network_binary.nb                      # 87MB uint8 NB
├── nbg_meta.json                          # 양자화 파라미터
├── ground_truth.txt                       # 50개 LibriSpeech test-clean GT
├── test_results_librispeech.csv           # GT/pred/CER/WER 결과
├── wav2vec2_base_960h_5s_inputmeta_fixed.yml  # Pegasus 입력 메타
├── wav2vec_postprocess.cpp                # CTC 디코딩 JNI 코드
├── wav2vec_postprocess.h
└── scripts/
    ├── compare_onnx_npu.py                # ONNX vs NPU 출력 비교
    ├── compare_onnx_npu_50.py             # 50개 샘플 배치 비교
    └── eval_wav2vec_cer.py                # CER/WER 평가
```

## 변환 과정

1. HuggingFace → ONNX export (opset 14)
2. opset 12로 다운그레이드 + onnxsim (동적 Shape 연산 제거)
3. Pegasus import → uint8 양자화 (51개 calibration, moving_average)
4. Pegasus export → network_binary.nb
5. `reverse_channel: false` 필수 (inputmeta 설정)

## 발견된 버그

1. **reverse_channel 버그**: Acuity 기본값 true → 입력 뒤집힘 → 쓰레기 출력
2. **JNI float→uint8 캐스팅**: quantized_input 만들어놓고 processedAudio(float32) 전달
3. **출력 텐서 레이아웃**: [vocab,seq] vs [seq,vocab] 혼동
4. **모델 차원 상수**: 3초/5초 모델 혼용으로 잘못된 상수
