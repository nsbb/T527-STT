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

## 도메인 미스매치 검증 (2026-03-17)

NPU 양자화 실패와 별개로, 모델 자체의 성능을 검증.

### Zeroth-Korean 테스트셋 (학습 도메인)

XLS-R-300M Korean의 학습 데이터는 base-korean과 동일한 Zeroth-Korean (51시간, 낭독체).
kresnik 공개 지표 기준:

| 항목 | XLS-R-300M | base-korean (12L) |
|------|------------|-------------------|
| Zeroth-Korean CER | **1.78%** | 9.5% |
| 파라미터 | 300M | 94.4M |
| Pretrain | 128개 언어 436K시간 | 영어 960시간 |
| Vocab | 2,617 음절 | 56 자모 |

→ XLS-R은 동일 데이터셋에서 base보다 5배 나은 CER. 다국어 pretrain + 대형 모델의 효과.

### 월패드 테스트셋

base-korean의 월패드 CER이 132~210% (ONNX FP32)인 점을 고려하면, 동일 학습 데이터로 fine-tuning된 XLS-R도 월패드에서 유사하게 실패할 것으로 예상. 다만 pretrain 품질이 높아 fine-tuning 후 잠재력은 더 큼.

### Fine-tuning 고려사항

```
[Option C] XLS-R-300M 기반 fine-tuning (최고 성능 가능)
  - kresnik/wav2vec2-large-xlsr-korean 또는 facebook/wav2vec2-xls-r-300m에서 시작
  - 월패드 데이터로 추가 fine-tuning
  - 장점: 128개 언어 pretrain → 한국어 음소 기반 지식 풍부, Zeroth-Korean CER 1.78%
  - 단점: 300M params → T527 NPU uint8 양자화 **불가능** (이미 8종+ 실패 확인)
  - 대안: 서버 추론 (GPU/CPU) 또는 int16/bf16 지원 NPU 하드웨어
```

> **핵심 딜레마**: 가장 좋은 모델(XLS-R-300M)이 T527 NPU에 올라가지 않고, T527에 올라가는 모델(base-korean)은 성능이 부족.

---

## 결론

### 1. NPU 양자화: T527 uint8로 동작 불가능

base-korean (12L)과 동일한 이유로 실패. 24L 모델이라 양자화 열화가 더 심함.

### 2. 도메인 미스매치: 학습 데이터와 월패드의 완전한 불일치

base-korean과 동일한 Zeroth-Korean 51시간으로 학습. 월패드 도메인과 완전 불일치.

### 3. 해결 불가

- Fine-tuning으로 FP32 성능 개선 가능하나, 300M params는 T527 NPU uint8에 절대 올라가지 않음
- T527 NPU 배포가 목표라면 base-korean(94.4M) 또는 CNN 모델(KoCitrinet)만 후보
