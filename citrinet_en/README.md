# CitriNet EN — 영어 음성인식 (T527 NPU)

NVIDIA NeMo CitriNet 영어 모델. T527 NPU uint8 양자화.

## 상태

**NB 변환 완료, 디바이스 테스트 미완료.**

## 모델 스펙

| 항목 | 값 |
|------|-----|
| NB 크기 | 7MB |
| 입력 | `[1, 80, 1, 300]` uint8 mel-spectrogram |
| 출력 | `[1, 1025, 1, 38]` uint8 |
| 입력 scale/zp | 0.03090 / 130 |
| 출력 scale/zp | 1.0 / 0 (주의: 미보정 가능) |
| 입력 길이 | 3초 (300프레임) |
| Vocab | 1,025 classes (1,024 BPE + blank) |
| 디코딩 | CTC greedy |

## 아키텍처

- **모델**: CitriNet (1D Depthwise Separable Conv + SE)
- KoCitrinet과 동일한 구조, 영어 학습
- 한국어 대비 vocab 절반 (1,025 vs 2,049)

## 파일 구조

```
citrinet_en/
├── README.md
├── network_binary.nb     # 7MB, uint8
├── nbg_meta.json         # 양자화 파라미터
└── convert_export.sh     # Pegasus 변환 스크립트
```

## 문서 및 데이터

| 파일 | 설명 |
|------|------|
| [convert_export.sh](convert_export.sh) | Pegasus 변환 스크립트 (ONNX → NB) |

## 전처리

KoCitrinet과 동일:
```
16kHz mono WAV → STFT (n_fft=512, win=25ms, hop=10ms)
→ 80 mel bins → log → normalize → [1, 80, 1, 300] → uint8 양자화
```

## 비고

- 출력 scale=1.0, zp=0은 양자화 미보정 가능성 있음 (확인 필요)
- NB 7MB로 매우 경량 (KoCitrinet 62MB의 1/9)
- 영어 STT 대안으로 Wav2Vec2 (87MB, CER 17.52%)와 비교 필요
