# DeepSpeech2 — 영어 음성인식 (T527 NPU)

Baidu DeepSpeech2 (TensorFlow 원본). T527 NPU uint8 양자화.

## 상태

**NB 변환 완료, 디바이스 테스트 미완료.**

## 모델 스펙

| 항목 | 값 |
|------|-----|
| NB 크기 | 56MB |
| 입력 | `[1, 756, 161, 1]` uint8 (NHWC) |
| 출력 | `[1, 378, 29]` uint8 |
| 입력 scale/zp | 0.01115 / 0 |
| 출력 scale/zp | 0.17175 / 159 |
| Vocab | 29 classes (26 영문자 + space + apostrophe + blank) |
| 디코딩 | CTC greedy |

## 아키텍처

- **모델**: DeepSpeech2 (5층 RNN, Baidu 2014)
- **입력**: spectrogram (161 frequency bins × 756 time steps)
- **레이아웃**: NHWC (TensorFlow 원본)
- **출력**: 378 time steps × 29 classes

## 파일 구조

```
deepspeech2/
├── README.md
├── network_binary.nb           # 56MB, uint8
├── nbg_meta.json               # 양자화 파라미터
├── deepspeech2_inputmeta.yml   # Pegasus 입력 메타 설정
└── scripts/
    ├── pre_process.py          # 전처리 스크립트
    └── post_process.py         # 후처리 스크립트
```

## 문서 및 데이터

### 스크립트 (`scripts/`)

| 파일 | 설명 |
|------|------|
| [pre_process.py](scripts/pre_process.py) | 전처리 스크립트 (WAV → spectrogram) |
| [post_process.py](scripts/post_process.py) | 후처리 스크립트 (NPU 출력 → 텍스트) |

## 전처리

```
16kHz mono WAV → spectrogram (161 frequency bins)
→ 756 time steps (고정 길이)
→ [1, 756, 161, 1] NHWC → uint8 양자화 (scale=0.01115, zp=0)
```

## 비고

- RNN 기반으로 Transformer(Wav2Vec2)나 CNN(CitriNet)과 다른 아키텍처
- TensorFlow 원본이라 NHWC 레이아웃 사용
- 입력 zp=0, min=0 → 입력이 양수만 존재 (spectrogram magnitude)
- 29-class vocab (영어 문자 수준)
- ai-sdk `examples/deepspeech2/`에 추론 예제 코드 있음

## 참고 문헌

- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) (Baidu, 2014)
