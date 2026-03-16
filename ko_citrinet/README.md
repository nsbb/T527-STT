# KoCitrinet — 한국어 음성인식 (T527 NPU)

NVIDIA NeMo CitriNet 기반 한국어 STT 모델. T527 NPU int8 양자화.

## 성능

| 평가 | CER | exact | 추론시간 | 샘플 수 |
|------|-----|-------|----------|---------|
| sample30 (mode7) | **44.44%** | 44/330 | 120ms | 330 |
| sample30 (mode1) | 45.94% | 46/330 | 120ms | 330 |
| sample30 (mode0) | 60.79% | 27/330 | 120ms | 330 |
| ONNX FP32 (일반 100샘플) | 36.03% | — | — | 100 |
| NB int8 (일반 100샘플) | 36.36% | — | — | 100 |

- 양자화 열화: +0.33%p (ONNX → NB)
- **mode7**이 최적: Slaney mel, log10, preEmph=0.97, reflect padding, energy window

### 예측 샘플 (modelhouse 실환경 월패드 녹음, NB int8)

| # | 정답 (GT) | NPU 예측 | CER |
|---|-----------|----------|-----|
| 0 | 관리사무소 전화번호가 어떻게 되지? | 서아 어떻게 되지 | 69% |
| 2 | 지난 공지사항 알려줘 | 지난 공지사항 알려줘 | **0%** |
| 5 | 오늘 경제뉴스 알려줘 | 오늘 경제 뉴스 알려줘 | **0%** |
| 6 | 집에서 회사까지 몇분 걸려? | 집에서 회사까지 몇 걸려 | 17% |
| 7 | 근처 우리동네 주유소 휘발유 평균 가격이 어때? | 근처 우리 동네 주유소 휘발유 평균 가격이 | 15% |
| 9 | 응 닫아줘 | 줘 | 75% |
| 10 | 나 지금 외출해, 엘리베이터 불러줘 | 나 지금 외출해 엘리베이터 불러 | 13% |
| 11 | 문 열어줘 | 누 알아줘 | 75% |
| 13 | 이번달에 관리비 얼마나 썼니? | 이번달에 관리비 얼마나 썼니 | 8% |
| 16 | 알림음 켜줘 | 알리면 사죠 | 80% |

> **경향**: 긴 문장(5글자+)은 비교적 잘 인식하지만, 짧은 문장(2~3글자)은 맥락 부족으로 오인식률 높음.
> 월패드 실환경 녹음(잡음, 마이크 품질)이라 깨끗한 음성 대비 CER 높음.

## 모델 스펙

| 항목 | 3초 (300f) | 5초 (500f) |
|------|-----------|-----------|
| NB 크기 | 62MB | 62MB |
| 입력 | `[1, 80, 1, 300]` int8 | `[1, 80, 1, 500]` int8 |
| 출력 | `[1, 2049, 1, 38]` int8 | `[1, 2049, 1, 63]` int8 |
| 입력 scale/zp | 0.02096 / -37 | 0.02096 / -37 |
| 출력 scale/zp | 0.11266 / 127 | 0.15194 / 127 |

## 아키텍처

- **모델**: CitriNet (1D Depthwise Separable Conv + Squeeze-and-Excitation)
- **Encoder**: 22개 Jasper 블록 (feat_in=80, filters=1024, SE 전부 활성화)
- **Decoder**: ConvASRDecoder (feat_in=1024, num_classes=2048)
- **디코딩**: CTC greedy + SentencePiece BPE
- **Vocab**: 2,048 BPE 토큰 + `<unk>` = 2,049개 (`vocab_ko.txt`)
- **토크나이저**: SentencePiece unigram (`tokenizer.model`)
- **학습 데이터**: Zeroth-Korean 등 한국어 음성 코퍼스

## 전처리 (mel-spectrogram)

```
16kHz mono WAV
  → preEmph=0.97 (mode7)
  → STFT: n_fft=512, win=25ms(400), hop=10ms(160), Hann window
  → 80 mel bins (Slaney normalization)
  → log10 (mode7) 또는 ln
  → per_feature normalize (mean/std)
  → [1, 80, 1, 300] 고정 (pad/truncate)
  → int8 양자화 (scale=0.02096, zp=-37)
```

## 파일 구조

```
ko_citrinet/
├── README.md                 # 이 파일
├── network_binary.nb         # 3초 NB (62MB, int8)
├── nbg_meta.json             # 양자화 파라미터
├── model_config_ko.yaml      # NeMo 모델 설정
├── tokenizer.model           # SentencePiece BPE 토크나이저
├── vocab_ko.txt              # 2,048 BPE 토큰
├── input_0_ref.dat           # 레퍼런스 입력 (동작 확인용)
├── 5s/                       # 5초 모델
│   ├── network_binary.nb     # 5초 NB (62MB, int8)
│   ├── nbg_meta.json
│   └── input_5s.dat
└── scripts/
    ├── run_pipeline_ko.sh             # 전체 파이프라인 (nemo → NB)
    ├── export_onnx_ko.py              # .nemo → ONNX
    ├── make_npy_dataset.py            # WAV → mel feature npy
    ├── run_one_wav_to_text_int8.sh    # WAV → 텍스트 (디바이스)
    ├── run_one_wav_to_text_server_int8.sh  # WAV → 텍스트 (서버)
    ├── decode_nb_output_ko.py         # NPU 출력 → 한국어 텍스트
    └── eval_test_cer.py               # CER 평가
```

## 사용법

### 디바이스 테스트 (vpm_run)

```bash
adb push network_binary.nb /data/local/tmp/ko_cit/
adb push input_0.dat /data/local/tmp/ko_cit/
adb shell "cd /data/local/tmp/ko_cit && LD_LIBRARY_PATH=/vendor/lib64 ./vpm_run_aarch64 -s sample.txt -b 0"
adb pull /data/local/tmp/ko_cit/output_0.dat .
python3 scripts/decode_nb_output_ko.py output_0.dat
```

### WAV → 텍스트 (디바이스 연결 필요)

```bash
bash scripts/run_one_wav_to_text_int8.sh /path/to/test.wav "정답 텍스트"
```

### 전체 파이프라인 (NeMo → NB 재빌드)

```bash
bash scripts/run_pipeline_ko.sh \
  MODEL_NEMO=/path/to/model.nemo \
  TRAIN_CSV=/path/to/train.csv \
  CALIB_COUNT=120 TEST_COUNT=20 \
  QTYPE=int8 ALGORITHM=moving_average MA_WEIGHT=0.004
```

## Android 앱

- **awaiasr_2**: `CitrinetTestActivity` — awnn API, processWithJavaMel()
- **android_stt_bundle_app**: `MainActivity` — VNN/OpenVX API (현재 주력)

## 변환 이력

1. NeMo `.nemo` → ONNX (`export_onnx_ko.py`)
2. ONNX → Pegasus import → int8 양자화 (120개 calibration, moving_average)
3. Pegasus export → network_binary.nb (62MB)
4. mel 전처리 버그 수정 (HTK→Slaney, log guard, periodic Hann)
5. vocab 1-token offset 버그 수정 (`<unk>` index 0 추가)
