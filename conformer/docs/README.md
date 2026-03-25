# Korean Conformer CTC — T527 NPU 변환 전체 기록

## 최종 결과

| 모델 | NB | 추론 | CER (100샘플) |
|------|-----|------|------|
| **KR Conformer CTC (cwwojin)** | **29MB** | **111ms/chunk** | **55.10%** |
| KoCitrinet 300f | 62MB | 120ms | 44.44% |
| Wav2Vec2 KO (NAS 80k) | 77MB | 424ms | ~100% |

- 모델: [cwwojin/stt_kr_conformer_ctc_medium](https://huggingface.co/cwwojin/stt_kr_conformer_ctc_medium)
- 31.8M params, 18 layers, d_model=256, 4 heads
- Vocab: 5001 (한국어 BPE 5000 + blank)
- 슬라이딩 윈도우: 301f window, 250f stride, 111ms/chunk

---

## 시행착오 전체 기록

### 1차 시도: 영어 Conformer small ONNX 수술 (실패)

NeMo `stt_en_conformer_ctc_small` (13.2M params)를 ONNX export 후 graph surgery:

1. NeMo export → opset 16, 3982 nodes
2. mel 추출 분리 (STFT 제거, mel 입력으로 변경)
3. opset 12 re-export
4. **Where op 48개 제거** (attention mask → Identity로 교체)
5. **Pad op 16개 수정** (빈 constant_value → 0.0)
6. onnxsim: 1647 nodes, 53MB

**Acuity import 성공 → NB 14MB, 74ms 추론** — 하지만 **FP32 ONNX에서 이미 garbage 출력.** graph surgery가 모델을 깨뜨린 것. 영어 WAV로 테스트해도 의미없는 영어 단어 나열.

### 2차 시도: SpeechBrain 한국어 Conformer (실패)

`speechbrain/asr-conformer-transformerlm-ksponspeech` (42.9M params):
- ONNX 134MB, 1438 nodes
- Acuity import: 성공
- Acuity quantize: 성공
- **Acuity export NB: error 64768 (모델 너무 큼)**

T527 NPU가 42.9M params를 처리 못 함.

### 3차 시도: cwwojin 한국어 Conformer (성공!)

`cwwojin/stt_kr_conformer_ctc_medium` (31.8M params):

#### 3-1. 다운로드

HuggingFace에서 128MB .nemo 파일 다운로드.
- wget: 5~6KB/s (6시간 예상)
- **aria2c -x16 -s16**: 16커넥션으로 ~30초 (하지만 file pre-allocation 버그로 trailing zeros 발생)
- **aria2c --file-allocation=none**: 해결

#### 3-2. ONNX export 1차 (실패)

torch.onnx.export (opset 12) → 4013 nodes → onnxsim → 1905 nodes.
Where 54개, Pad 18개 남아있어서 graph surgery 시도:

- Where 제거 (출력을 true branch로 rewire)
- Pad fix (empty constant_value → 0.0)
- Softmax axis=-1 → positive axis로 수정

**onnxruntime shape inference 에러.** Where 제거 후 텐서 rank 정보가 사라져서 Softmax axis 검증 실패.

#### 3-3. ONNX export 2차 — NeMo model.export() (성공하지만 garbage)

NeMo Docker에서 `model.export()` 사용. opset 16, dynamic shape로 정상 export.
static shape [1,80,301] 고정 + length 상수화 + onnxsim.

**FP32 ONNX inference 결과: garbage** ("erse uh sing answer..." 같은 의미없는 출력).

#### 3-4. mel 전처리 문제 발견

calibration mel을 librosa로 생성했는데, **librosa mel ≠ NeMo mel!**

```
NeMo mel range:    [-1.75, 5.14], mean=0.0000
librosa mel range: [-2.24, 4.89], mean=-0.0048
```

차이 원인: NeMo의 AudioToMelSpectrogramPreprocessor 내부 구현이 librosa와 다름 (dither, window, normalization 등).

**NeMo preprocessor로 mel 생성 후에도 여전히 FP32 garbage** → mel 문제가 아님.

#### 3-5. 핵심 발견: ONNX는 정상, vocab 매핑이 잘못

디버깅:
1. `m.transcribe(["test.wav"])` → **정상 한국어** ("평소 오전 아홉 시에서...")
2. `m.forward()` → **garbage** ("남았없##번장난그거했...")
3. encoder output hook으로 비교 → **완전 동일** (diff=0)
4. decoder output hook으로 비교 → **log_probs 완전 동일** (diff=0)
5. **그런데 디코딩 결과만 다름!**

**원인: .nemo에 들어있는 `vocab.txt` 파일의 순서 ≠ SentencePiece tokenizer의 실제 ID 매핑.**

`vocab.txt`로 디코딩하면 garbage, `tok.ids_to_text()`로 디코딩하면 정상.

#### 3-6. 추가 발견: dither와 pad_to

`m.transcribe()` 소스 코드 분석 결과:
```python
self.preprocessor.featurizer.dither = 0.0   # 기본값 1e-5 → noise 제거
self.preprocessor.featurizer.pad_to = 0     # 기본값 != 0 → padding 제거
```

mel 생성 시 이 설정을 해야 transcribe()와 동일한 mel이 나옴.

#### 3-7. 최종 파이프라인 (성공)

```
NeMo Docker:
  1. model.export() → ONNX (opset 16, dynamic shape)
  2. preprocessor(dither=0, pad_to=0)로 NeMo mel 생성
  3. tokenizer에서 올바른 vocab mapping 추출 → vocab_correct.json

WSL:
  4. ONNX static shape [1,80,301] + length 상수화
  5. onnxsim: 4462 → 1905 nodes
  6. Pad op 18개 empty constant_value → 0.0 수정

Acuity Docker:
  7. import → kr_v2.json
  8. quantize uint8 KL divergence (NeMo mel calibration 10개)
  9. export ovxlib → 29MB NB

T527 디바이스:
  10. vpm_run → 111ms/chunk, 한국어 출력 동작!
```

**핵심: ONNX graph surgery 불필요. NeMo export 그대로 사용 + Pad fix만 하면 됨. Where op 54개가 남아있어도 Acuity가 처리함.**

---

## 교훈

### 1. ONNX graph surgery는 위험

Where/Pad op을 수동으로 제거/수정하면 shape inference가 깨져서 모델이 망가짐. 가능하면 원본 export를 최대한 유지하고, Acuity가 자체 처리하게 맡기는 것이 안전.

### 2. vocab.txt ≠ tokenizer ID

NeMo BPE 모델의 .nemo 파일에 들어있는 `vocab.txt`는 SentencePiece vocab 파일이지 tokenizer의 실제 ID→token 매핑이 아님. **반드시 `tokenizer.ids_to_tokens()` 또는 `tokenizer.ids_to_text()`로 디코딩해야 함.**

### 3. librosa mel ≠ NeMo mel

같은 파라미터(n_fft=512, hop=160, n_mels=80)로 만들어도 결과가 다름. NeMo preprocessor로 생성한 mel만 사용해야 함. 또한 `dither=0.0`, `pad_to=0` 설정 필수.

### 4. Conformer는 uint8에서 동작한다

5001 vocab (BPE)에도 불구하고 T527 uint8에서 한국어 출력. CNN 컴포넌트(depthwise conv, conv subsampling)가 양자화에 강해서 중간 feature를 보존.

| 아키텍처 | uint8 결과 |
|---------|-----------|
| 순수 CNN (KoCitrinet) | CER 44.44% |
| CNN+Attention (Conformer) | CER 55.10% |
| 순수 Transformer (Wav2Vec2) | CER ~100% |

---

## 결과 파일

| 파일 | 내용 |
|------|------|
| `kr_cwwojin_t527_full_100.csv` | 100샘플 전체 음성 슬라이딩 윈도우 결과 (CER 55.10%) |
| `kr_cwwojin_t527_full_results.csv` | 20샘플 전체 음성 결과 (CER 57.14%) |
| `kr_cwwojin_t527_results.csv` | 20샘플 3초 truncated 결과 (CER 88.31%) |

## 파일 구조

```
conformer/
├── README.md                               # 이 문서
├── kr_cwwojin_t527_full_100.csv            # 100샘플 전체 결과
├── kr_cwwojin_t527_full_results.csv        # 20샘플 전체 결과
├── kr_cwwojin_t527_results.csv             # 20샘플 3초 결과
└── (ai-sdk 쪽)
    kr_cwwojin/
    ├── stt_kr_conformer_ctc_medium.nemo    # 원본 모델 (123MB)
    ├── model_nemo_export.onnx              # NeMo export (dynamic, opset 16)
    ├── model_acuity_v2.onnx                # static [1,80,301] + Pad fix (126MB)
    ├── vocab_correct.json                  # tokenizer ID→token 매핑 (5001)
    ├── nemo_mel_full/                      # NeMo mel 전체 (100개)
    ├── nemo_calib/                         # NeMo mel 3초 (20개)
    ├── wksp_v2_nbg_unify/
    │   ├── network_binary.nb              # T527 NPU NB (29MB)
    │   └── nbg_meta.json
    └── patch_relpos.py, patch_relpos_v2.py # RelPos 패치 시도 (불필요였음)
```

## 다음 단계

1. **SungBeom 모델 (122.5M)** — 더 큰 모델, NB export 가능 여부 확인
2. **입력 길이 확장** — 5초(501f) 또는 10초(1001f) NB로 재변환 시 슬라이딩 오버헤드 감소
3. **Android 앱 통합** — NeMo mel 전처리를 JNI로 구현
4. **FP32 ONNX CER 측정** — 양자화 손실 정량화 (FP32 CER vs uint8 CER)
