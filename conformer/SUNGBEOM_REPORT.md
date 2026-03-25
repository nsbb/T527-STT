# SungBeom Conformer CTC Medium — T527 NPU uint8 실험 보고서

## 1. 모델 개요

| 항목 | 값 |
|------|-----|
| HuggingFace | [SungBeom/stt_kr_conformer_ctc_medium](https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium) |
| 프레임워크 | NVIDIA NeMo |
| 아키텍처 | Conformer CTC (CNN + Self-Attention 하이브리드) |
| 파라미터 | **122.5M** |
| Vocab | 2049 (한국어 BPE 2048 + blank) |
| Tokenizer | SentencePiece BPE |
| 학습 데이터 | AI Hub 한국어 음성 |
| FP32 정확도 | NeMo transcribe()로 검증 — 매우 높은 인식률 |

### Encoder 구조

| 항목 | 값 |
|------|-----|
| Layers | 18 |
| d_model | 512 |
| Attention heads | 8 |
| FF expansion | 4 |
| Conv kernel | 31 |
| Self-attention | rel_pos (Relative Positional Encoding) |
| Subsampling | striding, factor=4 |

### Preprocessor (mel spectrogram)

| 항목 | 값 |
|------|-----|
| Sample rate | 16000 Hz |
| Features (n_mels) | 80 |
| n_fft | 512 |
| Window size | 0.025s (400 samples) |
| Window stride | 0.01s (160 samples) |
| Dither | 1e-5 (추론 시 0.0으로 설정 필수) |
| Pad to | 0 (추론 시 0으로 설정 필수) |

### 입출력

| | Shape | 설명 |
|---|---|---|
| 입력 | `[1, 80, 301]` | mel spectrogram (3초, 16kHz) |
| 출력 | `[1, 76, 2049]` | log softmax (76 frames, 2049 vocab) |

301 mel frames ≈ 3.01초. Subsampling factor 4 → 301/4 ≈ 76 output frames.

---

## 2. T527 NPU 결과

### 핵심 수치

| 항목 | 값 |
|------|-----|
| NB 크기 | **102MB** |
| 추론 시간 | **233ms/chunk** |
| 양자화 | uint8 asymmetric_affine KL divergence |
| **CER (100샘플)** | **10.02%** |
| CER 중앙값 | 8.70% |
| CER 0% 샘플 | 4개 |
| CER < 5% | 24개 |
| CER < 10% | 64개 |
| CER < 20% | 97개 |
| CER > 50% | 2개 |
| Best | #11 CER 0.0% |
| Worst | #1 CER 82.0% |

### 모델 비교

| 모델 | Params | NB | 추론 | CER | 비고 |
|------|--------|-----|------|-----|------|
| **SungBeom Conformer** | **122.5M** | **102MB** | **233ms** | **10.02%** | **최고 정확도** |
| cwwojin Conformer | 31.8M | 29MB | 111ms | 55.10% | 빠르지만 정확도 낮음 |
| KoCitrinet 300f | ~10M | 62MB | 120ms | 44.44% | 기존 운용 모델 |
| Wav2Vec2 KO (NAS 80k) | 94.4M | 77MB | 424ms | ~100% | 실패 |

### 왜 SungBeom이 cwwojin보다 훨씬 좋은가

| | SungBeom | cwwojin |
|---|---|---|
| Params | **122.5M** | 31.8M |
| d_model | **512** | 256 |
| Heads | **8** | 4 |
| Vocab | **2049** | 5001 |
| Conv kernel | **31** | ? |

1. **4배 큰 모델** — d_model 512 vs 256로 표현력이 훨씬 높음
2. **Vocab 절반** — 2049 vs 5001로 토큰 간 logit 구분이 용이 → uint8 양자화에 유리
3. **학습 데이터** — AI Hub 데이터의 양과 질 차이 가능

### CER 높은 샘플 분석

CER > 20% 상위 5개:

| # | dur | CER | 원인 추정 |
|---|-----|-----|----------|
| 1 | 20.5s | 82.0% | 매우 긴 문장 (8 chunks), 슬라이딩 윈도우 경계에서 정보 손실 |
| 46 | 8.5s | 63.5% | 전문 용어 ("성공회대 노동아카데미 주임교수") 인식 실패 |
| 49 | 8.6s | 29.0% | 숫자 읽기 ("일 조 이천 구백 십 구 억원") 혼동 |
| 1 | 20.5s | 82.0% | 첫 3초 인식 실패 + 이후 누적 오류 |
| 10 | 8.5s | 19.4% | 비일상 단어 ("곡괭이", "쇠 파이프") |

**CER 높은 주요 원인:**
1. 슬라이딩 윈도우 경계에서 단어 잘림/반복
2. 전문 용어, 고유명사, 숫자 읽기
3. 긴 문장에서 누적 오류

### CER 0% 완벽 인식 샘플

| # | GT |
|---|---|
| 11 | 하지만 싫증 나면 버리면 그만이라는 식의 사고방식은 단지 물건에만 영향을 미치는 데서 그치지 않습니다 |
| 26 | 손재주가 별로 없는 사람들은 처음에 자주 상처를 입었을 것입니다 |
| 42 | 미국에서는 흑인이나 히스패닉이 고위직에 진출할 생각도 하지 않는다 |
| 96 | 조사해봤지만 트럼프의 주장을 뒷받침할 만한 정보를 찾지 못했다는 것이다 |

---

## 3. 변환 파이프라인 상세

### 3-1. 다운로드

```
HuggingFace: SungBeom/stt_kr_conformer_ctc_medium
파일: stt_kr_conformer_ctc_medium.nemo (491MB)
다운로드: aria2c -x16 -s16 --file-allocation=none (약 30초)
```

**주의:** `aria2c` 기본 `--file-allocation=prealloc`은 파일 끝에 trailing zeros를 남겨 torch.load 실패 유발. 반드시 `--file-allocation=none` 사용.

### 3-2. NeMo Docker에서 ONNX export

```bash
docker run --rm \
  -v $WORK:/workspace/model \
  nvcr.io/nvidia/nemo:23.06 \
  python3 -c "
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecCTCModelBPE.restore_from('model.nemo', map_location='cpu')
    m.eval()
    m.preprocessor.featurizer.dither = 0.0  # 필수!
    m.preprocessor.featurizer.pad_to = 0    # 필수!
    m.export('model.onnx')  # NeMo native export (opset 16, dynamic shape)
  "
```

**핵심: `dither=0.0`, `pad_to=0` 설정 필수.** NeMo `transcribe()` 내부에서 이 설정을 하기 때문에, mel 생성 시에도 동일하게 해야 함.

### 3-3. Vocab 매핑 추출

```python
# NeMo Docker 내에서
vocab_map = {}
for i in range(vocab_size):
    vocab_map[i] = tok.ids_to_tokens([i])[0]
json.dump(vocab_map, open("vocab_correct.json", "w"))
```

**주의:** .nemo 파일 안의 `vocab.txt` 순서 ≠ SentencePiece tokenizer의 실제 ID 매핑. 반드시 `tok.ids_to_tokens()`로 추출해야 함.

### 3-4. NeMo mel 생성

```python
# NeMo Docker 내에서
m.preprocessor.featurizer.dither = 0.0
m.preprocessor.featurizer.pad_to = 0
mel, ml = m.preprocessor(input_signal=audio_tensor, length=length_tensor)
np.save("mel.npy", mel.numpy())  # [1, 80, T]
```

**주의:** librosa로 mel을 만들면 NeMo와 결과가 다름 (range, normalization 등). 반드시 NeMo preprocessor 사용.

### 3-5. ONNX static shape 고정 + onnxsim

```python
import onnx
from onnx import TensorProto, helper
from onnxsim import simplify

m = onnx.load("model.onnx")  # 4462 nodes, opset 16

# 1) Input shape 고정: [1, 80, 301]
for inp in m.graph.input:
    if inp.name == "audio_signal":
        inp.type.tensor_type.shape.dim[0].dim_value = 1
        inp.type.tensor_type.shape.dim[1].dim_value = 80
        inp.type.tensor_type.shape.dim[2].dim_value = 301

# 2) length input → 상수 301
length_init = helper.make_tensor("length", TensorProto.INT64, [1], [301])
m.graph.initializer.append(length_init)
# length를 graph input에서 제거
inputs = [i for i in m.graph.input if i.name != "length"]
del m.graph.input[:]
m.graph.input.extend(inputs)

# 3) onnxsim
m2, ok = simplify(m)  # 4462 → 1905 nodes
```

### 3-6. Pad op 수정

```python
from onnx import numpy_helper
import numpy as np

# Conformer의 relative positional encoding에서 사용하는 Pad op 18개의
# constant_value가 빈 텐서('')로 되어있어 Acuity 6.12가 처리 못 함
pad_const = "__pad_zero__"
m2.graph.initializer.append(
    numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=pad_const))
for p in m2.graph.node:
    if p.op_type == "Pad" and len(p.input) >= 3 and p.input[2] == '':
        p.input[2] = pad_const  # 18개 수정
```

**Where op 54개는 수정하지 않음.** Acuity가 자체 처리함. graph surgery로 Where를 제거하면 shape inference가 깨져서 오히려 모델이 망가짐.

### 3-7. Acuity import

```bash
pegasus import onnx \
  --model model_acuity.onnx \
  --output-model sb.json --output-data sb.data
# Error(0), Warning(1)
```

### 3-8. Calibration 데이터 준비

NeMo preprocessor로 생성한 mel을 301 frames로 truncate:

```python
mel = np.load("nemo_mel/mel_0000.npy")  # [1, 80, T] full length
mel_trunc = mel[:, :, :301]
if mel_trunc.shape[2] < 301:
    mel_trunc = np.pad(mel_trunc, ((0,0),(0,0),(0, 301 - mel_trunc.shape[2])))
np.save("calib/calib_0000.npy", mel_trunc)  # [1, 80, 301]
```

calibration 10개 사용.

### 3-9. Acuity quantize

```bash
pegasus quantize \
  --model sb.json --model-data sb.data \
  --device CPU --with-input-meta inputmeta.yml \
  --rebuild-all \
  --model-quantize sb_uint8.quantize \
  --quantizer asymmetric_affine --qtype uint8 \
  --algorithm kl_divergence \
  --batch-size 1
# Error(0), Warning(0)
```

### inputmeta.yml

```yaml
input_meta:
  databases:
  - path: /work/dataset.txt
    type: TEXT
    ports:
    - lid: audio_signal_1507      # import 후 sb.json에서 확인
      category: frequency
      dtype: float32
      sparse: false
      layout: nchw
      shape: [1, 80, 301]
      fitting: scale
      preprocess:
        reverse_channel: false    # 오디오는 반드시 false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm: [0, 1, 2]
      redirect_to_output: false
```

### 3-10. Acuity export NB

```bash
cd /acuity612/bin  # vxcode/template/ 찾기 위해 bin에서 실행
export REAL_GCC=/usr/bin/gcc
./pegasus export ovxlib \
  --model sb.json --model-data sb.data \
  --dtype quantized --model-quantize sb_uint8.quantize \
  --with-input-meta inputmeta.yml \
  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk $VSIM --target-ide-project linux64 --batch-size 1 \
  --output-path wksp/
# Error(0), Warning(0)
# → network_binary.nb (102MB)
```

**122.5M params인데 NB export 성공!** 이전 SpeechBrain 42.9M이 실패(error 64768)했던 것과 대비. 같은 Conformer 아키텍처인데 NeMo 모델은 되고 SpeechBrain은 안 된 이유는 ONNX 그래프 구조 차이 (node 수, op 종류 등).

### 양자화 파라미터

| | scale | zero_point | range |
|---|---|---|---|
| 입력 mel `[1,80,301]` u8 | 0.02418 | 67 | [-1.63, 4.54] |
| 출력 logprobs `[1,76,2049]` u8 | 0.20301 | 255 | [-51.77, 0.0] |

출력 uint8 step = 0.203. 이는 logit margin min이 0.203보다 크면 argmax가 보존됨을 의미.

---

## 4. 테스트 방법

### 슬라이딩 윈도우

전체 음성을 3초(301f) 윈도우로 분할하여 T527 NPU에서 각각 추론 후 결과 연결:

```
Window: 301 mel frames (≈ 3초)
Stride: 250 mel frames (≈ 2.5초)
Overlap: 51 mel frames (≈ 0.5초)

Output stride: 76 * 250/301 ≈ 63 frames per chunk
마지막 chunk: 전체 76 frames 사용
```

### CER 계산

```
CER = edit_distance(NPU 출력, GT) / len(GT) * 100
```

- 공백 제거 후 문자 단위 비교
- `<unk>` 토큰은 제거 후 비교
- NPU 출력의 마침표(`.`) 등 구두점도 제거

### 테스트셋

- Zeroth-Korean test split 100개
- 16kHz mono WAV
- 뉴스/책 낭독체
- 길이: 5.8초 ~ 20.5초 (평균 ~10초)

---

## 5. 파일 구조

```
conformer/
├── SUNGBEOM_REPORT.md                      # 이 문서
├── kr_sungbeom_t527_full_100.csv          # 100샘플 결과 (id순)
├── kr_sungbeom_t527_sorted_by_cer.csv     # 100샘플 결과 (CER순)
├── kr_cwwojin_t527_full_100.csv           # cwwojin 100샘플 결과
├── README.md                              # Conformer 전체 시행착오 기록
└── (ai-sdk 쪽)
    kr_sungbeom/
    ├── stt_kr_conformer_ctc_medium.1.nemo # 원본 모델 (468MB)
    ├── model.onnx                         # NeMo export (dynamic, opset 16)
    ├── model_acuity.onnx                  # static [1,80,301] + Pad fix (477MB)
    ├── vocab_correct.json                 # tokenizer ID→token (2049)
    ├── nemo_mel/                          # NeMo mel 전체길이 (100개)
    ├── calib/                             # NeMo mel 3초 truncated (10개)
    ├── sb.json, sb.data                   # Acuity import 결과
    ├── sb_uint8.quantize                  # Acuity quantize 결과
    ├── inputmeta.yml, dataset.txt         # Acuity 설정
    └── wksp_nbg_unify/
        ├── network_binary.nb             # T527 NPU NB (102MB)
        └── nbg_meta.json

## 6. 재현 방법

### 전제 조건

- NeMo Docker: `nvcr.io/nvidia/nemo:23.06`
- Acuity Docker: `t527-npu:v1.2` + Acuity 6.12.0 + VivanteIDE 5.7.2
- T527 디바이스 (adb 연결, vpm_run_aarch64 설치)

### 전체 명령 순서

```bash
# 1. 다운로드
aria2c -x16 -s16 --file-allocation=none \
  -o model.nemo \
  "https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium/resolve/main/stt_kr_conformer_ctc_medium.nemo"

# 2. NeMo Docker: export + vocab + mel
docker run --rm -v $WORK:/workspace nvcr.io/nvidia/nemo:23.06 python3 export.py

# 3. WSL: static shape + onnxsim + Pad fix
python3 fix_onnx.py

# 4. Acuity Docker: import → quantize → export NB
docker run --rm -v $WORK:/work -v $ACUITY:/acuity -v $VIV:/viv t527-npu:v1.2 bash pipeline.sh

# 5. T527: vpm_run 테스트
adb push network_binary.nb /data/local/tmp/
adb shell "LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0"
```
