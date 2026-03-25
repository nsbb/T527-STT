# SungBeom Conformer CTC — 배포 파라미터 전체 정리

Android 앱 또는 임베디드 환경에서 이 모델을 돌리기 위해 필요한 파라미터 전부.

---

## 1. NB 파일

| 양자화 | 파일 | 크기 | 추론/chunk |
|--------|------|------|----------|
| **uint8 KL (권장)** | `wksp_nbg_unify/network_binary.nb` | **102MB** | **233ms** |
| uint8 MA | `wksp_uint8_ma_nbg_unify/network_binary.nb` | 102MB | 233ms |
| int16 DFP | `wksp_int16_nbg_unify/network_binary.nb` | 200MB | 564ms |

---

## 2. 입출력 양자화 파라미터

### uint8 (KL / MA 공통)

**입력 양자화 (float → uint8):**
```
uint8_value = clamp(round(float_value / scale) + zero_point, 0, 255)

scale      = 0.02418474107980728
zero_point = 67
```

**출력 역양자화 (uint8 → float):**
```
float_value = (uint8_value - zero_point) * scale

scale      = 0.2030104696750641
zero_point = 255
```

### int16 DFP

**입력 양자화 (float → int16):**
```
int16_value = clamp(round(float_value * 2^fl), -32768, 32767)

fl = 12  (즉 scale = 1/4096 = 0.000244)
```

**출력 역양자화 (int16 → float):**
```
float_value = int16_value / 2^fl

fl = 9  (즉 scale = 1/512 = 0.001953)
```

---

## 3. 입출력 텐서

| | shape | dtype | 크기 (bytes) |
|---|---|---|---|
| 입력 | `[1, 80, 301]` | uint8 | 24,080 |
| 출력 | `[1, 76, 2049]` | uint8 | 155,724 |

- 입력: 80 mel bins × 301 time frames
- 출력: 76 CTC frames × 2049 vocab (log softmax)
- 301 mel frames ≈ **3.01초** 음성
- subsampling factor 4 → 301/4 ≈ 76 output frames

---

## 4. mel spectrogram 전처리 (가장 중요!)

**librosa로 만들면 안 됨. NeMo AudioToMelSpectrogramPreprocessor와 동일하게 구현해야 함.**

### 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| sample_rate | 16000 | 입력 오디오 샘플링 레이트 |
| n_fft | 512 | FFT window size |
| win_length | 400 | 0.025초 × 16000 = 400 samples |
| hop_length | 160 | 0.01초 × 16000 = 160 samples |
| n_mels | 80 | mel filterbank 개수 |
| fmin | 0 | mel 최소 주파수 |
| fmax | 8000 | mel 최대 주파수 (= sample_rate / 2) |
| window | hann | Hann window |
| dither | **0.0** | 추론 시 반드시 0 (학습 시 1e-5) |
| pad_to | **0** | 추론 시 반드시 0 |

### 처리 순서

```
1. 오디오 로드 (16kHz mono float32, range [-1, 1])
2. STFT (n_fft=512, hop=160, win=400, hann window)
3. Power spectrum → mel filterbank (80 bins, 0~8000Hz)
4. Log: ln(mel + 1e-5)
5. Per-feature normalize:
     각 mel bin별로 (80개 각각):
       mean = bin의 전체 time frames 평균
       std  = bin의 전체 time frames 표준편차
       normalized = (value - mean) / (std + 1e-5)
6. 결과: [1, 80, T] float32 (T = 오디오 길이에 따라 가변)
7. 3초 윈도우: T를 301로 자르거나 패딩
```

### NeMo mel vs librosa mel 차이

동일한 오디오에 대해:
```
NeMo mel range:    [-1.75, 5.14], mean=0.0000
librosa mel range: [-2.24, 4.89], mean=-0.0048
```

차이 원인:
- NeMo는 내부적으로 STFT 구현이 librosa와 다름 (center padding, window normalization 등)
- per_feature normalize 구현 차이
- **이 차이가 모델 정확도에 치명적** — librosa mel 넣으면 garbage 출력

### mel 생성 참조 코드 (Python, NeMo)

```python
import nemo.collections.asr as nemo_asr
import torch

model = nemo_asr.models.EncDecCTCModelBPE.restore_from("model.nemo", map_location="cpu")
model.eval()
model.preprocessor.featurizer.dither = 0.0   # 필수!
model.preprocessor.featurizer.pad_to = 0     # 필수!

# audio: float32 tensor [1, num_samples]
# length: int64 tensor [1] = num_samples
mel, mel_len = model.preprocessor(input_signal=audio, length=length)
# mel: [1, 80, T] float32
```

### C/JNI 구현 시 주의

NeMo preprocessor를 C로 포팅할 때 반드시 확인할 점:
1. STFT window: periodic Hann (not symmetric)
2. mel filterbank: HTK 또는 Slaney (NeMo 기본은 Slaney)
3. log: natural log (ln), NOT log10
4. normalize: per-feature (각 mel bin별), NOT global
5. padding: center=True일 수 있음 (NeMo 버전에 따라 다름)

**검증 방법:** NeMo에서 생성한 mel npy파일과 C 구현 출력을 비교하여 max diff < 0.01 확인.

---

## 5. CTC 디코딩

### Vocab

| 항목 | 값 |
|------|-----|
| 총 토큰 수 | 2049 |
| BPE 토큰 | 0 ~ 2047 (2048개) |
| Blank (CTC) | **2048** |
| UNK | 0 (`<unk>`) |
| Tokenizer | SentencePiece BPE |

### vocab 매핑 파일

**반드시 `vocab_correct.json` 사용.** .nemo 안의 `vocab.txt`는 tokenizer ID 순서와 다름!

```json
{
  "0": "<unk>",
  "1": "▁이",
  "2": "▁그",
  ...
  "2047": "꿨",
  "2048": "<blank>"
}
```

`▁` (U+2581) = 단어 시작 (공백 위치 표시).

### CTC Greedy Decode 알고리즘

```
1. output [76, 2049]에서 각 frame별 argmax → token_ids [76]
2. 연속 중복 제거: [A, A, B, B, B, C] → [A, B, C]
3. blank(2048) 제거: [A, blank, B, C] → [A, B, C]
4. token_ids → vocab_correct.json으로 텍스트 변환
5. ▁ 토큰은 앞에 공백 추가: "▁이" → " 이"
```

---

## 6. 슬라이딩 윈도우 (3초 이상 음성 처리)

NB 입력이 301 mel frames (≈ 3초) 고정이므로 긴 음성은 슬라이딩 윈도우로 처리.

### 파라미터

| 항목 | 값 | 설명 |
|------|-----|------|
| window | 301 frames | ≈ 3.01초 |
| stride | 250 frames | ≈ 2.50초 |
| overlap | 51 frames | ≈ 0.51초 |
| output_stride | 63 frames | 76 × 250/301 ≈ 63 |

### 알고리즘

```
mel_full = [1, 80, T]  (전체 음성)

chunks = []
start = 0
while start < T:
    end = start + 301
    chunk = mel_full[:, :, start:end]
    if chunk.width < 301: zero-pad
    chunks.append(chunk)
    if end >= T: break
    start += 250

for each chunk:
    quantize → NPU 추론 → dequantize → logits [76, 2049]

merge:
    chunk 0: logits[:63]    (처음 63 frames)
    chunk 1: logits[:63]
    ...
    chunk N: logits[:76]    (마지막은 전체 76 frames)

concatenate → CTC decode
```

### 음성 길이별 chunks / 추론시간

| 음성 | chunks | 추론 (uint8) | 추론 (int16) |
|------|--------|------------|------------|
| 3초 | 1 | 233ms | 564ms |
| 5초 | 2 | 466ms | 1.1초 |
| 8초 | 3 | 699ms | 1.7초 |
| 10초 | 4 | 932ms | 2.3초 |
| 15초 | 6 | 1.4초 | 3.4초 |
| 20초 | 8 | 1.9초 | 4.5초 |

---

## 7. 파일 체크리스트

앱에 탑재해야 할 파일:

| 파일 | 크기 | 용도 |
|------|------|------|
| `network_binary.nb` | 102MB | NPU 모델 |
| `nbg_meta.json` | 1KB | 양자화 파라미터 (scale, zp) |
| `vocab_correct.json` | 60KB | token ID → 텍스트 매핑 |

mel 전처리는 코드로 구현 (파일 불필요).
SentencePiece .model 파일은 CTC greedy decode만 하면 불필요 (vocab_correct.json으로 충분).
