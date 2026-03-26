# Conformer QAT 서버 학습 지시서

## 목표

SungBeom Conformer CTC Medium 모델을 AIHub 데이터로 QAT (Quantization-Aware Training) 학습.
T527 NPU uint8 양자화 성능을 개선하는 것이 목표.

## 현재 상태

| | CER |
|---|---|
| FP32 (양자화 없음) | 9.93% |
| uint8 PTQ (현재 최고) | 10.59% |
| uint8 QAT (Zeroth 데이터) | 11.11% (도메인 불일치로 악화) |

**양자화 손실 0.66%p.** AIHub 데이터로 QAT하면 이 손실을 줄일 수 있음.

## 서버에 있는 파일

```
kr_sungbeom/
├── stt_kr_conformer_ctc_medium.1.nemo   # 원본 모델 (468MB)
├── stt_kr_conformer_ctc_medium.nemo     # 동일 모델 복사본
├── train_qat.py                         # QAT 학습 스크립트
├── vocab_correct.json                   # BPE 토큰 매핑 (2049)
└── qat_output/                          # 이전 Zeroth QAT 결과 (참고용)
```

## Step 1: 환경 설정

```bash
conda create -n nemo python=3.10 -y
conda activate nemo
pip install 'nemo_toolkit[asr]'
pip install einops
```

Python 3.10 필수 (3.8은 NeMo 최신 버전 문법 에러).

## Step 2: AIHub manifest 생성

NeMo 학습에는 manifest JSON 파일이 필요. 한 줄에 하나씩:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "정답 텍스트", "duration": 3.5}
```

AIHub 데이터 위치 확인 후 manifest 생성 스크립트:

```python
import os, json, soundfile as sf

DATA_DIR = "/nas04/nlp_sk/STT/data/train/"  # 실제 경로 확인 필요
OUTPUT = "aihub_train_manifest.json"

with open(OUTPUT, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith(".wav"): continue
            wav_path = os.path.join(root, fname)
            # 텍스트 파일 찾기 (같은 이름 .txt 또는 상위 폴더의 label)
            txt_path = wav_path.replace(".wav", ".txt")
            if not os.path.exists(txt_path): continue
            text = open(txt_path, encoding="utf-8").read().strip()
            if not text: continue
            try:
                audio, sr = sf.read(wav_path)
                duration = len(audio) / sr
            except: continue
            if duration < 0.5 or duration > 20.0: continue
            entry = {"audio_filepath": wav_path, "text": text, "duration": round(duration, 2)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Manifest saved: {OUTPUT}")
```

**주의:** AIHub 데이터 구조에 따라 텍스트 파일 찾는 로직을 수정해야 할 수 있음. CSV, JSON, 별도 label 파일 등 다양한 형식 가능.

train/val 분리도 필요:
- train: 전체의 95%
- val: 전체의 5% (또는 별도 validation set)

## Step 3: QAT 학습 실행

```bash
conda activate nemo

python3 train_qat.py \
  --nemo-path stt_kr_conformer_ctc_medium.1.nemo \
  --train-manifest aihub_train_manifest.json \
  --val-manifest aihub_val_manifest.json \
  --output-dir qat_aihub_output \
  --epochs 10 \
  --lr 1e-5 \
  --batch-size 16 \
  --num-workers 8 \
  --use-margin-loss \
  --margin-target 0.3 \
  --margin-lambda 0.1
```

### 파라미터 설명

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| epochs | 10 | QAT는 수렴 빠름, 10이면 충분 |
| lr | 1e-5 | 원본 학습보다 낮게 (fine-tune이므로) |
| batch-size | 16 | GPU 메모리에 따라 조절 (V100 16GB → 16, A100 → 32) |
| use-margin-loss | | CTC loss + margin penalty 동시 사용 |
| margin-target | 0.3 | top1-top2 logit 차이를 0.3 이상으로 유지 (uint8 step ~0.2의 1.5배) |
| margin-lambda | 0.1 | margin loss 가중치 |

### GPU 메모리 부족 시

```bash
--batch-size 8 --gradient-accumulation 2  # effective batch = 16
```

(train_qat.py에 gradient_accumulation 옵션 추가 필요할 수 있음)

### 예상 시간

| GPU | batch 16 | 1 epoch | 10 epochs |
|-----|---------|---------|-----------|
| RTX 4070 | ~4 it/s | ~20분 (22k samples) | ~3시간 |
| V100 | ~6 it/s | ~15분 | ~2.5시간 |
| A100 | ~10 it/s | ~10분 | ~1.5시간 |

AIHub 데이터가 22k보다 크면 비례해서 늘어남.

## Step 4: 학습 모니터링

```bash
# 학습 진행 확인
tail -f qat_aihub_output/qat_train.log

# 핵심 지표:
# - train_loss: 감소해야 함 (0.2 → 0.02)
# - val_loss: 감소 또는 유지 (0.08 이하면 좋음)
# - margin_mean: 7~9 유지
# - margin_min: 0 → 점점 올라가면 좋음
```

## Step 5: 학습 완료 후

학습이 끝나면 `qat_aihub_output/` 폴더에:
- `conformer_qat_epoch=XX_val_loss=X.XXXX.ckpt` — checkpoint들
- `conformer_qat_final.nemo` — 최종 모델

**WSL로 가져올 파일:**
```bash
# best checkpoint의 .nemo 또는 final .nemo
scp 서버:kr_sungbeom/qat_aihub_output/conformer_qat_final.nemo /home/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom/qat_aihub_output/
```

WSL에서:
1. NeMo Docker로 QAT 모델 ONNX export
2. static shape + Pad fix
3. Acuity uint8 KL quantize
4. NB export → T527 테스트

## train_qat.py 동작 설명

### FakeQuantize

학습 중 3곳에 uint8 양자화 시뮬레이션 삽입:
1. **Encoder 입력** (mel spectrogram) — 입력 양자화
2. **Encoder 출력** — 중간 feature 양자화
3. **Decoder 출력** (logits) — 출력 양자화

Forward: float → quantize → dequantize → float (양자화 에러 포함)
Backward: STE (Straight-Through Estimator) — gradient 그대로 통과

### MarginLoss

```
loss = CTC_loss + 0.1 × margin_loss
margin_loss = ReLU(0.3 - (top1_logit - top2_logit)).mean()
```

top1-top2 logit 차이가 0.3 미만이면 penalty → 모델이 더 확신 있게 예측하도록 유도 → uint8 양자화 후에도 argmax 보존.

### CNN freeze

Conformer의 conv subsampling (pre_encode)은 양자화 문제 없으므로 freeze.
Transformer encoder + decoder만 학습.

## 주의사항

1. **FP32 학습** (fp16 안 됨) — FakeQuantize가 fp16과 비호환
2. **dither=0.0, pad_to=0** — train_qat.py에서 자동 설정됨
3. **val_loss가 올라가면** lr을 낮추거나 early stop
4. **데이터 경로 주의** — manifest의 audio_filepath가 실제 파일과 일치하는지 확인
