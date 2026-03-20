# 원격 NAS 데이터로 로컬 다운로드 없이 GPU 학습하기

## 요약

NAS 서버에 수 TB의 음성 데이터가 있을 때, **로컬 디스크에 복사하지 않고** 네트워크를 통해 직접 읽으면서 GPU 학습이 가능하다.

핵심 도구: **SSHFS** (SSH Filesystem) — SSH 연결을 통해 원격 디렉토리를 로컬 폴더처럼 마운트.

---

## 왜 필요한가

| 상황 | 문제 |
|------|------|
| 학습 데이터 4356시간 (수 TB WAV) | 로컬 SSD에 안 들어감 |
| NAS에만 데이터 있음 | `scp`로 복사하면 며칠 걸림 |
| 여러 실험마다 다른 데이터 조합 | 매번 복사 불가능 |

**SSHFS 해결**: NAS를 로컬 폴더로 마운트 → `open()`, `read()`가 SSH 터널을 통해 원격 파일을 읽음 → 로컬 디스크 사용량 **0**.

---

## 설치 및 마운트

### 1. SSHFS 설치

```bash
sudo apt install -y sshfs sshpass
```

### 2. NAS 마운트

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/nas03
sudo chown $USER:$USER /mnt/nas03

# SSHFS 마운트 (비밀번호 방식)
sshfs user@192.168.110.108:/nas03/data_audio/ /mnt/nas03/ \
  -o password_stdin,allow_other,StrictHostKeyChecking=no \
  <<< "비밀번호"

# 확인
ls /mnt/nas03/
```

### 3. Docker에서 사용

```bash
docker run --gpus all \
  -v /mnt/nas03:/nas03:ro \
  -v $(pwd):/workspace \
  wav2vec2-ko-train:v1 \
  python /workspace/train.py --data_csv /nas03/train.csv
```

Docker 컨테이너 안에서 `/nas03/` 경로로 NAS 파일에 직접 접근 가능.

---

## 학습 코드에서의 사용법

### 방법 1: SSHFS 마운트 + 일반 파일 읽기

SSHFS 마운트 후에는 **코드 변경 없이** 로컬 파일과 동일하게 접근:

```python
import soundfile as sf

# NAS의 파일을 마치 로컬 파일처럼 읽기
audio, sr = sf.read("/mnt/nas03/135.명령어/B-M2642M.wav")
# → 실제로는 SSH를 통해 원격 파일을 읽음
# → 로컬 디스크 사용량 0
```

### 방법 2: HuggingFace Datasets Streaming

공개 데이터셋의 경우, 다운로드 없이 스트리밍:

```python
from datasets import load_dataset

# streaming=True → 로컬 저장 없이 실시간 스트리밍
dataset = load_dataset(
    "kresnik/zeroth_korean",
    split="train",
    streaming=True  # ← 핵심
)

for sample in dataset:
    audio = sample["audio"]["array"]  # numpy array, 실시간 디코딩
    text = sample["text"]
    # 학습 로직...
```

### 방법 3: PyTorch DataLoader + SSHFS

SSHFS 마운트 + PyTorch DataLoader 조합:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf

class NASAudioDataset(Dataset):
    def __init__(self, csv_path, nas_mount="/mnt/nas03"):
        self.df = pd.read_csv(csv_path)
        self.nas_mount = nas_mount

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # CSV의 /nas03/ 경로를 마운트 경로로 변환
        wav_path = row["raw_data"].replace("/nas03/data_audio", self.nas_mount)
        audio, sr = sf.read(wav_path)
        text = row["transcript"]
        return audio, text

# num_workers > 0 으로 prefetch하면 네트워크 지연 숨김
loader = DataLoader(dataset, batch_size=8, num_workers=4, prefetch_factor=2)
```

---

## 성능 고려사항

### 네트워크 속도 vs 학습 속도

| 항목 | 값 |
|------|-----|
| 기가비트 이더넷 실측 | ~100 MB/s |
| WAV 1개 평균 (3초, 16kHz) | ~96 KB |
| batch 8개 읽기 | ~768 KB → **~0.008초** |
| GPU 학습 1 step | ~0.3~1.0초 |
| **병목** | **GPU 연산 (네트워크 아님)** |

**결론: 기가비트 네트워크면 네트워크가 병목이 되지 않는다.** GPU가 1 step 계산하는 동안 DataLoader가 다음 batch를 prefetch.

### 최적화 팁

```python
DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,      # 병렬 로딩 (SSHFS도 병렬 OK)
    prefetch_factor=2,  # 미리 2 batch 로딩
    pin_memory=True,    # GPU 전송 최적화
)
```

### 주의사항

1. **WiFi는 느림** — 유선 기가비트 이더넷 권장
2. **SSHFS 연결 끊김** — 학습 중 NAS 재부팅 시 끊어짐. `autossh` 또는 `-o reconnect` 옵션 사용
3. **랜덤 접근 패턴** — DataLoader shuffle 시 랜덤 파일 읽기. SSHFS 캐시(`-o cache=yes`)로 완화
4. **NAS 부하** — 여러 사람이 동시에 NAS 사용 시 느려질 수 있음

---

## 마운트 안정성 옵션

```bash
# 재연결 + 캐시 + 타임아웃 설정
sshfs user@server:/path /mnt/point \
  -o reconnect \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o cache=yes \
  -o cache_timeout=600 \
  -o allow_other
```

---

## 언마운트

```bash
fusermount -u /mnt/nas03
# 또는
sudo umount /mnt/nas03
```

---

## 실제 적용 예시: Wav2Vec2 한국어 Fine-tuning

```
NAS 서버 (192.168.110.108)
  └── /nas03/data_audio/    ← 4356시간 한국어 음성 (수 TB)
  └── /nas04/nlp_sk/STT/    ← 학습 CSV (486만 행)
        │
        │ SSHFS (SSH 터널, 기가비트 이더넷)
        │
로컬 WSL (RTX 4070 Super 16GB)
  ├── /mnt/nas03/  ← NAS 마운트 (디스크 사용 0)
  ├── /mnt/nas04/  ← NAS 마운트 (디스크 사용 0)
  └── Docker (wav2vec2-ko-train:v1)
        ├── GPU 학습 (PyTorch + Transformers)
        ├── DataLoader(num_workers=4) → /mnt/nas03/ 에서 WAV 읽기
        └── 학습된 모델 → 로컬 저장 (~1.5GB)
```

**로컬 디스크 사용**: 모델 체크포인트 ~1.5GB만. 학습 데이터 0.
