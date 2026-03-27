# QAT 실험 로그 — 2026-03-27

**서버:** gpu-112 (RTX 6000 Ada × 4), gpu-111, gpu-114
**SLURM 클러스터:** gpu-111 ~ gpu-114

---

## 1. 실험 결과 요약

| # | 실험 | 데이터 | Train/Val/Test | Epoch | T527 CER | 비고 |
|---|------|--------|---------------|-------|----------|------|
| 0 | PTQ (양자화만, 기존) | - | - | - | 10.02% | baseline |
| 1 | AIHub 100k QAT final | 95,000개 (84hr) | 95k/5k | 10 (final) | **5.3%** | best |
| 2 | AIHub 100k QAT ep04 | 95,000개 (84hr) | 95k/5k | 4 (best val_loss) | **100%** | 수렴 안 됨 |
| 3 | 자체데이터 QAT ep13 | 294개 | 294/36/38 | 13 (best val_loss) | **~6%** | 2배 개선 |
| 4 | AIHub 전체 QAT | 4,092,104개 (4,356hr) | 4.09M/215k | 진행 중 | - | gpu-112 |

---

## 2. 핵심 발견: val_loss ≠ T527 CER

### 현상

AIHub 100k QAT에서:
- epoch 4: val_loss **0.1416** (best) → T527 CER **100%** (완전 실패)
- epoch 10: val_loss **0.151** (worse) → T527 CER **5.3%** (best)

**val_loss가 낮다고 T527 CER이 낮은 게 아니다.**

### 원인

- val_loss는 **FP32 + FakeQuantize** 환경에서 측정
- T527 CER은 **실제 uint8 하드웨어**에서 측정
- FakeQuantize는 이상적인 uint8 시뮬레이션이고, 실제 NPU는 tiling, accumulator, LUT 근사 등에서 차이 발생
- 이전에도 "Acuity 시뮬레이션과 디바이스 argmax 일치율 31.5%"로 확인된 문제

### 교훈

- **val_loss만 보고 best checkpoint를 고르면 안 된다**
- 여러 epoch의 모델을 **T527 디바이스에서 직접 CER 측정**해서 비교해야 한다
- val_loss는 참고 지표일 뿐, 최종 판단은 디바이스 테스트

### 반면 자체 데이터 QAT에서는

- epoch 13: val_loss **0.0797** (best) → T527 CER **~6%** (best)
- val_loss와 T527 CER이 일치

이 차이의 원인은 불명확. 가설:
- 자체 데이터가 적어서(294개) val_loss가 더 안정적
- AIHub 대용량에서는 FakeQuantize 시뮬레이션 오차가 누적
- epoch 4에서는 QAT가 아직 양자화 적응을 충분히 못 한 상태

---

## 3. QAT 실험 상세

### 3.1 AIHub 100k QAT

**데이터:**
- 소스: `/nas04/nlp_sk/STT/data/train/base/train_base_4356hr.csv`
- 전체 4,860,928행에서 100,000개 랜덤 샘플링 (seed=42)
- 필터: CER=0.0 (whisper 검증 통과), duration 0.5~15초
- Train: 95,000 / Val: 5,000
- 총 84.63시간

**하이퍼파라미터:**
- epochs: 10, lr: 1e-5, batch_size: 16
- MarginLoss: target=0.3, lambda=0.1
- GPU: RTX 6000 Ada × 1 (SLURM, gpu-108)
- 소요: 약 2시간

**val_loss 추이:**
- epoch 3: 0.1446
- epoch 4: 0.1416 (best val_loss, 근데 T527 CER 100%)
- epoch 5: 0.1438
- epoch 10 (final): 0.151 (T527 CER 5.3%)

**출력 파일:**
```
qat_aihub_output/
├── conformer_qat_100k_84hr_final.nemo    # epoch 10, T527 CER 5.3%
├── conformer_qat_100k_84hr_final.onnx
├── conformer_qat_100k_best_ep04.nemo     # epoch 4, T527 CER 100%
├── conformer_qat_epoch=03_val_loss=0.1446.ckpt
├── conformer_qat_epoch=04_val_loss=0.1416.ckpt
└── conformer_qat_epoch=05_val_loss=0.1438.ckpt
```

### 3.2 자체 녹음 데이터 QAT

**데이터:**
- 소스: 월패드 자체 녹음 (7F_HJY, 7F_KSK, modelhouse_2m, modelhouse_2m_noheater, modelhouse_3m)
- 총 368개, 80/10/10 분리
- Train: 294 / Val: 36 / Test: 38

**val_loss 추이 (100 epoch 실험):**
```
Epoch  0~9:   0.427 → 0.100  (급격히 감소)
Epoch 10~20:  0.109 → 0.051  (계속 감소)
Epoch 20~42:  0.051 → 0.044  (수렴, 최저 0.0435 @epoch42)
Epoch 42~99:  0.044~0.056    (정체, 약간 흔들림)
```

**best: epoch 13, val_loss 0.0797 → T527 CER ~6%**
- epoch 13 이후 val_loss 상승 시작 (오버피팅)
- 100 epoch은 과도, 20~30이면 충분

**출력 파일:**
```
qat_custom_30ep_output/
├── conformer_qat_best_ep13.nemo     # best, T527 CER ~6%
├── conformer_qat_best_ep13.onnx
├── conformer_qat_final.nemo         # epoch 30
├── conformer_qat_final.onnx
└── conformer_qat_epoch=*.ckpt       # epoch 12,13,17,21,24
```

### 3.3 AIHub 전체 QAT (진행 중)

**데이터:**
- 전체 4,307,477개, 필터 후 train 4,092,104 / val 215,373
- 약 4,356시간

**설정:**
- GPU: RTX 6000 Ada × 4 (DDP, SLURM job 6618, gpu-112)
- epochs: 10, lr: 1e-5, batch_size: 16
- 1 epoch ≈ 3.5시간, 전체 ≈ 1.5일

**현재 상태:**
- Epoch 0 진행 중 (~74%)
- val_loss: 0.113 (epoch 0 중간 validation)
- 이미 100k final(0.151)보다 낮음

---

## 4. 인프라 관련 교훈

### 4.1 SLURM 클러스터 구조

- gpu-106~110: 별도 클러스터 (login 노드에서 보이는 것)
- gpu-111~114: 별도 클러스터 (서버에 직접 SSH 해야 보임)
- 같은 `sinfo`라도 어느 서버에서 치느냐에 따라 다른 클러스터가 보임

### 4.2 DDP + 적은 데이터 = validation 실패

- 자체 데이터 val이 36개인데 GPU 4개로 DDP하면 GPU당 9개
- batch_size 16이면 1 step도 안 돼서 validation 자체가 스킵됨
- 해결: 적은 데이터는 GPU 1개로 돌려야 함

### 4.3 nohup vs SLURM

- nohup: 간단하지만 GPU 수동 지정 필요, 서버별 SSH 필요
- SLURM: 자동 배정, 관리 편함, 근데 InvalidAccount 에러 가끔 발생 (잠시 기다리면 해결)

### 4.4 Multi-GPU 속도

| 설정 | 속도 | 1 epoch |
|------|------|---------|
| GPU 1개 (nohup) | 1.31 it/s | ~13.5시간 |
| GPU 4개 (DDP, SLURM) | 4~5 it/s | ~3.5시간 |
| 속도 향상 | ~3.5배 | |

완벽한 4배는 아님 (DDP 통신 오버헤드)

---

## 5. CSV 데이터 구조 정리

### train_base_4356hr.csv

| column | 이름 | 설명 |
|--------|------|------|
| [0] | raw_data | wav 파일 경로 |
| [1] | before_whisper | whisper 추론 텍스트 |
| [2] | transcript | 원본 정답 라벨 |
| [3] | time | 음성 길이 (초) |
| [4] | cer | [1]과 [2]의 CER (품질 필터용) |
| [5] | answer_tmp | [1]의 공백 제거 버전 |
| [6] | inf_tmp | [2]의 공백 제거 버전 |
| [7~9] | file, answer, inf | 미사용 |

- CER=0.0: whisper 추론 = 원본 정답 → 깨끗한 데이터
- CER>0: whisper와 원본이 다름 → 라벨 불확실
- QAT에는 CER=0.0만 사용 (before_whisper = transcript)

### 다른 CSV 파일

```
/nas04/nlp_sk/STT/data/train/base/
├── train_base_4356hr.csv              ← 사용 중
├── with_cmd_2166hr.csv                # 명령어 포함
├── with_cmd_2166hr_checked_1868.2hr.csv  # 명령어 검증본
├── without_cmd_2190hr.csv             # 명령어 제외
├── whisper_v3_inferred_50k.csv        # whisper v3 추론
└── question_balanced_whisper_v3.csv   # 질문 균형

/nas04/nlp_sk/STT/data/
└── train_9000hr.csv                   # 9000시간 (10,054,680행)
```

---

## 6. CER 성능 지표 정리

### 기본 지표

| 지표 | 설명 |
|------|------|
| CER | edit_distance / ref_len |
| edit_distance | S + I + D |
| substitution | 글자 치환 |
| insertion | 글자 삽입 |
| deletion | 글자 삭제 |
| ref_len | 정답 글자수 |
| hyp_len | 추론 글자수 |
| exact | 완전 일치 여부 (0/1) |
| contains_ref | 정답 포함 여부 (0/1) |

### 짧은 명령어에서 CER의 한계

```
긴 문장 (20자): 1글자 틀려도 CER = 5%
짧은 명령어 (3자): 1글자 추가되면 CER = 33%
```

- 짧은 문장일수록 CER이 과대 측정됨
- 명령어 인식에는 **contains_ref**가 더 실용적
- **exact**(sentence accuracy)는 한 글자만 틀려도 0이라 가혹함

### substitution의 한계

```
GT: 전화번호
HYP: 전호번하 → S=2
HYP: 전쉑번콱 → S=2 (같은 S인데 체감 심각도 다름)
```

- edit distance는 "몇 글자 다른지"만 셈, "얼마나 다른지"는 안 봄

---

## 7. 향후 실험 계획

### 즉시
- [ ] AIHub 전체 QAT 완료 대기 (예상 1.5일)
- [ ] 완료 후 여러 epoch의 .nemo 추출 → T527 CER 직접 측정 (val_loss 믿지 말고)
- [ ] AIHub 100k final(5.3%) vs AIHub 전체 비교

### 단기
- [ ] 커스텀 .quantize 파이프라인 구축 (로드맵 Step 3)
- [ ] 레이어별 mixed precision 실험

### 중기
- [ ] 양자화 수학 손계산 (로드맵 Step 1)
- [ ] 논문 읽기 (Jacob 2018, Nagel 2021)
