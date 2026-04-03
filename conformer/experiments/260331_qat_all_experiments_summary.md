# QAT 전체 실험 요약

**기간:** 2026-03-26 ~ 2026-04-03

---

## 1. 실험 총괄 테이블

| # | 실험명 | 데이터 | 데이터 개수 | 음성 시간 | Train/Val/Test | Epoch | 학습 GPU | 학습 소요 | Best val_loss | T527 CER (18k avg_all) |
|---|--------|--------|-----------|----------|---------------|-------|---------|----------|--------------|----------|
| 0 | **PTQ (baseline)** | - | - | - | - | - | - | - | - | 15.59% |
| 1 | AIHub 100k QAT | AIHub (샘플링) | 100,000개 | 84시간 | 95k/5k/- | 10 (0~9) | RTX 6000 Ada × 1 | ~2시간 | 0.1416 (ep04) | 9.30% |
| 2 | 자체데이터 QAT | 월패드 자체녹음 | 368개 | - | 294/36/38 | 30 (0~29) | RTX 6000 Ada × 1 | ~3분 | 0.0797 (ep13) | - |
| 3 | AIHub 전체 QAT (ep09) | AIHub (전체) | 4,307,477개 | 4,356시간 | 4.09M/215k/- | 10 (0~9) | RTX 6000 Ada × 4 | ~42시간 | 0.0692 (ep08,09) | 12.85% (real만) |
| 4 | KD m0.5 (100k) | AIHub (샘플링) | 100,000개 | 84시간 | 95k/5k/- | 10 (0~9) | RTX 6000 Ada × 1 | ~2시간 | - | 8.40% (real만) |
| **5** | **AIHub 1M QAT** | **AIHub (샘플링)** | **1,000,000개** | **~840시간** | **950k/50k/-** | **1** | **RTX 6000 Ada × 1** | **~2시간** | **-** | **8.86%** |
| 6 | AIHub 전체 QAT (ep00) | AIHub (전체) | 4,307,477개 | 4,356시간 | 4.09M/215k/- | 0 (초기) | RTX 6000 Ada × 4 | - | 0.102 | 테스트 중 |
| 7 | KD m1.0 (100k) | AIHub (샘플링) | 100,000개 | 84시간 | 95k/5k/- | 10 (0~9) | RTX 6000 Ada × 1 | - | - | 테스트 중 |

---

## 2. 각 실험 상세

### 실험 1: AIHub 100k QAT

- **소스:** `train_base_4356hr.csv`에서 100,000개 랜덤 샘플링 (seed=42)
- **필터:** CER=0.0, duration 0.5~15초
- **하이퍼파라미터:** lr=1e-5, batch=16, MarginLoss (target=0.3, lambda=0.1)
- **서버:** gpu-108 (SLURM), GPU 1개

**val_loss 추이:**

| epoch | val_loss |
|-------|----------|
| 3 | 0.1446 |
| 4 | **0.1416 (best)** |
| 5 | 0.1438 |
| 9 (final) | 0.151 |

**T527 CER:**

| 모델 | T527 CER |
|------|----------|
| ep04 (best val_loss) | **100%** (완전 실패!) |
| ep09 (final), unseen 38개 | **5.3%** (소량 테스트) |
| ep09 (final), **18K avg_all** | **9.30%** (정식 결과) |

**교훈:** val_loss best ≠ T527 CER best. 반드시 디바이스 테스트 필요.

**출력 파일:**
```
qat_aihub_output/
├── conformer_qat_100k_84hr_final.nemo      # ep09 final, CER 5.3%
├── conformer_qat_100k_84hr_final.onnx
├── conformer_qat_100k_best_ep04.nemo       # ep04 best val, CER 100%
├── conformer_qat_epoch=03_val_loss=0.1446.ckpt
├── conformer_qat_epoch=04_val_loss=0.1416.ckpt
└── conformer_qat_epoch=05_val_loss=0.1438.ckpt
```

---

### 실험 2: 자체 녹음 데이터 QAT

- **소스:** 월패드 자체 녹음 데이터 (7F_HJY, 7F_KSK, modelhouse_2m, modelhouse_2m_noheater, modelhouse_3m)
- **총 368개**, 80/10/10 분리 → Train 294 / Val 36 / Test 38
- **하이퍼파라미터:** lr=1e-5, batch=16, MarginLoss (target=0.3, lambda=0.1)
- **서버:** gpu-114 (SLURM), GPU 1개

**val_loss 추이 (100 epoch 실험에서 확인):**

| 구간 | val_loss |
|------|----------|
| epoch 0~9 | 0.427 → 0.100 (급격 감소) |
| epoch 10~20 | 0.109 → 0.051 (계속 감소) |
| epoch 20~42 | 0.051 → 0.0435 (수렴) |
| epoch 42~99 | 0.044~0.056 (정체) |

**Best: epoch 13, val_loss 0.0797 → T527 CER ~6%**

**주의사항:**
- DDP 4GPU + val 36개 → validation 스킵됨 (GPU당 9개 < batch 16)
- 해결: GPU 1개로 학습해야 val_loss 정상 기록
- 데이터 294개로 100 epoch 하면 완전 오버피팅 (train_loss 마이너스까지)

**출력 파일:**
```
qat_custom_30ep_output/
├── conformer_qat_best_ep13.nemo     # best, CER ~6%
├── conformer_qat_best_ep13.onnx
├── conformer_qat_final.nemo         # ep29 final
├── conformer_qat_final.onnx
└── conformer_qat_epoch=*.ckpt       # ep12,13,17,21,24
```

---

### 실험 3: AIHub 전체 QAT

- **소스:** `train_base_4356hr.csv` 전체
- **필터:** CER=0.0, duration 0.5~15초, 005.명령어 음성 경로 제외
- **필터 후:** 4,307,477개 → Train 4,092,104 / Val 215,373
- **하이퍼파라미터:** lr=1e-5, batch=16, MarginLoss (target=0.3, lambda=0.1)
- **서버:** gpu-112 (SLURM), GPU 4개 (DDP)
- **SLURM Job ID:** 6618
- **학습 시작:** 2026-03-27 11:00, **완료:** 2026-03-29 05:37 (**약 42시간**)

**val_loss 추이 (epoch 끝 기준):**

| epoch | val_loss | 감소폭 |
|-------|----------|--------|
| 0 | 0.1020 | - |
| 1 | 0.0893 | -1.27%p |
| 2 | 0.0846 | -0.47%p |
| 3 | 0.0792 | -0.54%p |
| 4 | 0.0755 | -0.37%p |
| 5 | 0.0723 | -0.32%p |
| 6 | 0.0709 | -0.14%p |
| 7 | 0.0695 | -0.14%p |
| 8 | **0.0692** | -0.03%p |
| 9 | **0.0692** | 0.00%p |

**epoch 8~9에서 완전 수렴. 오버피팅 없음.**

**출력 파일:**
```
qat_aihub_full_output/
├── conformer_qat_full_best_ep08.nemo       # ep08, val_loss 0.0692
├── conformer_qat_full_best_ep09.nemo       # ep09, val_loss 0.0692
├── conformer_qat_full_ep09_final.nemo      # ep09 final (= best)
├── conformer_qat_full_ep09_final.onnx
├── conformer_qat_epoch=08_val_loss=0.0692.ckpt
├── conformer_qat_epoch=09_val_loss=0.0692.ckpt
├── conformer_qat_epoch=09_val_loss=0.0693.ckpt
└── conformer_qat_epoch=00_val_loss=0.1128.ckpt
```

---

## 3. T527 테스트 대기 목록

| # | 파일 | 설명 | 위치 |
|---|------|------|------|
| 1 | `conformer_qat_full_best_ep08.nemo` | AIHub 전체, ep08 | `qat_aihub_full_output/` |
| 2 | `conformer_qat_full_best_ep09.nemo` | AIHub 전체, ep09 | `qat_aihub_full_output/` |

**T527 테스트 완료:**

| 모델 | T527 CER (unseen 38) | T527 avg_real (368) | 비고 |
|------|---------------------|--------------------|----|
| PTQ (baseline) | 13.3% | 16.44% | |
| AIHub 100k final (ep09) | 5.3% | 7.24% (aihub100 calib) | **최고** |
| 자체데이터 best (ep13) | ~6% | | |
| **AIHub 전체 ep09** | | **12.85% (real100 calib)** | 100k보다 나쁨 |
| **AIHub 전체 ep09** | | **14.81% (aihub100 calib)** | 100k보다 나쁨 |

**AIHub 전체 QAT 결론:** val_loss는 좋지만(0.069) 실제 T527 CER은 100k보다 나쁨. 43배 더 많은 gradient step으로 모델이 과적합.
**calibration 실험:** calib 10개→100개 필수. calib 소스는 다양할수록 좋음 (aihub > real).

---

### 실험 5: AIHub 1M QAT (1 epoch) — 새 최고 성능

- **소스:** `train_base_4356hr.csv`에서 1,000,000개 랜덤 샘플링 (seed=42)
- **필터:** CER=0.0, duration 0.5~15초
- **하이퍼파라미터:** lr=1e-5, batch=16, CosineAnnealingLR, MarginLoss (target=0.3, lambda=0.1)
- **핵심 아이디어:** 100k×10ep = 59,380 steps ≈ 1M×1ep = 59,375 steps. **동일 step, 데이터 10배 다양**
- **서버:** gpu (SLURM 6641), GPU 1개

**T527 CER (18,368 샘플):**

| 평균 | 100k (실험1) | 1m_ep01 (실험5) | 차이 |
|------|-----------|---------|------|
| avg_real | 7.24% | **6.86%** | **-0.38%p** |
| avg_aihub | 11.01% | **10.53%** | **-0.48%p** |
| avg_all | 9.30% | **8.86%** | **-0.43%p** |

**11개 중 10개 데이터셋에서 개선. "다양한 데이터 1번 > 적은 데이터 10번 반복" 가설 검증.**

**출력 파일:**
```
qat_1m_1ep_output/
├── conformer_qat_aihub1m_margin0.3_ep01.nemo
├── experiment_config.yaml
└── wksp_nbg_unify_nbg_unify/network_binary.nb (102MB)
```

---

### 실험 6: AIHub 전체 QAT (ep00, 초기 체크포인트)

- **소스:** 실험 3과 동일 (AIHub 전체 4.3M)
- **ep00:** 학습 시작 직후 체크포인트 (val_loss 0.102)
- **목적:** ep09(과적합)보다 ep00(초기)이 나을 수 있는지 확인

**T527 CER:** 테스트 진행 중

**출력 파일:**
```
qat_aihub_full_output/
├── conformer_qat_aihubfull_margin0.3_ep00.nemo
└── wksp_nbg_unify_nbg_unify/network_binary.nb (102MB)
```

---

### 실험 7: KD + Margin 1.0 (100k)

- **소스:** 실험 1과 동일 데이터 (AIHub 100k)
- **변경:** margin 0.5 → **1.0**, KD lambda=0.5, temp=2.0
- **Loss:** CTC + 0.1 × MarginLoss + 0.5 × KD_Loss
- **batch:** 32 (기존 16에서 증가)
- **목적:** KD가 CTC 정확도를 보존해주므로 margin을 더 세게 줄 수 있다는 가설 검증

**T527 CER:** 테스트 진행 중

**출력 파일:**
```
qat_kd_100k_margin1.0_output/
├── conformer_qat_aihub100k_kd_margin1.0_ep09.nemo
├── experiment_config.yaml
└── wksp_nbg_unify_nbg_unify/network_binary.nb (102MB)
```

---

## 4. 핵심 교훈

1. **val_loss ≠ T527 CER** — val_loss best(ep04)가 CER 100%였음. 반드시 디바이스 테스트 필요.
2. **데이터 다양성 > 반복** — 동일 step 수에서 1M×1ep(8.86%)이 100k×10ep(9.30%)보다 우수. 다양한 데이터 1번 > 같은 데이터 10번.
3. **과적합 주의** — AIHub 전체 10ep(12.85%)는 100k 10ep(9.30%)보다 나쁨. 데이터 많아도 epoch 과하면 역효과.
4. **적은 데이터로도 효과 있음** — 294개로도 CER 12% → 6% (2배 개선). 도메인 매칭이 중요.
5. **DDP + 적은 val 데이터 = validation 실패** — GPU당 val 샘플 < batch_size이면 validation 스킵됨.
6. **Calibration 개수 중요** — 10개 → 100개로 늘리면 CER 대폭 개선. 소스보다 개수가 중요.
