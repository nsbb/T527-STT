# QAT 전체 실험 요약

**기간:** 2026-03-26 ~ 2026-03-29

---

## 1. 실험 총괄 테이블

| # | 실험명 | 데이터 | 데이터 개수 | 음성 시간 | Train/Val/Test | Epoch | 학습 GPU | 학습 소요 | Best val_loss | T527 CER |
|---|--------|--------|-----------|----------|---------------|-------|---------|----------|--------------|----------|
| 0 | **PTQ (baseline)** | - | - | - | - | - | - | - | - | **10.02%** |
| 1 | AIHub 100k QAT | AIHub (샘플링) | 100,000개 | 84시간 | 95k/5k/- | 10 (0~9) | RTX 6000 Ada × 1 | ~2시간 | 0.1416 (ep04) | **5.3% (final)** |
| 2 | 자체데이터 QAT | 월패드 자체녹음 | 368개 | - | 294/36/38 | 30 (0~29) | RTX 6000 Ada × 1 | ~3분 | 0.0797 (ep13) | **~6% (ep13)** |
| 3 | **AIHub 전체 QAT** | AIHub (전체) | 4,307,477개 | 4,356시간 | 4.09M/215k/- | 10 (0~9) | RTX 6000 Ada × 4 | **~42시간** | 0.0692 (ep08,09) | **테스트 필요** |

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

## 4. 핵심 교훈

1. **val_loss ≠ T527 CER** — val_loss best(ep04)가 CER 100%였음. 반드시 디바이스 테스트 필요.
2. **데이터 양이 중요** — 100k(84hr) val_loss 0.14 vs 전체(4356hr) val_loss 0.069. 데이터 많을수록 QAT 효과 큼.
3. **적은 데이터로도 효과 있음** — 294개로도 CER 12% → 6% (2배 개선). 도메인 매칭이 중요.
4. **DDP + 적은 val 데이터 = validation 실패** — GPU당 val 샘플 < batch_size이면 validation 스킵됨.
5. **오버피팅 시점 확인 필수** — 자체 데이터는 epoch 13에서, AIHub 전체는 epoch 8~9에서 수렴.
