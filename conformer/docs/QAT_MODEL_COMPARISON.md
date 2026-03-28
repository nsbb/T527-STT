# QAT 모델 비교 — SungBeom Conformer CTC on T527 NPU

## 모델 목록

| 모델 | 학습 데이터 | 에폭 | NB 크기 | NB 경로 |
|------|---------|------|-------|-------|
| **PTQ** (baseline) | 없음 (calibration 10개만) | — | 102MB | `wksp_nbg_unify/` |
| **QAT ailab ep13** | 자체 ailab 데이터 | 13 (best checkpoint) | 102MB | `qat_ailab/wksp_conformer_qat_best_ep13_nbg_unify/` |
| **QAT ailab final** | 자체 ailab 데이터 | 최종 | 102MB | `qat_ailab/wksp_conformer_qat_final_nbg_unify/` |
| **QAT AIHub 100k final** | AIHub 100K (84시간) | 10 | 102MB | `qat_aihub_output/wksp_100k_nbg_unify/` |
| **QAT AIHub 100k best** | AIHub 100K (84시간) | best (ep04) | 102MB | `qat_aihub_output/wksp_100k_best_nbg_unify/` |

### QAT 학습 설정
- FakeQuantize 3곳: encoder input, encoder output, decoder output
- MarginLoss: target=0.3, lambda=0.1
- lr=1e-5, batch 16
- 서버: nsbb@192.168.110.108

---

## 1. Unseen Test Split (자체 데이터 38개) — QAT 모델 간 비교

| 데이터셋 | 샘플 | PTQ | QAT ailab ep13 | QAT ailab final | QAT AIHub 100k final |
|--------|------|-----|---------------|----------------|---------------------|
| 7F_HJY | 13 | 10.3% | 11.1% | 12.0% | — |
| 7F_KSK | 11 | 18.9% | 3.8% | 3.8% | — |
| modelhouse_2m | 3 | 13.6% | 9.1% | 9.1% | — |
| modelhouse_noheater | 5 | 13.2% | 3.8% | 3.8% | — |
| modelhouse_3m | 6 | 9.7% | 3.2% | 3.2% | — |
| **AVG (unseen 38개)** | **38** | **13.3%** | **6.4%** | **6.7%** | **5.3%** |

> QAT AIHub 100k best (ep04): CER 103.6% — garbage, 사용 불가.
> val_loss가 낮아도 실제 NB 성능과 무관함을 확인.

---

## 2. 18,368개 전체 테스트 — PTQ vs QAT AIHub 100k final

### 데이터셋별 비교 (CER 오름차순, QAT 기준)

| 데이터셋 | 분류 | 샘플 | PTQ CER | QAT CER | 개선 |
|--------|------|------|---------|---------|------|
| 7F_KSK | 자체 | 108 | 21.20% | **3.29%** | **-17.91%p** |
| modelhouse_noheater | 자체 | 51 | 7.49% | **4.21%** | -3.28%p |
| 012.상담음성 | AIHub | 3000 | 10.03% | **7.86%** | -2.17%p |
| 7F_HJY | 자체 | 107 | 15.64% | **8.60%** | -7.04%p |
| modelhouse_2m | 자체 | 51 | 16.14% | **8.80%** | -7.34%p |
| eval_other | AIHub | 3000 | 13.30% | **11.00%** | -2.30%p |
| eval_clean | AIHub | 3000 | 16.01% | **12.10%** | -3.91%p |
| 007.저음질 | AIHub | 3000 | 19.01% | **12.54%** | -6.47%p |
| 009.한국어_강의 | AIHub | 3000 | 15.66% | **13.03%** | -2.63%p |
| 010.회의음성 | AIHub | 3000 | 15.29% | **13.79%** | -1.50%p |
| modelhouse_3m | 자체 | 51 | 21.73% | **14.70%** | -7.03%p |

### 평균 비교

| 평균 | PTQ CER | QAT CER | 개선 |
|------|---------|---------|------|
| avg_aihub (AIHub 6개) | 14.88% | **11.72%** | **-3.16%p** |
| avg_real (자체 5개) | 16.44% | **7.92%** | **-8.52%p** |
| avg_all (전체 11개) | 15.59% | **9.99%** | **-5.60%p** |

---

## 3. FP32 vs PTQ vs QAT 종합 비교

| 모델 | Zeroth 100 CER | unseen 38 CER | 18k avg CER |
|------|---------------|---------------|-------------|
| FP32 ONNX | 9.93% | — | — |
| PTQ uint8 KL | 10.59% | 13.3% | 15.59% |
| QAT ailab ep13 | — | 6.4% | — |
| QAT ailab final | — | 6.7% | — |
| **QAT AIHub 100k final** | — | **5.3%** | **9.99%** |

---

## 4. 분석

### QAT AIHub가 자체 데이터에서 더 효과적인 이유
- AIHub 데이터로 학습했는데 자체 데이터(real) 개선 폭이 더 큰 이유:
  1. **자체 데이터 샘플 수 적음** (51~108개) — 소수 샘플에서 개선 효과가 평균에 크게 반영
  2. **PTQ에서 자체 데이터가 더 나빴음** (16.44% vs 14.88%) — 바닥이 낮으니 회복 폭 큼
  3. **자체 데이터 = 짧은 명령어** ("엘리베이터 불러줘") — QAT로 양자화 정밀도 올라가면 짧은 발화에서 효과 큼

### QAT best vs final 교훈
- QAT AIHub 100k **best** (ep04, val_loss 최소): CER **103.6%** — 완전 garbage
- QAT AIHub 100k **final** (ep10, 마지막): CER **5.3%** — 최고 성능
- **교훈: val_loss ≠ 실제 NB 성능. 반드시 NB 변환 후 디바이스에서 직접 테스트해야 함.**

### 개선 한계
- AIHub 데이터셋(3000개씩)은 개선 폭이 1.5~6.5%p로 안정적
- 자체 데이터(51~108개)는 편차가 큼 (3.3%p ~ 17.9%p)
- 전체 평균 CER 9.99% — **10% 벽 돌파**

---

## 5. 결과 CSV 위치

### PTQ (18k)
```
conformer/results/conformer_*.csv (11개)
```

### QAT AIHub 100k final (18k)
```
conformer/results/conformer_qat100k_*.csv (11개)
```

### QAT 모델 비교 (unseen 38개)
```
conformer/results/qat_comparison_test_split.csv
```
