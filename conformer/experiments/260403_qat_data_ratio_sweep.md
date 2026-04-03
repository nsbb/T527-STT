# QAT 데이터 비율 sweep 실험

**날짜:** 2026-04-03
**목적:** 1 epoch 고정, 데이터 양만 변경하여 최적 QAT 데이터 비율 찾기
**원본 모델:** SungBeom Conformer (13,946시간, 10,916,423개로 1 epoch 학습)
**우리 데이터:** AIHub train_base_4356hr.csv (4,307,477개, 4,356시간)

---

## 실험 설계

모든 실험: 1 epoch, lr=1e-5, batch=16, margin_target=0.3, margin_lambda=0.1

| # | 데이터 | 개수 | 시간 | 원본 대비 | Steps | GPU | 상태 |
|---|--------|------|------|----------|-------|-----|------|
| 1 | 0.5% | 83K | 70hr | 0.5% | 5,187 | 1 | [ ] TODO |
| 2 | 1% | 166K | 140hr | 1.0% | 10,375 | 1 | [ ] TODO |
| 3 | 2% | 330K | 279hr | 2.0% | 20,625 | 1 | [ ] TODO |
| 4 | 100k | 95K | 84hr | 0.6% | 59,380 (10ep) | - | [x] CER 9.29% |
| 5 | 1M | 950K | 800hr | 5.7% | 59,375 | - | [x] CER 8.86% |
| 6 | 200k | 200K | 168hr | 1.2% | 12,500 | 1 | [ ] TODO |
| 7 | 500k | 500K | 420hr | 3.0% | 31,250 | 1 | [ ] TODO |
| 8 | 2M | 1.9M | 1680hr | 12% | 118,750 | 1 | [ ] TODO |
| 9 | 3M | 2.85M | 2520hr | 18% | 178,125 | 1 | [ ] TODO |
| 10 | 4M(full) | 4.09M | 4356hr | 31% | 255,756 | - | [x] 테스트 대기 |

---

## 우선순위

**1차 (0.5%, 1%, 2%):** 권장 범위 탐색
**2차 (200k, 500k, 2M, 3M):** 구간 세분화
**3차 (4M full 1ep):** 이미 nemo 있음, T527 테스트만

---

## 기존 결과 (참고)

| 실험 | Steps | 원본 대비 | CER |
|------|-------|----------|-----|
| 100k × 10ep | 59,380 | 8.7% (step) | 9.29% |
| 1M × 1ep | 59,375 | 5.7% (시간) | 8.86% |
| full × 10ep | 2,557,560 | 375% (step) | 14.81% (과적합) |

---

## 모델 파일 위치

```
완료:
  100k×10ep: qat_aihub_output/conformer_qat_100k_84hr_final.nemo
  1M×1ep:    qat_1m_1ep_output/conformer_qat_aihub1m_margin0.3_ep01.nemo
  full×1ep:  qat_aihub_full_output/conformer_qat_aihubfull_margin0.3_ep00.nemo
  full×10ep: qat_aihub_full_output/conformer_qat_full_ep09_final.nemo

TODO:
  0.5%×1ep:  qat_83k_1ep_output/conformer_qat_aihub83k_margin0.3_ep01.nemo
  1%×1ep:    qat_166k_1ep_output/conformer_qat_aihub166k_margin0.3_ep01.nemo
  2%×1ep:    qat_330k_1ep_output/conformer_qat_aihub330k_margin0.3_ep01.nemo
```
