# 테스트셋

STT 모델 평가용 음성 테스트셋 및 ground truth CSV 모음.

## 테스트셋 목록

### 월패드 실환경 녹음

| CSV | 오디오 폴더 | 설명 |
|-----|-----------|------|
| [modelhouse_3m.csv](modelhouse_3m.csv) | `modelhouse_3m/` | 모델하우스 3m 거리 |
| [modelhouse_2m.csv](modelhouse_2m.csv) | `modelhouse_2m/` | 모델하우스 2m 거리 |
| [modelhouse_2m_noheater.csv](modelhouse_2m_noheater.csv) | `modelhouse_2m_noheater/` | 모델하우스 2m 거리 (난방 꺼짐) |
| [7F_KSK.csv](7F_KSK.csv) | `7F_KSK/` | 7층 KSK 화자 |
| [7F_HJY.csv](7F_HJY.csv) | `7F_HJY/` | 7층 HJY 화자 |

### 공개 데이터셋 (KsponSpeech 기반)

| CSV | 설명 |
|-----|------|
| [eval_clean_p.csv](eval_clean_p.csv) | KsponSpeech eval_clean (깨끗한 음성) |
| [eval_other_p.csv](eval_other_p.csv) | KsponSpeech eval_other (노이즈 포함) |
| [007.저음질_eval_p.csv](007.저음질_eval_p.csv) | 저음질 음성 평가셋 |
| [009.한국어_강의_eval_p.csv](009.한국어_강의_eval_p.csv) | 한국어 강의 음성 |
| [010.회의음성_eval_p.csv](010.회의음성_eval_p.csv) | 회의 음성 |
| [012.상담음성_eval_p.csv](012.상담음성_eval_p.csv) | 상담 음성 |

## worst30 — CER 최악 30개 샘플 추출

[worst30/](worst30/) 폴더에는 각 테스트셋별 CER이 가장 높은 30개 샘플이 추출되어 있다.

| CSV | 전체 CER | worst30 평균 CER |
|-----|---------|-----------------|
| [007_results_5.76_top30_avg_53.38.csv](worst30/007_results_5.76_top30_avg_53.38.csv) | 5.76% | 53.38% |
| [009_results_9.97_top30_avg_61.27.csv](worst30/009_results_9.97_top30_avg_61.27.csv) | 9.97% | 61.27% |
| [010_results_10.76_top30_avg_88.25.csv](worst30/010_results_10.76_top30_avg_88.25.csv) | 10.76% | 88.25% |
| [012_results_4.86_top30_avg_37.73.csv](worst30/012_results_4.86_top30_avg_37.73.csv) | 4.86% | 37.73% |
| [7F_HJY_results_9.27_top30_avg_26.95.csv](worst30/7F_HJY_results_9.27_top30_avg_26.95.csv) | 9.27% | 26.95% |
| [7F_KSK_results_2.66_top30_avg_9.59.csv](worst30/7F_KSK_results_2.66_top30_avg_9.59.csv) | 2.66% | 9.59% |
| [eval_clean_p_results_10.07_top30_avg_91.39.csv](worst30/eval_clean_p_results_10.07_top30_avg_91.39.csv) | 10.07% | 91.39% |
| [eval_other_p_results_9.54_top30_avg_69.09.csv](worst30/eval_other_p_results_9.54_top30_avg_69.09.csv) | 9.54% | 69.09% |
| [modelhouse_2m_noheater_results_3.59_top30_avg_6.10.csv](worst30/modelhouse_2m_noheater_results_3.59_top30_avg_6.10.csv) | 3.59% | 6.10% |
| [modelhouse_2m_results_8.51_top30_avg_14.48.csv](worst30/modelhouse_2m_results_8.51_top30_avg_14.48.csv) | 8.51% | 14.48% |
| [modelhouse_3m_results_15.72_top30_avg_26.55.csv](worst30/modelhouse_3m_results_15.72_top30_avg_26.55.csv) | 15.72% | 26.55% |

## 유틸리티 스크립트

| 파일 | 설명 |
|------|------|
| [download_all.sh](download_all.sh) | 전체 테스트셋 다운로드 |
| [csv_drop.py](csv_drop.py) | CSV 필터링/정리 |
| [worst30/remain_worst30.py](worst30/remain_worst30.py) | worst30 샘플 추출 |
