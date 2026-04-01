# Conformer STT 성능 개선 TODO

현재 최고 성능: **CER 7.24%** (QAT 100k + aihub calib 100개, 자체 368개 기준)
FP32 서버: 6.03% → 양자화 손실 +1.21%p

---

## 1. [HIGH] CSV 에러 유형 분석

18,368샘플 결과 CSV에서 insertion/deletion/substitution 비율 분석.

- [ ] 에러 유형별 분포 계산 (ins/del/sub 각 비율)
- [ ] 짧은 발화(ref < 5글자) CER 별도 분석 — 기존 분석에서 3.1배 악화 확인
- [ ] insertion 과다 패턴 파악 (끝에 "음", "어" 등 간투어)
- [ ] 데이터셋별 에러 유형 차이 비교
- [ ] worst case 분석 — CER > 50% 샘플 공통점 파악

**목표:** 에러 유형별 대응 방안 도출

---

## 2. [HIGH] 후처리 (Post-processing)

CSV 분석 결과 기반으로 CTC 디코드 후 규칙 적용.

- [ ] 발화 끝 단독 간투어 제거 ("음", "어", "네", "에", "아", "응")
- [ ] 연속 반복 음절 제거 ("면면" → "면")
- [ ] 공백 정규화 (다중 공백, 앞뒤 공백)
- [ ] 특수문자/숫자 후처리 (필요 시)
- [ ] 동일 18k 테스트셋으로 개선 효과 측정

**기대:** CER 1~2%p 개선

---

## 3. [HIGH] Calibration 데이터 분포 최적화

현재 랜덤 100개 → 분포 기반 선택으로 calib 품질 개선.

- [ ] 전체 학습 데이터 mel feature 추출
- [ ] K-Medoids (K=100) 클러스터링으로 대표 샘플 선택
- [ ] Herding (coreset) 기법도 비교 실험
- [ ] 선택된 calib으로 재양자화 → 18k 테스트
- [ ] 랜덤 100개 vs K-Medoids 100개 CER 비교

**기대:** 같은 100개에서 CER 0.2~0.5%p 추가 개선

---

## 4. [HIGH] QAT Loss 함수 실험

현재 MarginLoss (margin=0.3) + KD → 다른 loss 조합 시도.

- [ ] margin 값 sweep (0.1, 0.2, 0.3, 0.5, 1.0)
- [ ] KD loss 비중 조정 (alpha sweep)
- [ ] CTC loss만 (KD 없이) vs KD+CTC 비교
- [ ] Label Smoothing CTC 시도
- [ ] Focal CTC Loss 시도 (어려운 샘플에 가중치)
- [ ] 최적 조합으로 재학습 → 18k 테스트

**기대:** CER 0.3~1.0%p 개선

---

## 5. [MEDIUM] 전처리 (Pre-processing)

녹음 음성 품질 개선 후 mel 추출.

- [ ] 입력 오디오 정규화 (peak normalization)
- [ ] 간단한 noise gate (에너지 기반 무음 제거)
- [ ] VAD 결과 활용 — 음성 구간만 STT에 전달 (이미 구현됨)
- [ ] 실환경(월패드) 녹음으로 효과 검증

---

## 6. [MEDIUM] Beam Search CTC 디코딩

현재 greedy argmax → beam search로 교체.

- [ ] Python으로 CTC beam search 구현 (beam width 10~20)
- [ ] 기존 NPU output .dat 파일 재활용 (재추론 불필요)
- [ ] greedy vs beam search CER 비교
- [ ] 최적 beam width 결정
- [ ] 효과 확인 후 C/JNI로 포팅

**기대:** CER 0.3~0.8%p 개선

---

## 7. [MEDIUM] Language Model (n-gram) 결합

CTC + shallow LM fusion. SungBeom 모델 README에서 KenLM 적용 시 WER 13.45% → 5.27% 언급.

- [ ] KenLM으로 한국어 n-gram LM 학습 (AIHub 텍스트)
- [ ] Beam search + LM weight 조합 실험
- [ ] LM weight sweep (0.1 ~ 1.0)
- [ ] 18k 테스트셋으로 효과 측정
- [ ] Android에서 KenLM 추론 가능 여부 확인

**기대:** CER 1.0~2.0%p 개선 (가장 큰 단일 개선 가능성)

---

## 8. [MEDIUM] 5초 슬라이딩 윈도우

현재 3초 (301 frames) → 5초 (501 frames)로 확장.

- [ ] ONNX static shape [1, 80, 501]로 변경
- [ ] fix_onnx_for_acuity.py에 --frames 501 적용
- [ ] Acuity import → quantize → NB export
- [ ] NB 크기 확인 (예상 ~130MB, T527 한계 ~120MB 초과 가능)
- [ ] 크기 OK면: 슬라이딩 윈도우 stride 조정, 18k 테스트
- [ ] 경계 에러 감소 효과 확인

**기대:** 경계 에러 감소, chunk 수 감소, CER 0.3~0.5%p 개선
**리스크:** NB 크기 초과 시 불가

---

## 9. [LOW] 슬라이딩 윈도우 오버랩 튜닝

현재 stride 250/301 (17% overlap) → overlap 비율 조정.

- [ ] stride 200/301 (33% overlap) 시도
- [ ] 겹치는 구간 logit 평균 또는 가중 평균
- [ ] CER 비교 (오버헤드 증가 vs 정확도 개선 트레이드오프)

**기대:** CER 0.1~0.3%p 개선

---

## 10. [LOW] 도메인 특화 Fine-tune

월패드 명령어에 특화된 추가 학습.

- [ ] 월패드 도메인 데이터 수집 (명령어 + 자연어)
- [ ] LoRA 또는 full fine-tune
- [ ] 범용 성능 저하 없는지 확인 (AIHub 테스트)

**기대:** 월패드 CER 2~5%p 개선
**리스크:** 다른 도메인 성능 저하 가능

---

## 우선순위 요약

| 순위 | 항목 | 기대 효과 | 난이도 |
|------|------|----------|--------|
| 1 | CSV 에러 분석 | 방향 설정 | 쉬움 |
| 2 | 후처리 (간투어 제거) | -1~2%p | 쉬움 |
| 3 | Calib 분포 최적화 | -0.2~0.5%p | 쉬움 |
| 4 | QAT Loss 실험 | -0.3~1.0%p | 중간 |
| 5 | 전처리 | -0.2~0.5%p | 쉬움 |
| 6 | Beam Search | -0.3~0.8%p | 중간 |
| 7 | LM (KenLM) | -1.0~2.0%p | 어려움 |
| 8 | 5초 윈도우 | -0.3~0.5%p | 중간 |
| 9 | 오버랩 튜닝 | -0.1~0.3%p | 쉬움 |
| 10 | 도메인 fine-tune | -2~5%p | 어려움 |

### 누적 개선 시나리오

```
현재:                          CER 7.24%
+ 후처리 + 에러 분석 기반:      → 6.0%  (-1.2%p)
+ Calib 분포 + QAT loss:       → 5.3%  (-0.7%p)
+ Beam Search + LM:            → 3.5%  (-1.8%p)
+ 5초 윈도우 + 오버랩:          → 3.0%  (-0.5%p)
```

**목표: CER 7.24% → 3~5%**
