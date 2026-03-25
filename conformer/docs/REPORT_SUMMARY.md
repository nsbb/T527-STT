# T527 NPU 한국어 음성인식 — Conformer CTC 모델 배포 보고

## 요약

한국어 Conformer CTC 모델을 T527 NPU에 uint8 양자화하여 배포에 성공. 기존 KoCitrinet 대비 **CER 4배 개선** (44% → 10%).

| 항목 | 기존 (KoCitrinet) | **신규 (Conformer)** |
|------|-------------------|---------------------|
| 모델 | KoCitrinet 256 | **SungBeom Conformer CTC Medium** |
| 아키텍처 | CNN (1D Conv + SE) | **CNN + Self-Attention 하이브리드** |
| 파라미터 | ~10M | **122.5M** |
| NB 크기 | 62MB | **102MB** |
| 추론 시간 (3초) | 120ms | **233ms** |
| **CER (100샘플)** | **44.44%** | **10.02%** |
| RTF | 0.040 | **0.112** |
| Android 앱 | 동작 확인 | **동작 확인** |

---

## 1. 성과

### CER 10.02% 달성

Zeroth-Korean 테스트셋 100개 샘플 전체 평가 결과:

| 지표 | 값 |
|------|-----|
| 평균 CER | **10.02%** |
| 중앙값 CER | 8.70% |
| CER 0% (완벽 인식) | 4개 |
| CER < 5% | 24개 |
| CER < 10% | 64개 |
| CER < 20% | 97개 |
| 최고 | #11 CER 0.0% |
| 최저 | #1 CER 82.0% |

### 인식 예시

| GT (정답) | NPU 출력 | CER |
|-----------|---------|-----|
| 평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다 | 평소 오전 아홉 시에서 오후 일곱 시까지 일하면 하루 이만 원 정도를 번다 | 3.4% |
| 하지만 싫증 나면 버리면 그만이라는 식의 사고방식은 단지 물건에만 영향을 미치는 데서 그치지 않습니다 | 하지만싫증 나면 버리 면 그만이라는 식의 사고방식은 단지 물건에만 영향을 미치는 데서 그치지 않습니다 | 0.0% |
| 그런데 어찌하여 그런 뜨거운 가슴을 지닌 사람이 개성공단 폐쇄나 사드 문제에 관해 그렇게 | 그런데 어찌하여 그런 뜨거운 가슴을 지닌 사람이 개성공단 폐 체나 사드 문제에 관해 그렇게 | 2.4% |

### 실시간 처리 성능

| 음성 길이 | chunks | 추론 시간 | RTF |
|----------|--------|---------|-----|
| 3초 | 1 | 233ms | 0.078 |
| 10초 | 4 | 932ms | 0.093 |
| 20초 | 8 | 1.9초 | 0.095 |

RTF < 0.13 → **실시간의 8~13배 빠른 처리.**

---

## 2. 기술 사항

### 모델

- 출처: [HuggingFace SungBeom/stt_kr_conformer_ctc_medium](https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium)
- 프레임워크: NVIDIA NeMo
- 학습 데이터: AI Hub 한국어 음성
- Vocab: 2049 (한국어 BPE 2048 + blank)

### 양자화

| 방식 | NB | 추론 | CER | 비고 |
|------|-----|------|-----|------|
| **uint8 AA KL (채택)** | **102MB** | **233ms** | **10.59%** | 속도/크기/정확도 최적 |
| uint8 AA MA | 102MB | 233ms | 10.79% | KL과 거의 동일 |
| int16 DFP KL | 200MB | 564ms | 10.18% | 0.4%p 개선이나 2.4배 느림 |

### 변환 파이프라인

```
HuggingFace .nemo → NeMo ONNX export → static shape 고정 → Pad op 수정
→ Acuity import → uint8 KL quantize → NB export (102MB)
→ T527 NPU 동작 확인 (vpm_run + Android 앱)
```

### Android 앱

기존 awaiasr_2 앱에 ConformerTestActivity 추가. 슬라이딩 윈도우 방식으로 전체 음성 처리.

- 입력: NeMo preprocessor로 생성한 mel spectrogram (uint8)
- NPU: awnn API (libVIPlite.so)
- 출력: BPE CTC greedy decode
- **vpm_run과 동일한 결과 확인**

---

## 3. 해결한 기술 과제

| # | 과제 | 해결 방법 |
|---|------|----------|
| 1 | Conformer ONNX export 시 FP32에서도 garbage | vocab.txt ≠ tokenizer ID 매핑 발견 → `tok.ids_to_tokens()` 사용 |
| 2 | librosa mel ≠ NeMo mel (range 차이) | NeMo preprocessor로 mel 생성 (dither=0, pad_to=0) |
| 3 | Pad op empty constant_value → Acuity 실패 | 명시적 0.0으로 수정 (18개) |
| 4 | Dynamic shape ONNX → Acuity 비호환 | static [1,80,301] 고정 + length 상수화 |
| 5 | Where op 54개 → graph surgery 시 모델 파괴 | 건드리지 않음 (Acuity가 자체 처리) |
| 6 | SpeechBrain Conformer 42.9M → NB export 실패 | NeMo 모델로 전환 (122.5M이지만 NB 성공) |
| 7 | awnn output layout 차이 | time-major [SEQ×VOCAB] 확인 (vocab-major 아님) |

---

## 4. 모델 비교 (T527 NPU 한국어 STT 전체)

| 모델 | 아키텍처 | NB | 추론 | CER | 상태 |
|------|---------|-----|------|-----|------|
| **SungBeom Conformer** | **CNN+Attention** | **102MB** | **233ms/chunk** | **10.02%** | **최고 정확도** |
| cwwojin Conformer | CNN+Attention | 29MB | 111ms/chunk | 55.10% | 경량 |
| KoCitrinet 300f | CNN | 62MB | 120ms | 44.44% | 기존 운용 |
| Wav2Vec2 KO (NAS 80k) | Transformer | 77MB | 424ms | ~100% | 실패 |
| Wav2Vec2 KO (fine-tune) | Transformer | 72MB | ~400ms | ~100% | 실패 |
| HuBERT KO | Transformer | 76MB | 423ms | 실패 | 동일 토큰 반복 |

**결론: Conformer (CNN+Attention 하이브리드)가 T527 uint8 양자화에서 유일하게 높은 정확도 달성.**
순수 Transformer (Wav2Vec2, HuBERT)는 uint8에서 실패, 순수 CNN (KoCitrinet)은 양자화에 강하나 정확도 한계.

---

## 5. 향후 계획

| 우선순위 | 작업 | 기대 효과 |
|---------|------|----------|
| 1 | NeMo mel 전처리 JNI 구현 | WAV 직접 입력 가능 (현재는 미리 변환 필요) |
| 2 | 실시간 마이크 입력 | 실제 사용 시나리오 |
| 3 | 입력 길이 확장 (5초/10초 NB) | 슬라이딩 윈도우 오버헤드 감소 |
| 4 | QAT (Quantization-Aware Training) | CER 추가 개선 가능성 |
| 5 | Language Model 결합 | 후처리로 CER 추가 개선 |

---

## 부록: 결과 파일

| 파일 | 내용 |
|------|------|
| `results/kr_sungbeom_uint8_kl_100.csv` | uint8 KL 100샘플 (CER순, 추론시간 포함) |
| `results/kr_sungbeom_uint8_ma_100.csv` | uint8 MA 100샘플 |
| `results/kr_sungbeom_int16_dfp_100.csv` | int16 DFP 100샘플 |
| `docs/SUNGBEOM_REPORT.md` | 상세 실험 보고서 |
| `docs/DEPLOYMENT_PARAMS.md` | 배포 파라미터 전체 |
| `docs/TROUBLESHOOTING_JOURNEY.md` | 시행착오 기록 |
| `docs/APP_TEST_RESULTS.md` | Android 앱 테스트 결과 |
