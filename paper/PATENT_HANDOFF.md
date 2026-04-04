# 특허 작성 핸드오프 문서

## 개요

이 문서는 T527 NPU 한국어 음성인식 프로젝트의 특허 출원을 위한 핸드오프 자료.
다른 세션에서 이 문서를 읽고 특허 명세서 작성을 시작할 것.

---

## 특허 vs 논문 차이

- **논문**: "왜 되는가" + 분석 + 새 방법론 필요. "포팅했다"만으로는 기여 부족.
- **특허**: "어떻게 하는가" 구체적 절차가 핵심. 포팅 방법 자체가 특허 대상 가능.
- 실제로 "신경망 양자화 방법" 특허 다수 등록됨 (US20220044114, KR20220109235 등)

---

## 특허 가능한 발명 목록

### 1. W8A8 NPU에서의 한국어 ASR 배포 방법 (시스템 특허)

**청구 범위:**
- NeMo .nemo → ONNX export (dither=0, pad_to=0)
- ONNX graph surgery: 동적 → 정적 형상 [1,80,301], onnxsim 4462→1905 노드
- Pad operator constant_value 패치
- Acuity import + uint8 KL-divergence calibration (100 샘플)
- NB export (VIP9000NANOSI_PLUS_PID0X10000016)
- Android app에서 awnn API로 NPU 추론

**기술적 효과:** 122.5M 파라미터 Conformer를 2 TOPS NPU에서 233ms/chunk, RTF 0.078로 동작

### 2. MarginLoss를 이용한 양자화 인식 학습 방법

**청구 범위:**
- CTC 모델의 출력 로짓에서 top-1과 top-2의 차이(마진)를 계산
- 마진이 목표값(m) 미만이면 ReLU 페널티 부과
- L = L_CTC + λ × L_margin (λ=0.1, m=0.3)
- 2,049 클래스 CTC에서 uint8 step size 0.19에 대해 margin 0.3 (1.5배) 확보
- FakeQuantize를 인코더 입력, 인코더 출력, 디코더 출력 3곳에 삽입

**기술적 효과:** PTQ CER 15.59% → QAT CER 9.30%, 78% 열화 회복

### 3. 고정 윈도우 제로패딩에 의한 CTC 환각 억제 방법

**청구 범위:**
- 정적 입력 형상 NPU에서 짧은 음성(< 3초)을 고정 길이(301 프레임)로 제로패딩
- 패딩 영역에 CTC blank 토큰이 할당되어 노이즈 프레임의 환각 출력 억제
- 슬라이딩 윈도우: 301 프레임 윈도우, 250 프레임 스트라이드, 51 프레임 오버랩
- 청크별 로짓 연결: 마지막 청크 제외 63 프레임씩, 마지막은 76 프레임 전체

**기술적 효과:** 짧은 저음질 발화에서 FP32 CER 17.85% → INT8 CER 8.65% (9.20%p 개선)

### 4. VT+VAD+STT 온디바이스 음성인식 파이프라인

**청구 범위:**
- Voice Trigger(VT) → 웨이크워드 감지
- Voice Activity Detection(VAD) → 음성 구간 검출 (시작/끝점)
- VAD 출력에 따라 음성 구간만 STT NPU에 전달
- 한국어/영어 모델 동시 로드 (ctx_ko, ctx_en), LID에 따라 선택 실행
- 고정 형상 NPU 제약 하에서 가변 길이 음성 처리

**기술적 효과:** 묵음 구간 NPU 처리 불필요 → 전력 절약, 가변길이 음성 처리

### 5. QAT-PTQ 불일치 해소를 위한 양자화 파라미터 주입 방법 (LSQ 성공 시)

**청구 범위:**
- LSQ(Learned Step Quantization)로 최적 scale/zero-point 학습
- 학습된 양자화 파라미터를 Acuity .quantize 파일에 직접 주입
- 닫힌 툴체인(Acuity)의 PTQ를 우회하여 QAT 학습 결과를 그대로 배포

**기술적 효과:** QAT-PTQ 불일치 31.5% 해소 (아직 실험 미완 — 결과에 따라)

---

## 핵심 기술 상세

### 하드웨어
- **SoC:** Allwinner T527, 옥타코어 ARM Cortex-A55
- **NPU:** VeriSilicon VIP9000NANOSI_PLUS, 696 MHz, 2 TOPS (INT8)
- **제약:** W8A8 only (uint8 asymmetric affine), 정적 입력 형상, NB ≤ 120MB

### 모델
- **SungBeom Conformer CTC Medium:** 122.5M params, 18 layers
- d_model=512, 8-head attention, depthwise conv kernel 31
- 2,049 BPE vocab (2,048 한국어 + 1 blank)
- 입력: [1, 80, 301] (80 mel bins, 301 frames = 3.01초)
- 출력: [1, 76, 2049]
- NB 크기: 102 MB

### 변환 파이프라인
```
.nemo (NeMo Docker 23.06)
  → ONNX export (dither=0, pad_to=0)
  → fix_onnx_for_acuity.py (정적 형상, onnxsim, Pad 패치)
  → Acuity 6.12 import (LD_LIBRARY_PATH 필수!)
  → Acuity quantize (uint8, KL-divergence, 100 calib 샘플)
  → Acuity export ovxlib (VIP9000NANOSI_PLUS_PID0X10000016)
  → network_binary.nb (102 MB)
```

### QAT 학습
- **Loss:** CTC + 0.1 × MarginLoss(m=0.3)
- **FakeQuantize 위치:** encoder input, encoder output, decoder output
- **데이터:** AIHub 100k 랜덤 샘플 (84.63시간)
- **하이퍼파라미터:** AdamW lr=1e-5, cosine schedule, batch 16, 10 epochs
- **GPU:** NVIDIA RTX 6000 Ada (48GB), ~2시간
- **결과:** PTQ 15.59% → QAT 9.30% (78% 회복)
- **최신:** 1M × 1ep = 8.86% (실험 진행 중)

### 아키텍처 비교 (W8A8 양자화)
| 모델 | FP32 CER | uint8 CER | 상태 |
|------|---------|----------|------|
| Conformer CTC | ~6-10% | 10.59% | 성공 |
| KoCitrinet | 8.44% | 44.44% | 열화 |
| Wav2Vec2 (EN) | 9.74% | 17.52% | 열화 |
| Wav2Vec2 (KO) | 9-18% | 92.83% | 실패 |
| Zipformer | 16.2% | 100% | 실패 |
| HuBERT (KO) | — | 100% | 실패 |

### CTC 환각 억제 (007-저음질, ≤3초)
| 지표 | FP32 | INT8 QAT |
|------|------|---------|
| 평균 가설/정답 비율 | 2.30× | 0.99× |
| 삽입 우세 비율 | 73.6% | 5.3% |
| CER | 17.85% | 8.65% |

---

## 관련 파일 위치

| 파일 | 위치 |
|------|------|
| QAT 학습 스크립트 | `conformer/scripts/train_qat.py` |
| QAT config | `conformer/experiments/qat_aihub100k_margin0.3_ep10.yaml` |
| ONNX 변환 스크립트 | `conformer/scripts/fix_onnx_for_acuity.py` |
| 18,368 샘플 결과 | `conformer/results/qat_100k_calib_aihub100_18k/` |
| CTC 환각 분석 | `conformer/results/qat_100k_calib_aihub100_18k/007_QAT_BEATS_FP32_ANALYSIS.md` |
| 논문 (영문) | `paper/paper.md`, `paper/paper.tex` |
| 논문 (한글) | `paper/paper_ko.md`, `paper/paper_ko.tex` |
| SLT 제출 계획 | `paper/SLT2026_PLAN.md` |
| QAT 실험 TODO | `conformer/experiments/QAT_TODO.md` |

---

## 선행 특허 (검색 결과)

| 특허번호 | 제목 | 관련성 |
|---------|------|--------|
| US20220044114 | Hybrid Quantization of Neural Networks for Edge Computing | INT8 DLA 배포 |
| KR20220109235 | Bitwidth Scheduling을 이용한 신경망 양자화 방법 | 한국 특허, 엣지 디바이스 |
| US20250348729 | Profile-Guided Quantization of Neural Networks | 레이어별 최적 양자화 |
| US20220092384 | 신경망 파라미터 양자화 방법 및 장치 | 디바이스별 양자화 |

---

## 저자/발명자

- **김희수 (Huisoo Kim)** — nsbb@hdc-labs.com
- **김희원 (Heewon Kim)** — ive2go@hdc-labs.com
- **이건희 (Gunhee Lee)** — Gunhee_Lee@hdc-labs.com
- **소속:** HDC LABS, Seoul, South Korea

---

## 주의사항

- 논문과 특허 동시 진행 가능하지만, **특허 출원을 논문 공개 전에 해야** 신규성 상실 방지
- 한국은 공개 후 12개월 내 출원하면 신규성 예외 주장 가능 (공지예외 적용)
- SLT 2026 제출 (6/17) 전에 특허 출원하는 것이 안전
- 실험 진행 중인 항목 (LSQ, 데이터 sweep 등)은 결과 확정 후 청구 범위 업데이트
