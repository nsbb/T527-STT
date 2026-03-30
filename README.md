# T527 NPU 음성인식(STT) 모델

Allwinner T527 NPU (Vivante VIP9000NANOSI_PLUS) 용 음성인식 모델 모음.
모든 모델은 uint8/int8 양자화되어 `.nb` (Network Binary) 형태로 변환 완료.

## 모델 목록

| 모델 | 언어 | 아키텍처 | 양자화 | CER | 입력 길이 | 추론시간 | RTF | NB 크기 |
|------|------|---------|--------|-----|----------|----------|-----|---------|
| [Conformer CTC](conformer_ctc/) | 한국어 | CNN + Attention (CTC) | **uint8 QAT** | **5.3%** | 3초 | 233ms | 0.11 | 102MB |
| [KoCitrinet](ko_citrinet/) | 한국어 | 1D Conv + SE (CTC) | int8 | **35%** | 3초 | 120ms | 0.04 | 62MB |
| [Wav2Vec2](wav2vec2/) | 영어 | CNN + Transformer (CTC) | uint8 | **17.52%** | 5초 | 715ms | 0.14 | 87MB |
| [Zipformer](zipformer/) | 한국어 | Zipformer (RNN-T) | uint8/int16/PCQ | **100%** (실패) | 스트리밍 (~0.4초/청크) | ~50ms/chunk | — | 63~118MB |
| [CitriNet EN](citrinet_en/) | 영어 | 1D Conv + SE (CTC) | uint8 | 미측정 | 3초 | 미측정 | 미측정 | 7MB |
| [DeepSpeech2](deepspeech2/) | 영어 | RNN (CTC) | uint8 | 미측정 | ~4.7초 (756f) | 미측정 | 미측정 | 56MB |

> **RTF** (Real-Time Factor) = 추론시간 / 입력 음성 길이. RTF < 1이면 실시간보다 빠름.

## 하드웨어

- **SoC**: Allwinner T527 (ARM Cortex-A55 옥타코어)
- **NPU**: Vivante VIP9000NANOSI_PLUS (PID 0x10000016)
- **드라이버**: VIPLite v0x00010d00
- **NPU 클럭**: 696MHz, DRAM 1.2GHz

## 도구 체인

- **Acuity Toolkit**: v6.12.0 (주력) / v6.21.16 (대안)
- **VivanteIDE**: v5.7.2 (6.12용) / v5.8.2 (6.21용)
- **양자화**: uint8 asymmetric_affine, moving_average
- **Export**: `pegasus export ovxlib --pack-nbg-unify`

---

## Conformer CTC QAT — 한국어 CER 5.3% (SungBeom, 122.5M params)

PTQ(Post-Training Quantization)로는 CER 44%에 불과했던 Conformer CTC 모델을 QAT(Quantization-Aware Training)로 재학습하여 **CER 5.3%** 달성. 60+ 양자화 시도(10 아키텍처 × 21 방법) 중 유일하게 실용 수준에 도달한 결과.

### 핵심 수치

| 항목 | 값 |
|------|-----|
| 모델 | Conformer CTC (NeMo, 122.5M params) |
| 아키텍처 | CNN + Attention hybrid (CTC) |
| 양자화 | uint8 QAT (FakeQuantize + STE + MarginLoss) |
| CER | 44% (PTQ) → **5.3%** (QAT) — 8배 개선 |
| 입력 | 3초 청크 (mel spectrogram [1,80,301]) |
| 추론시간 | 233ms/chunk |
| RTF | 0.11 (9배 실시간) |
| NB 크기 | 102MB |

### QAT 학습 환경

- **데이터**: AIHub 84시간 (95K 샘플)
- **GPU**: 4-GPU DDP (RTX 6000 Ada)
- **QAT 기법**: FakeQuantize 3지점 (encoder in/out, decoder out), MarginLoss target=0.3
- **핵심 발견**: val_loss 최소점(epoch 4) = NPU CER 100%, 최종 epoch 10 = NPU CER 5.3%. val_loss로 조기 종료하면 실패.

### 양자화 실패 원인 분석 (logit margin)

한국어 모델이 uint8 양자화에 특히 취약한 근본 원인:
- **한국어 logit margin**: 0.005 (1위-2위 logit 차이)
- **영어 logit margin**: 0.34
- **uint8 양자화 step**: 0.05
- 한국어 margin(0.005) < uint8 step(0.05) → 양자화 시 1위-2위 역전 빈발
- CNN+Attention 하이브리드(Conformer)만 QAT로 margin 확보에 성공

### 검증 결과 (18,368 샘플 × 11 데이터셋)

| 환경 | CER |
|------|-----|
| 잡음 없음 (No noise) | 7.49% |
| 상담 (Consultation) | 10.03% |
| 회의 (Meeting) | 15.29% |
| 강의 (Lecture) | 15.66% |
| 저품질 (Low quality) | 19.01% |
| 원거리 3m (Far-field) | 21.73% |

**에러 유형**: Substitution 63.7%, Insertion 23.6%, Deletion 12.7%

### ONNX → NB 변환 파이프라인

```
NeMo export (4462 nodes) → onnxsim (1905 nodes) → Pad fix (18 ops) → Acuity → NB (102MB)
```

### Android 앱

ConformerTestActivity (JNI/C) — 디바이스 실시간 추론 검증 완료.

---

## Acuity Toolkit 버전 비교 (6.12 vs 6.21)

### 설치 및 실행

| 항목 | Acuity 6.12.0 | Acuity 6.21.16 |
|------|---------------|----------------|
| 설치 형태 | 바이너리 (standalone) | pip wheel (Python) |
| 실행 | `./pegasus` | `python3 .../pegasus.py` |
| Docker | `t527-npu:v1.2` | `ubuntu-npu:v1.8.11` |
| VivanteIDE | 5.7.2 (OVXLIB 1.1.20) | 5.8.2 (OVXLIB 1.2.18) |

### 양자화 지원 비교

| 양자화기 | 6.12 | 6.21 | T527 NPU 실행 |
|---------|------|------|---------------|
| asymmetric_affine (uint8) | O | O | **동작** |
| perchannel_symmetric (PCQ int8) | O | O | **동작** |
| dynamic_fixed_point (int16) | O | O | **동작** (NB ≤118MB), status=-1 (NB ≥153MB) |
| bfloat16 / qbfloat16 | O | O | segfault / HANG |
| float16 (`--dtype float16`) | X | **O** | 동작 (셰이더 에뮬레이션, 25배 느림) |
| e4m3 / e5m2 (FP8) | X | **O** | 미테스트 |

> **결론**: T527 NPU에서 실용적인 양자화는 **uint8/int8**. int16 DFP도 NB 크기 ≤118MB이면 동작 확인 (Zipformer encoder), 다만 153MB 이상은 status=-1. fp16은 NB 생성·실행 가능하나 하드웨어 가속 아님 (17.7초 vs uint8 0.7초).

### 주요 차이점

| 항목 | 6.12 | 6.21 |
|------|------|------|
| export 시 `--with-input-meta` | 불필요 | **필수** |
| inputmeta lid 검증 | 관대 | **엄격** (정확히 일치 필요) |
| Graph optimization | 보수적 | 공격적 (NB 작아지나 정확도 하락 가능) |

### 실측 결과 (Wav2Vec2 영어, 50샘플)

| 양자화 | Acuity | CER | WER | NB크기 | 추론시간 |
|--------|--------|-----|-----|--------|---------|
| **uint8** | **6.12** | **17.52%** | **27.38%** | **87MB** | **~720ms** |
| PCQ int8 | 6.21 | 19.24% | 34.39% | 99MB | ~826ms |
| uint8 | 6.21 | 23.41% | 40.57% | 76MB | ~720ms |

### 권장 사항

- **신규 모델**: 6.12 uint8로 먼저 시도, 정확도 부족시 6.21 PCQ 비교
- **Docker export**: 6.12는 `EXTRALFLAGS` rpath 설정 필수, 6.21은 lib 심링크 필수
- **fp16**: 정확도 검증용으로만 사용 (17.7초/추론 → 실서비스 불가)

## NPU API

```c
#include <awnn_lib.h>

awnn_init();
ctx = awnn_create("model.nb");

unsigned char *inputs[1] = {quantized_input};
awnn_set_input_buffers(ctx, inputs);
awnn_run(ctx);
float **outputs = awnn_get_output_buffers(ctx);

awnn_destroy(ctx);
awnn_uninit();
```

## 디바이스 테스트 (vpm_run)

```bash
adb push network_binary.nb /data/local/tmp/test/
adb push input_0.dat /data/local/tmp/test/
# sample.txt: [network]\n path \n[input]\n path \n[output]\n path
adb shell "cd /data/local/tmp/test && LD_LIBRARY_PATH=/vendor/lib64 ./vpm_run_aarch64 -s sample.txt -b 0"
adb pull /data/local/tmp/test/output_0.dat .
```

## 핵심 발견 사항

### T527 NPU 양자화 제약

- **uint8 QAT로 한국어 CER 5.3% 달성** — Conformer CTC (122.5M params) + QAT (FakeQuantize + STE + MarginLoss)로 PTQ CER 44% → QAT CER 5.3%, 8배 개선
- **CNN+Attention 하이브리드만 uint8 생존** — 60+ 양자화 시도 (10 아키텍처 × 21 방법) 중 Conformer만 유의미한 결과. 한국어 logit margin 0.005 vs 영어 0.34 → uint8 step 0.05에서 한국어가 특히 취약
- **uint8 안정 동작** — 모든 모델에서 NB 생성·실행 가능
- **int16 DFP 조건부 동작** — NB 크기 ≤118MB이면 정상 실행 (Zipformer int16 118MB 동작 확인). ≥153MB면 status=-1 (Wav2Vec2 int16 153MB 실패). 이전 "T527 NPU는 int16 미지원" 결론은 **오류** — NB 크기 제한이 원인이었음.
- **bf16 NB 생성 실패** — Acuity gen_nbg segfault 또는 export error 64768
- **대형 Transformer 양자화 한계** — 5868노드(Zipformer)는 uint8/int16 모두 CER 100% (에러 누적). ~2000노드(Wav2Vec2 영어)는 uint8 CER 17.5% 성공
- **val_loss ≠ NPU CER** — QAT 학습 중 val_loss 최소 지점(epoch 4)이 NPU CER 100%, 최종 epoch 10이 CER 5.3%. 반드시 NPU 실측으로 검증 필요

### 모델별 상세 결과

각 모델 폴더의 README.md 참조:
- [Conformer CTC](conformer_ctc/): **CER 5.3%**, 233ms — **한국어 최고 성능 (QAT)** — [상세 결과](conformer_ctc/)
- [KoCitrinet](ko_citrinet/): CER 35%, 120ms — 한국어 PTQ 기준 최선
- [Wav2Vec2](wav2vec2/): 영어 CER 17.52%, 한국어 불가능 — [상세 분석](wav2vec2/)
- [Zipformer](zipformer/): uint8/int16/PCQ 전 방식 CER 100% 실패 — [상세 결과](zipformer/)
- [CitriNet EN](citrinet_en/): NB 변환 완료, 디바이스 테스트 대기
- [DeepSpeech2](deepspeech2/): NB 변환 완료, 디바이스 테스트 대기
- [테스트셋](testset/): 평가용 음성 데이터 — [자사 수집(ailab)](testset/ailab/), [영어(LibriSpeech)](testset/base_english/), [한국어(Zeroth-Korean)](testset/base_korean/)
