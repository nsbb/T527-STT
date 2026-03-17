# T527 NPU 음성인식(STT) 모델

Allwinner T527 NPU (Vivante VIP9000NANOSI_PLUS) 용 음성인식 모델 모음.
모든 모델은 uint8/int8 양자화되어 `.nb` (Network Binary) 형태로 변환 완료.

## 모델 목록

| 모델 | 언어 | 아키텍처 | 양자화 | CER | 입력 길이 | 추론시간 | RTF | NB 크기 |
|------|------|---------|--------|-----|----------|----------|-----|---------|
| [KoCitrinet](ko_citrinet/) | 한국어 | 1D Conv + SE (CTC) | int8 | **44.44%** | 3초 | 120ms | 0.04 | 62MB |
| [Wav2Vec2](wav2vec2/) | 영어 | CNN + Transformer (CTC) | uint8 | **17.52%** | 5초 | 715ms | 0.14 | 87MB |
| [Zipformer](zipformer/) | 한국어 | Zipformer (RNN-T) | uint8 | 미측정 | 스트리밍 (~0.4초/청크) | 미측정 | 미측정 | 68MB |
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
| dynamic_fixed_point (int16) | O | O | HANG |
| bfloat16 / qbfloat16 | O | O | segfault / HANG |
| float16 (`--dtype float16`) | X | **O** | 동작 (셰이더 에뮬레이션, 25배 느림) |
| e4m3 / e5m2 (FP8) | X | **O** | 미테스트 |

> **결론**: T527 NPU에서 실용적인 양자화는 **uint8/int8만**. fp16은 NB 생성·실행 가능하나 하드웨어 가속 아님 (17.7초 vs uint8 0.7초).

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

- **uint8만 안정 동작** — int16/DFP는 NPU HANG 발생 (물리적 리셋 필요)
- **bf16 NB 생성 실패** — Acuity gen_nbg segfault
- **Transformer 모델 한국어 불가** — Wav2Vec2 한국어 50종+ 시도, 전부 실패 (uint8 양자화 열화 누적)
- **CNN 모델만 한국어 가능** — CitriNet (1D Conv) 계열만 의미 있는 출력

### 모델별 상세 결과

각 모델 폴더의 README.md 참조:
- [KoCitrinet](ko_citrinet/): CER 44.44%, 120ms — **한국어 유일한 선택**
- [Wav2Vec2](wav2vec2/): 영어 CER 17.52%, 한국어 불가능 — [상세 분석](wav2vec2/)
- [Zipformer](zipformer/): NB 변환 완료, 디바이스 테스트 대기
- [CitriNet EN](citrinet_en/): NB 변환 완료, 디바이스 테스트 대기
- [DeepSpeech2](deepspeech2/): NB 변환 완료, 디바이스 테스트 대기
- [테스트셋](testset/): 평가용 음성 데이터 — [자사 수집(ailab)](testset/ailab/), [영어(LibriSpeech)](testset/base_english/), [한국어(Zeroth-Korean)](testset/base_korean/)
