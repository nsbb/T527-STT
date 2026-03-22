# T527 NPU STT 모델 인벤토리

모든 시도한 모델, 양자화 방법, NB 파일, 성능 결과를 정리한 문서.

---

## 최종 성능 요약

| # | 모델 | 언어 | ONNX 크기 | 양자화 | NB 크기 | T527 NPU | CER | 추론시간 | 상태 |
|---|------|------|----------|--------|---------|----------|-----|----------|------|
| 1 | KoCitrinet 300f | 한국어 | 543MB | int8 asymmetric | 62MB | 동작 | **44.44%** | 120ms | **운용중** |
| 2 | KoCitrinet 500f | 한국어 | 543MB | int8 asymmetric | 62MB | 동작 | 미측정 | ~120ms | 탑재완료 |
| 3 | KoCitrinet 300f | 한국어 | 543MB | int16 | 미시도 | — | — | — | **미시도** |
| 4 | Wav2Vec2 base-960h 5s | 영어 | 361MB | uint8 asymmetric | 88MB | 동작 | **~17.52%** | 715ms | 검증완료 |
| 5 | Wav2Vec2 base-960h 5s | 영어 | 361MB | int16 DFP | 152MB | status=-1 | — | — | **실패** (NB 크기 초과) |
| 6 | Wav2Vec2 base-korean 3s | 한국어 | 361MB | ONNX float (CPU) | — | CPU only | **33.74%** | 75ms(서버) | 서버용만 가능 |
| 7 | Wav2Vec2 base-korean 3s | 한국어 | 361MB | uint8 (300 calib) | 72MB | 동작 | 100.86% | 415ms | **실패** |
| 8 | Wav2Vec2 base-korean 3s | 한국어 | 361MB | int16 DFP | 153MB | status=-1 | — | — | **실패** (NB 크기 초과) |
| 9 | Wav2Vec2 base-korean 3s | 한국어 | 361MB | fp16 | 182MB | CPU fallback | — | 17,740ms | **실패** (HW 미가속) |
| 10 | Wav2Vec2 XLS-R-300M 3s | 한국어 | 1.2GB | uint8 | 249MB | 동작 | ALL PAD | 1098ms | **실패** |
| 11 | CitriNet EN 3s | 영어 | 40MB | uint8 | 7MB | 동작 | 미측정 | 미측정 | NB 변환완료 |
| 12 | CitriNet EN 3s | 영어 | 40MB | fp16 | 21MB | 미측정 | 미측정 | 미측정 | NB 변환완료 |
| 13 | DeepSpeech2 | 영어 | — | uint8 | 56MB | 동작 | 미측정 | 미측정 | NB 변환완료 |
| 14 | Zipformer Encoder | 한국어 | 280MB | uint8 | 63MB | **동작** | 100% | ~50ms/chunk | **실패** (에러 누적) |
| 15 | Zipformer Encoder | 한국어 | 280MB | int16 DFP | 118MB | **동작** | 100% | ~50ms/chunk | **실패** (에러 누적) |
| 16 | Zipformer Encoder | 한국어 | 280MB | PCQ int8 | 71MB | **동작** | 100% | ~50ms/chunk | **실패** (에러 누적) |
| 17 | Zipformer Decoder | 한국어 | 11MB | uint8 | 2.8MB | 동작 | — | — | Encoder 실패로 무의미 |
| 18 | Zipformer Joiner | 한국어 | 9.8MB | uint8 | 1.9MB | 동작 | — | — | Encoder 실패로 무의미 |

### int16 DFP (dynamic_fixed_point) T527 NPU 지원 여부

이전에 "T527 NPU는 int16 미지원"이라 결론 내렸으나 **오류였음**. Zipformer encoder int16 (118MB)이 정상 동작 확인:

| 모델 | int16 NB 크기 | T527 결과 | 비고 |
|------|-------------|-----------|------|
| Zipformer Encoder | **118MB** | **정상 동작** (쓰레기값이지만 crash 없음) | int16 지원 증명 |
| Wav2Vec2 base-korean | 153MB | status=-1 (실행 거부) | NB 크기 제한이 원인 |
| Wav2Vec2 base-960h EN | 152MB | status=-1 (실행 거부) | NB 크기 제한이 원인 |
| Wav2Vec2 XLS-R 12L | 262MB | 미테스트 | 크기 초과 예상 |
| KoCitrinet | 미시도 | — | ONNX 543MB → int16 NB ~120MB 예상, 시도 가치 있음 |

> **결론**: T527 NPU는 int16 DFP를 **지원함**. 실패 원인은 NB 크기 제한 (~120MB 이하 필요). KoCitrinet int16은 NB ~120MB로 예상되어 동작 가능성 있음 (미시도).

---

## 1. KoCitrinet 300f int8 (운용중)

**원본 모델**: NVIDIA NGC Korean Citrinet (`stt_ko_citrinet_256.nemo`, 543MB ONNX)
- 아키텍처: 1D Conv + Squeeze-and-Excitation, CTC
- 입력: `[1, 80, 1, 300]` int8 mel spectrogram (3초, 24KB)
- 출력: `[1, 2049, 1, 38]` int8 CTC logits
- Vocab: 2048 SentencePiece + blank (한국어)

**양자화**: asymmetric_affine int8, 120 calibration samples, moving_average(0.004)

**파일 위치**:
| 파일 | 경로 | 크기 |
|------|------|------|
| ONNX | `ai-sdk/models/ko_citrinet_ngc/citrinet_npu_300f_sim.onnx` | 40MB |
| NB (운용) | `ai-sdk/models/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/network_binary.nb` | 62MB |
| NB meta | 같은 폴더 `nbg_meta.json` | scale=0.02096, zp=-37 |
| NB (5초) | `ai-sdk/models/ko_citrinet_ngc/handover_onnx_to_nb_5s/bundle_int8/network_binary.nb` | 62MB |

**성능 결과**:
| 테스트셋 | CER | CSV 파일 |
|----------|-----|----------|
| sample30 (330샘플, NPU mode7) | **44.44%** | `t527-stt/ko_citrinet/test_results_sample30.csv` |
| modelhouse 3m (51샘플, ONNX) | 28.94% | `ai-sdk/models/ko_citrinet_ngc/whisper_test/modelhouse_3m_onnx_result_CER28.94_TIME0.94.csv` |
| modelhouse 3m (51샘플, NB) | 33.72% | `ai-sdk/models/ko_citrinet_ngc/eval_modelhouse3m_compare/compare_onnx_vs_nb_modelhouse3m.csv` |
| test_manifest (100샘플, ONNX) | 8.44% | `ai-sdk/models/ko_citrinet_ngc/whisper_test/test_manifest_whisper_style_onnx_result_CER8.44_TIME0.93.csv` |
| test_manifest (100샘플, NB) | 12.03% | `ai-sdk/models/ko_citrinet_ngc/whisper_test/test_manifest_whisper_style_nb_result_CER12.03_TIME42.35.csv` |

---

## 2. Wav2Vec2 base-960h 5s uint8 (영어, 검증완료)

**원본 모델**: `facebook/wav2vec2-base-960h` (PyTorch → ONNX, 361MB)
- 아키텍처: CNN feature extractor + 12-layer Transformer, CTC
- 입력: `[1, 80000]` float32 raw audio (5초, 16kHz)
- 출력: `[1, 249, 32]` uint8 CTC logits
- Vocab: 32 (영문자 A-Z + space + blank 등)

**양자화**: asymmetric_affine uint8, reverse_channel 버그 수정 후 재양자화

**파일 위치**:
| 파일 | 경로 | 크기 |
|------|------|------|
| ONNX | `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_960h_5s/wav2vec2_base_960h_5s.onnx` | 361MB |
| NB (fixed) | `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_960h_5s/wksp/wav2vec2_base_960h_5s_uint8_fixed_nbg_unify/network_binary.nb` | 88MB |
| NB meta | 같은 폴더 `nbg_meta.json` | in: scale=0.00285, zp=119 / out: scale=0.1514, zp=179 |

**성능 결과**:
| 테스트셋 | CER | WER | CSV 파일 |
|----------|-----|-----|----------|
| LibriSpeech test-clean (50샘플, NPU) | ~17.52% | ~47.92% | `t527-stt/wav2vec2/base-960h-en/test_results_librispeech.csv` |

---

## 3. Wav2Vec2 base-korean 3s (한국어, 양자화 실패)

**원본 모델**: `Kkonjeong/wav2vec2-base-korean` (PyTorch → ONNX, 361MB)
- 아키텍처: base-960h와 동일 (12L, 768H)
- 입력: `[1, 48000]` float32 raw audio (3초, 16kHz)
- 출력: `[1, 149, 56]` CTC logits
- Vocab: 56 (한글 자모 ㄱ~ㅎ, ㅏ~ㅣ, 겹받침, space, PAD, UNK)

### 3a. ONNX float (서버 CPU) — 성공

**파일**: `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_eager_op12/wav2vec2_ko_eager_op12_3s.onnx` (361MB)

| 테스트셋 | CER | 추론시간 | CSV 파일 |
|----------|-----|----------|----------|
| Zeroth-Korean test 50샘플 (서버CPU) | **33.74%** | 75ms | `ai-sdk/.../clean_pipeline/compute_cer_fair.py` 결과 |
| Zeroth-Korean full (PyTorch CPU) | 7.50% | 462ms | `t527-stt/wav2vec2/base-korean/test_results_zeroth_korean_pytorch_fp32.csv` |
| 월패드 7F_HJY (ONNX CPU) | 높음 | 79ms | `t527-stt/wav2vec2/base-korean/test_results_7F_HJY_onnx_fp32.csv` |
| 월패드 7F_KSK (ONNX CPU) | 높음 | 78ms | `t527-stt/wav2vec2/base-korean/test_results_7F_KSK_onnx_fp32.csv` |
| 모델하우스 2m (ONNX CPU) | 높음 | 78ms | `t527-stt/wav2vec2/base-korean/test_results_modelhouse_2m_onnx_fp32.csv` |
| 모델하우스 3m (ONNX CPU) | 높음 | 79ms | `t527-stt/wav2vec2/base-korean/test_results_modelhouse_3m_onnx_fp32.csv` |

> 참고: 월패드/모델하우스 CER이 높은 이유 — 모델이 자모(ㄱㅏㄴ)로 출력하는데 GT는 음절(간)이라 직접 비교 불가. 별도 자모→음절 조합 필요.

### 3b. NPU uint8 양자화 실험 (전부 실패)

**ONNX 원본**: `wav2vec2_ko_eager_op12_3s.onnx` (361MB, eager attention, opset 12)

| # | 양자화 설정 | quantize 파일 | NB 파일 | NB크기 | non-PAD | CER |
|---|-----------|--------------|---------|--------|---------|-----|
| 1 | 1 sample, MA(0.004) | `wav2vec2_ko_uint8.quantize` | `export_nbg_unify/` | 72MB | ~10% | 거의 blank |
| 2 | 100 samples, MA(0.004) | `wav2vec2_ko_uint8_100iter.quantize` | - | - | - | input range 문제 |
| 3 | 100 samples + input fix | `wav2vec2_ko_uint8_100iter_fixinput.quantize` | `export_ko_100iter_fix_nbg_unify/` | 72MB | 38.9% | ~100% |
| 4 | **300 samples + input fix** | `wav2vec2_ko_uint8_300samples_fix.quantize` | `export_ko_300_nbg_unify/` | 72MB | **46.3%** | **100.86%** |
| 5 | 1000 samples + input fix | `wav2vec2_ko_uint8_1000samples_fix.quantize` | `export_ko_1k_nbg_unify/` | 72MB | 26.6% | worse |
| 6 | min_max algorithm | `wav2vec2_ko_normal_100iter.quantize` | `export_ko_normal_nbg_unify/` | 72MB | ~0% | ALL PAD |
| 7 | KL divergence | `wav2vec2_ko_kl_100iter.quantize` | `export_ko_kl_nbg_unify/` | 72MB | ~0% | ALL PAD |
| 8 | normalized input | `wav2vec2_ko_norm_uint8_100iter_fixinput.quantize` | `export_ko_norm_nbg_unify/` | 72MB | worse | worse |

> 모든 quantize/NB 파일 위치: `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_eager_op12/wksp/clean_pipeline/`
>
> "input fix" = moving_average가 입력 범위를 [-0.12, 0.11]로 축소시키는 문제를 [-1, 1]로 수동 복원

### 3c. NPU fp16 시도 (실패)

| # | 양자화 | NB 파일 | NB크기 | 결과 |
|---|--------|---------|--------|------|
| 1 | float16 | `wav2vec2_ko_base_3s/wksp_export_fp16_nbg_unify/network_binary.nb` | 182MB | CPU fallback, 17.7초 |
| 2 | dfp16 i16 | `wav2vec2_ko_eager_op12/wksp/clean_pipeline/export_dfp16_nbg_unify/network_binary.nb` | 153MB | status=-1 (리소스 부족) |

fp16 NPU 테스트 결과: `t527-stt/wav2vec2/base-korean/test_results_fp16_npu.csv`

---

## 4. Wav2Vec2 XLS-R-300M 3s (한국어, 실패)

**원본 모델**: `facebook/wav2vec2-xls-r-300m` fine-tuned Korean (1.2GB ONNX)
- 24 Transformer layers, 300M params, 2617 vocab (한글 음절)

**파일 위치**: `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_xls_r_300m_3s/`

| 양자화 | NB크기 | 결과 |
|--------|--------|------|
| uint8 (50 samples) | 249MB | ALL PAD — 양자화 오류 누적 |
| uint8 (200 samples) | 249MB | ALL PAD |
| PCQ int8 (200 samples) | 249MB | ALL PAD |

**실패 원인**: 24L + 2617 vocab에서 양자화 오류 누적이 극심. PAD 토큰 logit bias(~23)가 항상 텍스트 토큰(~20)을 이김.

---

## 5. Zipformer Korean (양자화 실패 — 전 방식)

**원본 모델**: `sherpa-onnx-streaming-zipformer-korean-2024-06-16`
- 아키텍처: 5-stack Zipformer encoder (5868 nodes), RNN-Transducer
- 입력: `[1, 39, 80]` mel + 30 cached state tensors (streaming)
- 출력: `[1, 8, 512]` encoder_out + 30 new state tensors
- ONNX baseline CER: **16.2%** (4 Korean test samples, KSS Dataset)

**파일 위치**: `ai-sdk/models/zipformer/zipformer_encoder_folded4/`

### Encoder 양자화 결과 (모두 실패)

| 양자화 | NB크기 | ONNX 대비 상관계수 | CER | 비고 |
|--------|--------|-------------------|-----|------|
| uint8 asymmetric_affine | 63MB | 0.627 | **100%** | state input 수동 교정 |
| int16 dynamic_fixed_point | 118MB | 0.643 | **100%** | state+내부 300개 노드 수동 교정 |
| PCQ int8 perchannel_symmetric | 71MB | 0.275 | **100%** | 오히려 악화 |
| bf16 bfloat16 | — | — | — | export 실패 (error 64768) |

### 실패 원인

1. **양자화 에러 누적**: 5868개 노드 sequential quantization → encoder 출력 상관계수 0.6 수준
2. **Acuity multi-input 캘리브레이션 버그**: 31개 입력 모델에서 state 입력 calibration 무시 (scale=1.0/fl=300)
3. int16 (2배 정밀도)도 correlation 개선 미미 (0.627→0.643)

### Decoder/Joiner NB (정상 변환, 사용 불가)

| 컴포넌트 | NB크기 | 상태 |
|----------|--------|------|
| Decoder | 2.8MB | NB 생성완료 (Encoder 실패로 무의미) |
| Joiner | 1.9MB | NB 생성완료 (Encoder 실패로 무의미) |

> 5868노드 transformer encoder는 Acuity 6.12.0 양자화 한계를 초과. CNN 기반 모델(Citrinet, ~200노드)이나 중간 크기 transformer(Wav2Vec2, ~2000노드)까지만 가능.

---

## 결론

T527 NPU에서 **실제 사용 가능한 STT 모델은 2개**:

1. **KoCitrinet 300f int8** — 한국어, CER 44.44%, 120ms (운용중)
2. **Wav2Vec2 base-960h uint8** — 영어, CER ~17.52%, 715ms (검증완료)

**실패한 모델**:
- 한국어 Wav2Vec2 (base-korean) — ONNX CER 33.74%이나 T527 uint8 양자화 실패
- Zipformer — ONNX CER 16.2%이나 uint8/int16/PCQ/bf16 **전 방식 실패** (5868노드 에러 누적)

**T527 NPU 양자화 경험칙**: CNN 기반(~200노드) 또는 12L 이하 transformer(~2000노드)까지 가능. 5-stack transformer(5868노드)는 불가.

---

## 추가 실험 (2026-03-19)

### KoCitrinet int16 DFP 실험

| 양자화 | NB 크기 | 추론시간 | CER | 결과 |
|--------|---------|---------|-----|------|
| int8 AA (기존) | 62MB | 128ms | **44.44%** | 정상 |
| int16 DFP (신규) | 150MB | 216ms | **330.95%** | **실패** |

**실패 원인**: int16 DFP(dynamic_fixed_point)는 2의 거듭제곱 스케일 + 제로포인트 없음. int8 asymmetric_affine (자유 스케일 + 제로포인트)보다 표현력이 떨어짐. Acuity 6.12에서 int16은 DFP만 가능하고 asymmetric_affine int16은 미지원.

### Wav2Vec2 base-korean 추가 실험

| 시도 | 결과 |
|------|------|
| amplitude norm 5.0 + 기존 uint8 NB | CER 100% — 캘리브레이션 불일치 (NB는 norm 없이 양자화) |
| amp norm 5.0 + KL divergence 재양자화 (12L) | CER 100% — 전 프레임 non-blank |
| 6L pruned 모델 uint8 | CER 100% — 6L 모델이 FP32에서도 garbage (fine-tuning 필요) |
| 3-part split (CNN + L0-5 + L6-11) 각각 uint8 | CER 100% — Part A(CNN)가 입력 무시하고 고정값 출력 |

### T527 NPU 양자화 지원 정리

| quantizer | 지원 qtype | T527 NPU 동작 | 비고 |
|---|---|---|---|
| asymmetric_affine | uint8, int8 | **정상 동작** | 유일하게 신뢰 가능 |
| symmetric_affine | int8 | 미테스트 | |
| perchannel_symmetric | int8 | 동작 (Zipformer corr 0.275) | 효과 미미~악화 |
| dynamic_fixed_point | int16, int8 | **실행되나 결과 부정확** | DFP 2^fl 스케일, zp 없음 |
| bfloat16 | bfloat16 | export 실패 | error 64768 |

> **결론**: T527 NPU에서 실용적인 양자화는 **uint8/int8 asymmetric_affine만**. int16 DFP는 실행은 되지만 int8 AA보다 오히려 나쁜 결과.

---

## 추가 실험 (2026-03-23) — Wav2Vec2 한국어 uint8 양자화 최초 성공

### 영어 base-960h → 한국어 fine-tune 접근

기존 한국어 모델(base-korean)은 logit margin이 극도로 작아 (min 0.005) uint8 양자화 시 argmax 전부 뒤집힘.

**해결**: 영어 base-960h (logit margin min=0.34, uint8 정상 동작) 모델에서 시작하여 한국어 CTC fine-tune.

### 학습

| 단계 | Epochs | LR | Frozen | WER |
|------|--------|-----|--------|-----|
| Attempt 1 | 30 | 1e-4 | CNN | 100% (CTC 수렴 실패) |
| Attempt 2 | 50 | 3e-5 | CNN + L0-5 | 54.18% |
| Attempt 3 | +50 | 1e-5 | CNN only | **44.04%** |

- 데이터: Zeroth-Korean (22,263 train, 457 test)
- GPU: RTX 4070 Ti Super 16GB, 총 ~3시간

### T527 NPU uint8 테스트

```
NB: 72MB (uint8 KL divergence)
ko_test_0001: blank=123/149 | "ㅂㅏㄹㅏㅁ ㅁㅔㄱㄱㅔ ㅇㅏㄹㄴㅔㄴㅇㅣ ㅁ"
ko_test_0002: blank=128/149 | "ㄴㅏㄴㅐ ㅇㅣㄷㅇㅡㄹ ㅋㅡㄹ"
ko_test_0003: blank=131/149 | "ㄱㅡㄹㅣ ㄱㅜ ㅇㅓㄱㅣ ㄴㅏㄴ"
```

**이전까지 한국어 Wav2Vec2 uint8은 CER 100% 전부 실패였으나, 최초로 의미있는 한국어 자모 출력!**

### Logit margin 비교

| 모델 | logit std | margin min | uint8 |
|------|-----------|------------|-------|
| 영어 base-960h (원본) | 8.39 | 0.3400 | 성공 (CER 17.52%) |
| 한국어 base-korean (기존) | 1.95 | 0.0050 | 실패 (CER 100%) |
| **영어→한국어 fine-tune (attempt3)** | **4.03** | **0.0196** | **부분 성공 (자모 출력!)** |

### 개선 방향

1. **NAS 4356시간 데이터로 재학습** — Zeroth-Korean 50시간 → 4356시간이면 WER 대폭 개선
2. **더 긴 학습** — 100+ epochs
3. **Label smoothing / Temperature** — logit margin 추가 확대
