# T527 NPU STT 모델 인벤토리

모든 시도한 모델, 양자화 방법, NB 파일, 성능 결과를 정리한 문서.

---

## 최종 성능 요약

| # | 모델 | 언어 | 양자화 | NB크기 | T527 NPU | CER | 추론시간 | 상태 |
|---|------|------|--------|--------|----------|-----|----------|------|
| 1 | KoCitrinet 300f | 한국어 | int8 asymmetric | 62MB | 동작 | **44.44%** | 120ms | **운용중** |
| 2 | Wav2Vec2 base-960h 5s | 영어 | uint8 asymmetric | 88MB | 동작 | **~17.52%** | 715ms | 검증완료 |
| 3 | Wav2Vec2 base-korean 3s | 한국어 | ONNX float (CPU) | - | CPU only | **33.74%** | 75ms(서버) | 서버용만 가능 |
| 4 | Wav2Vec2 base-korean 3s | 한국어 | uint8 (300 calib) | 72MB | 동작 | 100.86% | 415ms | **실패** |
| 5 | Wav2Vec2 base-korean 3s | 한국어 | fp16 (182MB) | 182MB | CPU fallback | - | 17,740ms | **실패** (HW 미가속) |
| 6 | Wav2Vec2 base-korean 3s | 한국어 | dfp16 i16 (153MB) | 153MB | status=-1 | - | - | **실패** (리소스 부족) |
| 7 | Wav2Vec2 XLS-R-300M 3s | 한국어 | uint8 | 249MB | 동작 | ALL PAD | 1098ms | **실패** |
| 8 | Zipformer Encoder | 한국어 | uint8 | 63MB | 미테스트 | - | - | NB 생성완료 |

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

## 5. Zipformer Korean (NB 생성완료, 미테스트)

**원본 모델**: `sherpa-onnx-streaming-zipformer-korean-2024-06-16`

**파일 위치**: `ai-sdk/models/zipformer/bundle_uint8/`

| 컴포넌트 | NB크기 | 상태 |
|----------|--------|------|
| Encoder | 63MB | NB 생성완료 |
| Decoder | 2.8MB | NB 생성완료 |
| Joiner | 1.9MB | NB 생성완료 |

> Encoder+Decoder+Joiner 3개를 연동하는 Android 파이프라인 미구현

---

## 결론

T527 NPU에서 **실제 사용 가능한 STT 모델은 2개**:

1. **KoCitrinet 300f int8** — 한국어, CER 44.44%, 120ms (운용중)
2. **Wav2Vec2 base-960h uint8** — 영어, CER ~17.52%, 715ms (검증완료)

한국어 Wav2Vec2는 ONNX float CER 33.74%로 KoCitrinet보다 우수하나,
T527 NPU uint8 양자화가 출력을 완전히 파괴하여 사용 불가.
