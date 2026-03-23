# wav2vec2-base-korean uint8 양자화 레이어별 디버깅 보고서

**날짜:** 2026-03-23
**작업 디렉토리:** `work/hybrid_v1/wksp_uint8_debug/`
**모델:** wav2vec2-base-korean (Kkonjeong/wav2vec2-base-korean)
**타겟:** Allwinner T527 NPU (VIP9000NANOSI_PLUS)

---

## 1. 배경 및 문제

### 1.1 상황
- 영어 wav2vec2-base-960h는 uint8에서 CER 17.52%로 동작 (문제없음)
- **동일 아키텍처**인 한국어 모델은 uint8 NB에서 non-blank agreement ~58% (PAD bias + KL 최선)
- int16은 시뮬에서 98.8% 일치하나 **T527 디바이스에서 크래시** (전원 꺼짐)
- bf16은 NB 추출 자체 실패 (T527 NPU 미지원)
- **T527에서 안정적으로 돌아가는 건 uint8뿐**

### 1.2 기존 최선 (v18_padbias8_kl)
```
ONNX: wav2vec2_ko_eager_op12_3s.onnx (padbias8 적용)
양자화: Acuity 6.12, asymmetric_affine, uint8, kl_divergence, 50 calib samples
결과: NB_agree ~58%, 디바이스 동작 (~400ms)
```

### 1.3 목표
uint8 양자화 품질을 개선하여 NB_agree를 높이고, 실제 CER을 낮추는 것.

---

## 2. Phase 1: 레이어별 dump 분석

### 2.1 방법
Acuity 6.21의 `pegasus dump` 명령으로 모델의 **모든 레이어 출력**을 FP32와 uint8 각각 추출하여 비교.

```bash
# Docker (t527-npu:v1.2) 안에서:
pegasus dump --model MODEL.json --model-data MODEL.data \
    --dtype float32 --output-dir dump_fp32 --save-file-type tensor
pegasus dump --model MODEL.json --model-data MODEL.data \
    --dtype quantized --model-quantize MODEL_uint8.quantize \
    --output-dir dump_uint8 --save-file-type tensor
```

한국어 모델 530개 레이어, 영어 모델 530개 레이어를 각각 dump.

### 2.2 한국어 모델 단독 결과

| 지표 | 값 |
|------|------|
| 총 레이어 | 530개 |
| 평균 cosine similarity | **0.685** |
| cos < 0.95 | 506/530 (95.5%) |
| cos < 0.80 | 443/530 (83.6%) |

→ **거의 모든 레이어**에서 FP32 대비 큰 열화 발생.

### 2.3 한국어 vs 영어 교차 비교 (핵심 발견)

같은 아키텍처인데 영어에서는 괜찮고 한국어에서만 나쁜 레이어를 식별.

**트랜스포머 레이어별 delta (ko_cos - en_cos):**

| Layer | delta | 해석 |
|-------|-------|------|
| 0~7 | -0.06~+0.02 | 거의 동일 |
| **8** | **-0.11** | 한국어가 뚜렷이 나쁨 |
| **9** | **-0.15** | 심각 |
| **10** | **-0.17** | 가장 심각 |
| **11** | **-0.17** | 가장 심각 |

→ **Layer 8~11이 핵심 문제 영역**. 초기 레이어(0~7)는 한/영 차이 거의 없음.

**가장 심각한 개별 레이어:**
1. Layer 10 final_layer_norm: ko_cos=0.48, en_cos=0.97 (delta=-0.50)
2. Layer 11 attention 전체: delta=-0.33~-0.41
3. Layer 9~10 attention/layernorm: delta=-0.22~-0.27

### 2.4 quantize 파라미터 비교

`.quantize` 파일에서 한국어/영어의 scale, min/max range를 비교.

**한국어 모델의 값 범위가 영어보다 훨씬 넓은 레이어:**

| 레이어 | 한국어 range | 영어 range | 비율 |
|--------|-------------|-----------|------|
| Layer 10 Add (residual) | 450 | 45 | **10x** |
| Layer 10 FFN GELU | 22.6 | 2.8 | **8.2x** |
| Layer 3 Add (residual) | 177 | 23 | **7.5x** |
| Softmax L7,8,9 | 0.8~0.96 | 0.12~0.17 | **5~7x** |

→ 한국어 모델의 후반 레이어(8~11)에서 activation range가 영어보다 훨씬 넓어서 uint8 (256단계)로 표현하기 어려움. 예: range 450을 256단계로 표현하면 해상도가 450/255 ≈ 1.76 per step — 매우 조잡.

---

## 3. Phase 2: 개선 시도

### 3.1 실패한 접근들

#### 3.1.1 Range Clipping (실패)
`.quantize` 파일에서 넓은 range를 가진 레이어의 min/max를 강제로 좁힘.

```
결과: 기존 58% → 27.5% (대폭 악화)
원인: range가 넓은 건 outlier가 아니라 실제 데이터 분포.
      강제 클리핑하면 saturation error 발생 → 오히려 나빠짐.
```

#### 3.1.2 Weight Clipping / SmoothQuant (실패)
ONNX weight의 99.9th percentile 이상을 clip하거나, SmoothQuant(activation→weight 이전) 적용.

```
결과: FP32 자체가 변형됨. uint8 내부 품질은 좋아져도 원본 FP32 대비 나빠짐.
```

#### 3.1.3 PCQ / Symmetric int8 (NB 추출 실패)
`perchannel_symmetric_affine` + int8, `symmetric_affine` + int8.

```
결과: 시뮬에서 동작하나 NB 추출 시 "Fatal model generation error: 65280"
원인: T527 NPU가 int8/symmetric 양자화 NB 생성을 지원하지 않음.
```

#### 3.1.4 bf16 (NB 추출 실패)
```
결과: 시뮬에서 FP32와 100% 일치하나 NB 추출 시 "Fatal model generation error: 64768"
원인: T527 NPU가 bf16 NB를 지원하지 않음.
```

#### 3.1.5 Hybrid int16/uint8 (디바이스 사용 불가)
문제 레이어만 int16, 나머지 uint8.

```
결과: 시뮬 62% (기존 58% 대비 +4%p). NB 추출 성공 (79M).
문제: int16이 포함된 NB는 T527 디바이스에서 크래시 (전원 꺼짐).
```

### 3.2 성공한 접근

#### 3.2.1 `--divergence-first-quantize-bits 16` (핵심 발견)

**Pegasus quantize 명령의 숨겨진 파라미터.**

```bash
pegasus quantize ... \
    --algorithm kl_divergence \
    --divergence-first-quantize-bits 16   # ← 이것
```

**이게 뭘 하는가:**

KL divergence 캘리브레이션 알고리즘의 내부 동작:
1. 캘리브레이션 데이터를 모델에 통과시켜 각 레이어의 activation 분포를 수집
2. 수집된 분포에서 **히스토그램**을 만듦
3. 이 히스토그램에서 **최적 클리핑 포인트**를 찾음 (KL divergence 최소화)
   - 너무 넓으면: 해상도 낮음 (quantization error ↑)
   - 너무 좁으면: 클리핑 error ↑
   - **KL divergence가 최소인 지점**이 최적 트레이드오프
4. 이 과정에서 히스토그램의 **초기 정밀도**를 `first-quantize-bits`로 제어

**기본값은 아마 8비트.** 이걸 16비트로 올리면:
- 히스토그램이 2^16 = 65536 단계로 정밀해짐 (기본 2^8 = 256)
- 더 정밀한 히스토그램 → 최적 클리핑 포인트를 더 정확하게 찾음
- **결과적으로 같은 uint8이지만 더 좋은 scale/zero_point 조합을 얻음**

```
기존 (fqb 기본값): NB_agree ~58% → 70.8% (+12.8%p)
```

**왜 이게 효과적인가:**

한국어 모델의 문제는 Layer 8~11의 activation range가 매우 넓다는 것.
넓은 range에서 최적 클리핑 포인트를 찾는 것은 어렵고, 기본 정밀도(8비트)의
히스토그램으로는 최적점을 정확히 찾지 못함. 16비트 히스토그램이 이 문제를 완화.

#### 3.2.2 Acuity 6.21 re-import

기존 v18은 Acuity 6.12로 import. 6.21로 re-import하면 내부 그래프 최적화가 달라짐.

```
6.12 KL: ~58%
6.21 KL: ~52% (오히려 약간 나빠짐)
6.21 KL + fqb16: ~70.8% (크게 개선)
```

→ 6.21 자체보다 fqb16 파라미터가 핵심.

#### 3.2.3 KL histogram bins 조정

`--divergence-nbins` 파라미터로 KL 히스토그램 bin 수 조정.

| bins | NB_agree |
|------|----------|
| 256 | 56.5% |
| 512 | 69.5% (10-sample) |
| 768 | 69.5% |
| 1024 | 68.8% |
| 4096 (기본) | 61.4% |

→ bins 512~768이 최적. 기본값(아마 2048 or 4096)보다 작은 값이 나음.

#### 3.2.4 PAD bias 튜닝

ONNX의 lm_head bias[53] (PAD 토큰 bias)를 수정하여 양자화 영향 조사.

| PAD bias | avg_Overall | avg_NB_agree | non-blank 수 |
|----------|------------|-------------|-------------|
| 0 (제거) | **70.4%** | 80.3% | 44.5 |
| 1 | 66.2% | 81.2% | 34.1 |
| 2 | 63.0% | 85.1% | 26.0 |
| 3 | 59.7% | 86.9% | 18.7 |
| 4 | 57.1% | 89.9% | 13.6 |
| **8 (기존)** | **65.0%** | **70.8%** | — |
| 10+ | ~50% | 0% | 0 (전부 PAD) |

→ PAD bias가 클수록 non-blank 토큰 수 감소 + 남은 것의 정확도(NB_agree) 증가.
→ CTC 디코딩 관점에서는 padbias 0~2가 유리 (더 많은 정보 출력).

---

## 4. 최종 결과

### 4.1 디바이스 테스트 결과

T527 디바이스에서 vpm_run으로 실제 추론 확인. 모두 안정 동작 (~400ms/inference).

| NB | 추론시간 | vs v18 NB_agree | 특징 |
|----|---------|-----------------|------|
| **kl_fqb16** | 417ms | **70.5%** | 원본모델(pb8) + fqb16 |
| padbias0_fqb16 | 406ms | 89.3% (28 nb) | PAD bias 제거 + fqb16 |
| padbias0_kl | 401ms | 93.1% (30 nb) | PAD bias 제거 + 기본 KL |
| v18 (기존) | 402ms | 기준 (100%) | 기존 최선 |

### 4.2 생성된 NB 파일

`wksp_uint8_debug/` 디렉토리에 총 27개 NB 생성. 모두 순수 uint8, 67M.

**디바이스 테스트 추천 순위:**

1. `wav2vec2_ko_eager_op12_3s_kl_fqb16_nbg_nbg_unify/network_binary.nb`
   - 원본 padbias8 모델 + KL fqb16
   - 가장 균형잡힌 결과

2. `wav2vec2_ko_eager_op12_3s_padbias0_fqb16_nbg_nbg_unify/network_binary.nb`
   - PAD bias 제거 + KL fqb16
   - Overall 최고 (70.4%)

3. `wav2vec2_ko_eager_op12_3s_kl_bins512_nbg_nbg_unify/network_binary.nb`
   - KL bins=512
   - 10-sample NB_agree 69.5%

---

## 5. 기술 요약

### 5.1 핵심 발견

1. **한국어 모델의 uint8 양자화 문제는 Layer 8~11에 집중** (영어 대비 delta -0.11~-0.17)
2. 원인: 후반 레이어의 activation range가 영어보다 5~10x 넓음
3. **`--divergence-first-quantize-bits 16`이 가장 효과적** (+12.8%p NB_agree)
4. Range clipping, weight clipping, SmoothQuant는 모두 실패
5. T527 NPU는 uint8만 안정 지원 (int8/int16/bf16 NB 추출 또는 디바이스 실패)

### 5.2 한계

- uint8 양자화의 물리적 한계: 256단계로 넓은 activation range를 표현하는 데 본질적 제약
- FP32 대비 CER 열화는 여전히 존재
- 3초 고정 입력 모델의 한계 (테스트 오디오가 3초보다 길면 잘림)

### 5.3 향후 방향

- **QAT (Quantization-Aware Training)**: 모델 훈련 단계에서 uint8 양자화를 시뮬레이션하여 양자화에 강건한 weight를 학습
- **더 많은 calibration 데이터**: 현재 50개 → 200~500개로 늘려 calibration 정확도 개선
- **입력 전처리 최적화**: 디바이스 입력의 볼륨 정규화 (현재 NB input range [-0.087, 0.097]에 맞추기)

---

## 6. 재현 방법

```bash
cd work/hybrid_v1/wksp_uint8_debug/

# 1. Docker에서 Acuity 6.21로 ONNX re-import + fqb16 양자화
docker run --rm -v "$(pwd):/workspace" t527-npu:v1.2 bash -c '
  # (torch stub 설치 등 생략)
  pegasus import onnx --model wav2vec2_ko_eager_op12_3s.onnx \
      --output-model wav2vec2_ko_eager_op12_3s_621.json \
      --output-data wav2vec2_ko_eager_op12_3s_621.data

  pegasus quantize --model wav2vec2_ko_eager_op12_3s_621.json \
      --model-data wav2vec2_ko_eager_op12_3s_621.data \
      --device CPU --with-input-meta inputmeta_docker_calib.yml --rebuild-all \
      --model-quantize wav2vec2_ko_eager_op12_3s_kl_fqb16.quantize \
      --quantizer asymmetric_affine --qtype uint8 \
      --algorithm kl_divergence \
      --divergence-first-quantize-bits 16

  pegasus export ovxlib --model wav2vec2_ko_eager_op12_3s_621.json \
      --model-data wav2vec2_ko_eager_op12_3s_621.data \
      --model-quantize wav2vec2_ko_eager_op12_3s_kl_fqb16.quantize \
      --dtype quantized --with-input-meta inputmeta_docker_calib.yml \
      --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
      --viv-sdk $VIV_SDK --target-ide-project linux64 --batch-size 1 \
      --output-path wav2vec2_ko_eager_op12_3s_kl_fqb16_nbg/
'

# 2. 디바이스 테스트 (Windows adb)
ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
$ADB push .../network_binary.nb /data/local/tmp/wav2vec2_uint8_debug/
$ADB shell "vpm_run -s sample.txt"
```

---

## 7. 관련 파일

| 파일 | 설명 |
|------|------|
| `run_dump_uint8.sh` | Docker: FP32 + uint8 레이어별 dump |
| `compare_layers.py` | FP32 vs uint8 레이어별 비교 분석 |
| `compare_ko_vs_en.py` | 한국어 vs 영어 교차 비교 |
| `analyze_quantize.py` | .quantize 파라미터 분석 |
| `run_requantize.sh` | 여러 알고리즘 양자화 + 비교 |
| `run_hybrid_fix.sh` | Hybrid int16/uint8 시도 |
| `run_uint8_sweep.sh` | PCQ/symmetric/SmoothQuant sweep |
| `run_fqb_sweep.sh` | fqb x bins 조합 sweep |
| `run_padbias_finetune.sh` | PAD bias 미세 조정 |
| `layer_comparison.csv` | 한국어 레이어별 cos/MSE 전체 결과 |
| `ko_vs_en_comparison.csv` | 한영 교차 비교 전체 결과 |
| `quantize_comparison.csv` | 양자화 파라미터 한영 비교 |
