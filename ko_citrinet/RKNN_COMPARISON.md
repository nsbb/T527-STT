# KoCitrinet — RK3588 vs T527 NPU 비교

> 원본: [nsbb/rknn-stt/ko_citrinet](https://github.com/nsbb/rknn-stt/blob/master/ko_citrinet/README.md)

---

## RK3588 NPU 결과 (FP16)

| 항목 | 값 |
|------|-----|
| 양자화 | FP16 |
| 모델 크기 | 281MB |
| 추론 속도 | 52.5ms (3초 오디오, NPU 3코어) |
| RTF | 0.0175 (57배 실시간) |
| ONNX↔RKNN Cosine | 0.999935 |
| 입력 | `[1, 80, 300]` mel spectrogram (3D — Squeeze 제거) |
| 출력 | `[1, 2049, 38]` CTC logits (3D) |
| CER (368샘플) | 39.9% |
| 평균 지연 | 63ms |

### 인식 결과 (4개 테스트): ONNX와 **100% 동일**

```
[57ms] call_elevator.wav: 엘리베이터블러서
[78ms] check_weather.wav: 날씨알려줬서
[53ms] sample.wav:         조명이 켜졌습니다
[97ms] turn_on_light.wav:  불켜서
```

### RKNN에서 수정한 버그 4건

| Fix | 내용 | T527 해당 여부 |
|-----|------|:---:|
| 1. LogSoftmax 제거 | RKNN 빌드 크래시 | **해당 없음** |
| 2. SE 마스크 → ReduceMean | 530 노드 제거 (고정 입력에서 마스크 불필요) | 해당 없음 (Acuity 정상 처리) |
| 3. ReduceMean → depthwise Conv | RKNN ReduceMean 버그 | **해당 없음** (Acuity 정상) |
| 4. Squeeze/Unsqueeze 제거 (**핵심**) | RKNN이 NCHW↔NHWC 변환 시 axis 오적용 | **해당 없음** (Acuity 정상) |

> T527 Acuity에서는 이 4개 버그가 모두 없으므로 ONNX 수정 없이 직접 변환 가능.

---

## T527 NPU 결과 (int8)

| 항목 | 값 |
|------|-----|
| 양자화 | int8 asymmetric_affine (moving_average) |
| 모델 크기 | 62MB |
| 추론 속도 | 120ms (3초 오디오, NPU 1코어) |
| RTF | 0.04 (25배 실시간) |
| CER (330샘플, mode7) | 44.44% |
| ONNX↔NB CER 차이 | +0.33%p |
| 입력 | `[1, 80, 1, 300]` mel spectrogram (4D) |
| 출력 | `[1, 2049, 1, 38]` CTC logits (4D) |

---

## 비교

| 항목 | T527 (Acuity) | RK3588 (RKNN) |
|------|--------------|---------------|
| NPU | 2 TOPS, 1코어 | 6 TOPS, 3코어 |
| 양자화 | **int8** (62MB) | **FP16** (281MB) |
| 속도 | 120ms | 52.5ms (63ms avg) |
| RTF | 0.04 | 0.0175 |
| CER | 44.44% (330샘플) | 39.9% (368샘플) |
| 양자화 열화 | +0.33%p (int8 vs ONNX) | 거의 없음 (FP16, cosine 0.9999) |
| ONNX 그래프 수정 | 불필요 | 4건 필요 (Squeeze, ReduceMean 등) |
| 입출력 차원 | 4D `[1,80,1,300]` | 3D `[1,80,300]` (Squeeze 제거) |

### CER 차이 원인 (44.44% vs 39.9%)

1. **양자화**: T527 int8 vs RK3588 FP16. int8은 +0.33%p 열화
2. **테스트셋**: T527 330개 (자사 수집) vs RK3588 368개 (자사 수집). 동일 테스트셋이 아님
3. **mel 전처리**: T527 mode7 (Slaney mel, log10) vs RK3588 (NeMo 기본)
4. **양자화 열화가 매우 적음** (+0.33%p) — CER 차이의 대부분은 테스트셋/전처리 차이

---

## T527 적용 가능한 개선 사항

### 1. int16 DFP 시도 (미시도, 가장 유망)

T527에서 int16 DFP가 동작하는 것이 Zipformer 118MB로 확인됨.

- KoCitrinet uint8 NB: 62MB → int16 NB: **~120MB** (추정)
- 128MB 제한 이내 → **동작 가능성 높음**
- RK3588 FP16 (cosine 0.9999) 수준의 정밀도 기대
- int8에서 이미 열화가 +0.33%p로 적지만, int16이면 거의 0에 가까울 것

```bash
# Acuity int16 양자화
pegasus quantize --quantizer dynamic_fixed_point --qtype int16 \
  --algorithm moving_average ...
```

### 2. KL divergence 양자화 (미시도)

현재 `moving_average` 알고리즘 사용 중. KL divergence로 변경 시 CER 추가 개선 가능.

- RK3588 Zipformer에서 KL: CER -1.46pp
- KoCitrinet은 이미 양자화 열화가 적어 효과 제한적일 수 있음

```bash
pegasus quantize --algorithm kl_divergence ...
```

### 3. 5초 입력 (KoCitrinet 500f)

500f (5초) NB도 62MB로 크기 동일. 더 긴 입력을 처리할 수 있어 긴 발화에 유리. awaiasr_2 앱에 탑재 완료 (정확도 미측정).

### 4. Amplitude normalization

RK3588 wav2vec2에서 CER 24pp 개선한 기법. CitriNet은 mel spectrogram 입력이라 raw waveform 기반 amplitude norm은 직접 적용 불가. 대신 mel 레벨의 에너지 정규화를 시도할 수 있음.

---

## 참고: CitriNet 모델 구조

```
Input mel [1, 80, 300] (또는 [1, 80, 1, 300])
  ↓
Prologue Conv (80 → 1024)
  ↓
22x Jasper Block:
  ├── Depthwise Separable Conv (1D)
  ├── SE (Squeeze-and-Excitation) block
  └── Residual connection + ReLU
  ↓
Decoder Conv (1024 → 2049)
  ↓
Output logits [1, 2049, 38]  →  CTC greedy decode  →  텍스트
```

순수 CNN 모델 → Transformer 대비 양자화에 강함. T527/RK3588 모두 양자화 성공.
