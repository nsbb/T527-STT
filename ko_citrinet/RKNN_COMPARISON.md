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
| 입력 | `[1, 80, 300]` mel spectrogram |
| 출력 | `[1, 2049, 38]` CTC logits |

인식 결과 (4개 테스트): ONNX와 **100% 동일**.

## T527 NPU 결과 (int8)

| 항목 | 값 |
|------|-----|
| 양자화 | int8 asymmetric_affine |
| 모델 크기 | 62MB |
| 추론 속도 | 120ms (3초 오디오, NPU 1코어) |
| RTF | 0.04 (25배 실시간) |
| CER | 44.44% (330샘플 mode7) |
| ONNX↔NB CER 차이 | +0.33%p |

## 비교

| 항목 | T527 (Acuity) | RK3588 (RKNN) |
|------|--------------|---------------|
| NPU | 2 TOPS, 1코어 | 6 TOPS, 3코어 |
| 양자화 | int8 (62MB) | FP16 (281MB) |
| 속도 | 120ms | 52.5ms |
| RTF | 0.04 | 0.0175 |
| 정확도 손실 | +0.33%p (int8 vs ONNX) | 거의 없음 (FP16, cosine 0.9999) |

## RK3588에서 발견한 RKNN 버그 (T527에는 해당 없음)

1. **Squeeze/Unsqueeze 버그** — RKNN이 NCHW↔NHWC 변환 시 axis를 잘못 적용. T527 Acuity에서는 정상 동작.
2. **ReduceMean 버그** — depthwise Conv로 교체 필요. T527 Acuity에서는 정상 동작.
3. **LogSoftmax 크래시** — T527에서는 CTC greedy decoding에 argmax만 사용하므로 무관.

## T527 적용 가능한 개선 사항

### 1. int16 DFP 시도 (미시도)

T527에서 int16 DFP가 동작하는 것이 Zipformer 118MB로 확인됨. KoCitrinet int16 NB는 **~120MB로 추정** (128MB 제한 이내).

int8에서 양자화 열화가 +0.33%p로 매우 적지만, int16이면 FP32와 거의 동일한 정확도를 기대할 수 있음. RK3588의 FP16 결과 (cosine 0.9999)가 이를 뒷받침.

```bash
# Acuity int16 양자화
pegasus quantize --quantizer dynamic_fixed_point --qtype int16 ...
```

### 2. KL divergence 양자화 (미시도)

현재 `moving_average` 알고리즘 사용 중. KL divergence로 변경하면 CER 추가 개선 가능.

```bash
pegasus quantize --algorithm kl_divergence ...
```

### 3. 5초 입력 (KoCitrinet 500f)

500f (5초) NB도 62MB로 크기 동일. 더 긴 입력을 처리할 수 있어 긴 발화에 유리. awaiasr_2 앱에 탑재 완료 (정확도 미측정).
