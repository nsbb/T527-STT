# Zipformer — RK3588 vs T527 NPU 비교

> 원본: [nsbb/rknn-stt/zipformer](https://github.com/nsbb/rknn-stt/blob/master/zipformer/README.md)

---

## RK3588 NPU 결과

| 구성 | Encoder/chunk | RTF | CER | 비고 |
|------|:---:|:---:|:---:|------|
| **RKNN nocache-static** | **27.5ms** | **0.10** | **22.97%** | INT8, 속도 최적 |
| **RKNN KL divergence** | 53ms | 0.20 | **21.51%** | INT8-KL 100샘플, 정확도 최적 |
| ONNX INT8 CPU 4-thread | 35ms | 0.13 | 19.95% | 비교 기준 |

모델: 15-layer 5-stage Zipformer (~69M params), RKNN INT8, 79~83MB
NPU 단일 코어가 최적 (멀티코어는 오히려 느림).

### 핵심 해결 사항

**1. CumSum 버그 패치 (CER 91.89% → 22.97%)**
- RKNN SDK CumSum이 non-zero 초기값에서 오동작
- 15개 CumSum → 하삼각 MatMul로 교체 (`fix_cumsum.py`)
- 첫 청크는 정상이나, 두 번째 청크부터 캐시 발산 → 수학적 등가 연산으로 해결

**2. 속도 최적화 (52.7ms → 27.5ms)**
```
52.7ms  rknnlite inputs_set (baseline)
→ 39.2ms  C API set_io_mem (DMA zero-copy)
→ 30.7ms  remove_reshape=True (내부 Reshape 제거)
→ 27.5ms  nocache-static (캐시 분리 + 정적 그래프)
```

**3. 양자화 알고리즘**
- KL divergence INT8 (100 calibration samples): CER -1.46pp 개선
- 캘리브레이션 30→100개로 늘리면 CER -2.27pp 개선
- 도메인 불일치 데이터 사용 시 오히려 악화 (37.63%)

### 병목: dispatch overhead
- RKNN 컴파일러가 ~4832개 내부 레이어 생성
- 레이어당 dispatch ~5.9μs → 실제 NPU 연산 2.6ms (9%), overhead 25ms (91%)
- SDK 옵션 15가지 시도 → 모두 무영향. 모델 레이어 수 줄여야만 개선

### RK3588 CER 상세 (test_wavs 4개)

| 파일 | 정답 | RKNN INT8-KL 결과 | CER |
|------|------|-------------------|:---:|
| 3.wav | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? | **0%** |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는... | 부모가 저질에는 큰 실수증 하나는... | 21.2% |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 있지 | 25.0% |
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척할고 에스린 것 같았다 | 41.2% |

---

## T527 NPU 결과 (양자화 전 방식 실패)

| 양자화 | NB 크기 | T527 실행 | Encoder 상관계수 | CER |
|--------|---------|-----------|-----------------|-----|
| uint8 | 63MB | **정상 동작** | 0.627 | 100% |
| int16 | 118MB | **정상 동작** | 0.643 | 100% |
| PCQ int8 | 71MB | **정상 동작** | 0.275 | 100% |
| bf16 | — | export 실패 | — | — |

T527에서 crash 없이 실행되지만, encoder 출력이 ONNX 대비 상관계수 0.3~0.6으로 열화 → 의미있는 텍스트 생성 불가.

---

## 비교

| 항목 | T527 (Acuity 6.12) | RK3588 (RKNN-Toolkit2) |
|------|---------------------|------------------------|
| NPU | 2 TOPS, 1코어 | 6 TOPS, 3코어 (단일 코어 최적) |
| 양자화 | uint8/int16/PCQ **전부 실패** | **INT8 KL 성공** (CER 21.51%) |
| Encoder 속도 | ~50ms/chunk | 27.5ms/chunk |
| CER | 100% (사용 불가) | **21.51%** |
| 캐시 텐서 | 30개 (cached_len 제외) | 35개 (cached_len 포함) |
| 핵심 버그 | Acuity multi-input 캘리브레이션 무시 | CumSum non-zero 초기값 오류 |

### 왜 RK3588은 성공하고 T527은 실패하는가?

1. **양자화 도구 차이**: RKNN-Toolkit2는 KL divergence + 100 calibration sample로 정밀한 양자화 범위 설정. Acuity 6.12는 multi-input 모델에서 state input 캘리브레이션 데이터를 무시하는 버그 존재.

2. **INT8 정밀도 차이**: 동일 INT8이지만 양자화 범위 결정 알고리즘이 다름. RKNN은 레이어별 최적 clipping 범위를 KL divergence로 탐색, Acuity는 moving_average 기반.

3. **5868 노드 누적 오차**: T527 Acuity에서 uint8/int16/PCQ 모두 encoder 출력 상관계수 0.3~0.6. 양자화 도구의 근본적 한계.

---

## T527에서 시도할 수 있는 것

### 1. KL divergence 양자화 (`--algorithm kl_divergence`)

RK3588에서 CER -1.46pp 개선. Acuity 6.12에도 옵션 존재. 다만 5868노드 모델의 근본적 한계를 넘기 어려울 것으로 예상.

### 2. CumSum 패치 적용 여부

T527 Acuity에서는 CumSum 버그 없음 (Acuity는 다른 백엔드). 대신 negative Gather index 버그가 있어 별도 패치 (`fix_negative_gather_v2.py`).

### 3. ONNX CPU 실행

T527 ARM Cortex-A55에서 ONNX Runtime으로 실행. RK3588에서 ONNX INT8 CPU 4-thread가 35ms → T527 A55에서는 ~100-200ms/chunk 예상. 실시간 가능하지만 NPU 활용 불가.

> **결론: Zipformer는 T527 NPU (Acuity)에서 양자화 불가. RK3588 NPU (RKNN)에서만 동작.** 동일 모델이라도 양자화 도구에 따라 결과가 완전히 다름.
