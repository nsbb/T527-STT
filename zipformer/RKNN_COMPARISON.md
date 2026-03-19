# Zipformer — RK3588 vs T527 NPU 비교

> 원본: [nsbb/rknn-stt/zipformer/rk3588](https://github.com/nsbb/rknn-stt/blob/master/zipformer/rk3588/README.md)

---

## RK3588 NPU 결과 (INT8)

| 항목 | 값 |
|------|-----|
| 양자화 | INT8 (KL divergence, 100-sample 캘리브레이션) |
| 모델 크기 | 79MB (nocache-static) |
| Encoder 속도 | 27.5ms/chunk (320ms 음성) |
| RTF | 0.10 (10배 실시간) |
| CER | 21.85% (KL), 22.97% (일반 INT8) |
| CPU 대비 | 22% 빠름 (vs ONNX INT8 CPU 35ms) |

### CumSum 버그 수정이 핵심

RKNN SDK의 CumSum이 non-zero 초기값에서 오동작 → 15개 CumSum을 하삼각 MatMul로 교체.
- 패치 전: CER **91.89%** (사용 불가)
- 패치 후: CER **22.97%** (-68.92pp)

### 속도 최적화

```
52.7ms  rknnlite inputs_set (baseline)
→ 39.2ms  C API set_io_mem (DMA zero-copy)
→ 30.7ms  remove_reshape=True (내부 Reshape 제거)
→ 27.5ms  nocache-static (캐시 분리 + 정적 그래프)
```

### 병목: dispatch overhead

RKNN 컴파일러가 ~4832개 내부 레이어 생성. 레이어당 ~5.9μs dispatch overhead.
실제 NPU 연산: 2.6ms (9%), dispatch overhead: 25ms (91%).

## T527 NPU 결과 (양자화 전 방식 실패)

| 양자화 | NB 크기 | T527 실행 | Encoder 상관계수 | CER |
|--------|---------|-----------|-----------------|-----|
| uint8 | 63MB | **정상 동작** | 0.627 | 100% |
| int16 | 118MB | **정상 동작** | 0.643 | 100% |
| PCQ int8 | 71MB | **정상 동작** | 0.275 | 100% |
| bf16 | — | export 실패 | — | — |

T527에서는 crash 없이 실행되지만, encoder 출력이 ONNX 대비 상관계수 0.6 수준으로 열화되어 의미있는 텍스트 생성 불가.

## 비교

| 항목 | T527 (Acuity) | RK3588 (RKNN) |
|------|--------------|---------------|
| NPU | 2 TOPS, 1코어 | 6 TOPS, 1코어 (최적) |
| 양자화 | uint8/int16/PCQ 전부 실패 | **INT8 KL 성공** (CER 21.85%) |
| Encoder 속도 | ~50ms/chunk | 27.5ms/chunk |
| CER | 100% (사용 불가) | **21.85%** |
| 핵심 버그 | Acuity multi-input 캘리브레이션 무시 | CumSum non-zero 초기값 오류 |

### 왜 RK3588은 성공하고 T527은 실패하는가?

1. **양자화 도구 차이**: RKNN-Toolkit2는 KL divergence + 100 calibration sample로 정밀한 양자화 범위 설정. Acuity 6.12는 multi-input 모델에서 state input 캘리브레이션 데이터를 무시하는 버그 존재.

2. **INT8 정밀도 차이**: 동일 INT8이지만 양자화 범위 결정 알고리즘이 다름. RKNN은 레이어별 최적 clipping 범위를 KL divergence로 탐색, Acuity는 moving_average 기반.

3. **5868 노드 누적 오차**: T527 Acuity에서 uint8/int16/PCQ 모두 encoder 출력 상관계수 0.3~0.6. 양자화 도구의 한계.

## T527에서 시도할 수 있는 것

### 1. KL divergence 양자화 (Acuity `--algorithm kl_divergence`)

RK3588에서 CER -1.12pp 개선. T527에서도 Acuity 6.12에 옵션 존재하나, 5868노드 모델의 근본적 한계를 넘기 어려울 것으로 예상.

### 2. Amplitude normalization (encoder 입력 전처리)

Wav2Vec2에서 효과적이었던 기법. Zipformer의 mel spectrogram 입력에도 amplitude norm을 적용하면 INT8 활용도 개선 가능. 다만 Zipformer는 mel 입력이라 raw waveform 대비 효과 제한적.

### 3. 포기하고 ONNX CPU 실행

T527 ARM Cortex-A55에서 ONNX Runtime으로 실행. RK3588에서 ONNX INT8 CPU 4-thread가 35ms인 점 참고하면, T527 A55에서는 ~100-200ms/chunk 예상. 실시간 가능하지만 NPU 활용 불가.

> **결론: Zipformer는 T527 NPU에서 양자화 불가. RK3588 RKNN에서만 동작.** 동일 모델이라도 양자화 도구(Acuity vs RKNN-Toolkit2)에 따라 결과가 완전히 다름.
