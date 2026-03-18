# 영어/한국어 Wav2Vec2 ONNX 구조 비교 분석

> 작성일: 2026-03-18 | 상태: 재변환 + uint8 양자화 + 디바이스 검증 완료

## 요약

**영어(base-960h)와 한국어(base-korean) ONNX 모델의 attention 구현이 완전히 다르다.**
원인은 ONNX export 시 `attn_implementation`과 `opset_version` 차이.
`attn_implementation="eager"` + `opset_version=12`로 재변환하면 **영어와 100% 동일한 구조**가 된다.

이것이 양자화 실패의 유일한 원인은 아니지만 (activation 분포 차이가 근본 원인),
ONNX 구조 차이가 Acuity 양자화 품질을 **추가로 악화**시켰을 가능성이 높다.

---

## 1. 분석 대상

| 모델 | 파일 | 크기 | opset |
|------|------|------|-------|
| **EN** (영어, 동작함) | `wav2vec2_base_960h_5s.onnx` | 361MB | **12** |
| **KO-orig** (한국어, 실패) | `wav2vec2_ko_base_3s.onnx` | 361MB | **14** |
| **KO-sim** (한국어, 실패) | `wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx` | 361MB | 12 (다운그레이드) |
| **KO-eager** (한국어, 신규) | 재변환 필요 | ~361MB | **12** (네이티브) |

---

## 2. 노드 수 비교

| 모델 | 총 노드 | MatMul | Reshape | Transpose | Softmax | Shape | Cast | Gather | Concat | Unsqueeze |
|------|---------|--------|---------|-----------|---------|-------|------|--------|--------|-----------|
| **EN** | **957** | 98 | 98 | 63 | 12 | 1 | 0 | 0 | 0 | 1 |
| **KO-orig** | **1306** | 98 | 50 | 51 | 12 | 37 | 24 | 24 | 48 | 73 |
| **KO-sim** | **667** | 98 | 48 | 51 | 12 | 0 | 0 | 0 | 0 | 1 |
| **KO-eager** | **957** | 98 | 98 | 63 | 12 | 1 | 0 | 0 | 0 | 1 |

KO-orig은 EN보다 **349개 노드가 더 많다**. KO-eager는 EN과 **완전히 동일**.

---

## 3. Attention 구조 차이 (핵심)

### 3.1 Scale 적용 위치

**EN (opset 12, eager attention):**
```
Q = Q_proj(x) * 0.125          ← Scale을 Q에 선곱
score = Q_scaled @ K^T          ← MatMul 출력이 이미 최종 범위
softmax(score) → attn @ V
```

**KO-orig (opset 14, SDPA):**
```
Q = Q_proj(x)                   ← Scale 없음
raw_score = Q @ K^T              ← MatMul 출력이 8배 큰 범위!
score = raw_score * 0.125        ← Scale을 후곱
softmax(score) → attn @ V
```

**양자화 영향:**
- EN: MatMul 출력 범위 ~[-20, +20] → uint8 scale ≈ 0.16/step
- KO: MatMul 출력 범위 ~[-160, +160] → uint8 scale ≈ 1.25/step (8배 거침)
- KO에서 Mul(0.125) 후 최종 범위는 같지만, **MatMul 출력 텐서가 이미 거친 양자화로 정보 손실**
- 추가로 KO는 MatMul→Mul 사이에 **양자화 경계가 하나 더 존재** → 오류 누적 증가

### 3.2 MatMul 텐서 차원

**EN (3D MatMul):**
```
Q: [1, seq, 768] → Reshape [1, seq, 12, 64] → Transpose [1, 12, seq, 64]
                 → Reshape [12, seq, 64]       ← batch×head 병합 (3D)
K^T: 동일 패턴 → [12, 64, seq]
score = MatMul([12, seq, 64], [12, 64, seq]) → [12, seq, seq]   ← 3D MatMul
```

**KO-orig (4D MatMul):**
```
Q: [1, seq, 768] → Reshape [1, seq, 12, 64] → Transpose [1, 12, seq, 64]   ← 4D 유지
K^T: 동일 패턴 → [1, 12, 64, seq]
score = MatMul([1, 12, seq, 64], [1, 12, 64, seq]) → [1, 12, seq, seq]   ← 4D MatMul
```

**양자화 영향:**
- Acuity는 per-tensor 양자화 → 3D든 4D든 동일 scale/zp 적용
- 그러나 Acuity 내부 그래프 최적화가 3D/4D를 **다르게 처리**할 수 있음
- EN의 3D 패턴이 Acuity에서 더 효율적으로 처리될 가능성

### 3.3 Softmax Axis

| 모델 | 텐서 rank | Softmax axis | 의미 |
|------|-----------|-------------|------|
| EN | 3D `[12, seq, seq]` | axis=**2** | last dim |
| KO-orig | 4D `[1, 12, seq, seq]` | axis=**3** | last dim |

의미적으로 동일하지만 Acuity op mapping이 다를 수 있음.

### 3.4 K Transpose 패턴

| 모델 | 방식 | Transpose 횟수 |
|------|------|---------------|
| EN | `perm=[0,2,1,3]` + `perm=[0,2,1]` (2단계) | 5회/layer |
| KO-orig | `perm=[0,2,1,3]` + `perm=[0,2,3,1]` (결합) | 4회/layer |

### 3.5 동적 Shape 연산 (KO-orig 전용)

KO-orig은 layer당 **18개 동적 shape 연산** (Shape, Gather, Unsqueeze, Concat, Cast, Slice)을 추가로 포함.
12 layers × 18 = **216개 추가 노드**. 이 노드들은 Reshape 대상 shape를 런타임에 계산.

EN은 모든 Reshape shape가 **정적 Constant**로 컴파일 타임에 확정.

---

## 4. 원인: ONNX Export 설정 차이

### 4.1 `aten::scaled_dot_product_attention` (SDPA)

```python
# 현재 transformers (4.46.3) 기본값: SDPA 사용
# SDPA는 ONNX opset 14+ 전용
# → 4D tensor, post-scaling, dynamic shapes

# 해결: eager attention 강제
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, attn_implementation="eager")
torch.onnx.export(model, dummy, output, opset_version=12)
# → EN과 100% 동일한 구조
```

### 4.2 Export 조건 비교

| 설정 | EN (성공) | KO-orig (실패) | KO-eager (예상 성공) |
|------|----------|---------------|-------------------|
| transformers 버전 | 구버전 (SDPA 이전) | 4.41.2+ (SDPA 도입) | 4.46.3 |
| `attn_implementation` | eager (기본값) | **sdpa** (기본값 변경) | **eager** (강제) |
| `opset_version` | **12** | **14** | **12** |
| ONNX 노드 수 | 957 | 1306 | **957** |

### 4.3 검증 결과

```
KO eager opset 12:  957 nodes  (= EN 957 nodes)
  MatMul=98, Reshape=98, Transpose=63, Softmax=12, Shape=1, Cast=0

EN original opset 12: 957 nodes
  MatMul=98, Reshape=98, Transpose=63, Softmax=12, Shape=1, Cast=0

Layer 0 attention 노드 구조: 1:1 완전 대응 확인 ✓
```

---

## 5. 동일한 부분 (차이 없음)

| 구성 요소 | 구조 비교 | 비고 |
|----------|----------|------|
| CNN feature extractor (7 Conv) | **동일** | kernel/stride/group 모두 일치 |
| Feature projection (LN + MatMul) | **동일** | |
| Positional conv embedding | **동일** | kernel=128, group=16 |
| LayerNorm (모든 위치) | **동일** | ReduceMean→Sub→Pow→ReduceMean→Add→Sqrt→Div→Mul→Add |
| FFN (intermediate_dense + dense) | **동일** | MatMul→Add→GELU→MatMul→Add |
| GELU activation | **동일** | Div→Erf→Add→Mul→Mul 패턴 |
| 가중치 shape | **동일** | lm_head만 32 vs 56 (vocab 차이) |
| Initializer 수 | **동일** | 211개 |

---

## 6. 가중치 값 비교 (참고)

### CNN feature extractor

| 가중치 | max_diff | mean_diff | 비고 |
|--------|----------|-----------|------|
| conv_layers.0.conv.weight | 0.015 | 0.002 | KO fine-tuning으로 미세 변경 |
| conv_layers.1.conv.weight | 0.027 | 0.003 | |
| conv_layers.6.conv.weight | 0.023 | 0.003 | |
| feature_projection.layer_norm | 0.136 | 0.024 | |
| encoder.layer_norm | 0.188 | 0.024 | |

> CNN 가중치 차이가 작음 → KO는 `facebook/wav2vec2-base`에서 출발, CTC fine-tuning으로 미세 조정.

### Attention 가중치 (Layer 0, 5, 11)

| Layer | Proj | EN std | KO std | KO/EN |
|-------|------|--------|--------|-------|
| L0 | q_proj.weight | 0.0861 | 0.0795 | 0.92 |
| L0 | k_proj.bias | 0.0399 | **0.0029** | **0.07** |
| L5 | q_proj.weight | 0.0993 | 0.0914 | 0.92 |
| L5 | k_proj.bias | 0.0686 | 0.0436 | 0.64 |
| **L11** | **q_proj.weight** | **0.1506** | **0.0877** | **0.58** |
| **L11** | **k_proj.bias** | **15.8573** | **0.2050** | **0.01** |

**L11 k_proj.bias**: EN=±18 (peaked attention 유도), KO=±1 (uniform attention).
**L11 q_proj.weight**: EN의 std가 KO의 1.7배 → EN이 더 강한 Q 투영.

### LM head bias

| 모델 | range | 특이사항 |
|------|-------|---------|
| EN | [-1.39, 0.49] | 균등한 분포 |
| KO | [**-9.98**, 0.01] | 하나의 극단적 outlier (-9.98 = blank 토큰 억제) |

---

## 7. 양자화 영향 분석

### ONNX 구조 차이가 양자화에 미치는 영향

| 차이 | EN (유리) | KO-orig (불리) | 영향 정도 |
|------|----------|--------------|----------|
| Scale 위치 (pre vs post) | MatMul 출력 범위 작음 | MatMul 출력 8배 큼 → 거친 양자화 | **중간** |
| 양자화 경계 수 | 1개 (MatMul 출력) | 2개 (MatMul + Mul 출력) | **중간** |
| 텐서 차원 (3D vs 4D) | Acuity 최적화 유리 추정 | 다를 수 있음 | **낮음~중간** |
| 동적 Shape ops (0 vs 216) | 정적 그래프 | Acuity가 동적 op 처리 가능성 | **낮음** |

### 가중치/활성값 차이의 영향 (근본 원인)

| 차이 | EN (유리) | KO (불리) | 영향 정도 |
|------|----------|----------|----------|
| Attention 분포 (peaked vs uniform) | top-1 66.6%, uint8 argmax 99.4% | top-1 1.8%, uint8 argmax 51.0% | **매우 높음** |
| Per-layer error | 1.3% → 12L 후 85% | 6.2% → 12L 후 46.3% | **매우 높음** |
| k_proj bias L11 | ±18 (peaked 유도) | ±1 (중립) | **높음** |
| lm_head bias outlier | 없음 | -9.98 (1개) | **낮음** |

> **결론: ONNX 구조 차이는 중간 정도의 추가 악화 요인. 근본 원인은 attention 분포 차이.**

---

## 8. 해결 방안: Eager Opset 12 재변환

### 즉시 실행 가능한 액션

```python
#!/usr/bin/env python3
"""Korean wav2vec2 → EN-identical ONNX structure (eager attention, opset 12)"""
import torch
from transformers import Wav2Vec2ForCTC

MODEL_ID = "Kkonjeong/wav2vec2-base-korean"  # 또는 로컬 경로
INPUT_LENGTH = 48000  # 3초

# 핵심: eager attention 강제 → SDPA 비활성화
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, attn_implementation="eager")
model.eval()

dummy = torch.randn(1, INPUT_LENGTH)
with torch.no_grad():
    out = model(dummy)
    print(f"Output: {out.logits.shape}")

# 핵심: opset 12 → EN과 동일한 3D MatMul + pre-scaling
torch.onnx.export(
    model, dummy,
    "wav2vec2_ko_base_3s_eager_op12.onnx",
    input_names=["input_values"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
)

# Shape 고정 (Acuity용)
import onnx
m = onnx.load("wav2vec2_ko_base_3s_eager_op12.onnx")
for inp in m.graph.input:
    if inp.name == "input_values":
        inp.type.tensor_type.shape.dim[0].dim_value = 1
        inp.type.tensor_type.shape.dim[1].dim_value = INPUT_LENGTH
for out_node in m.graph.output:
    if out_node.name == "logits":
        out_node.type.tensor_type.shape.dim[0].dim_value = 1
        out_node.type.tensor_type.shape.dim[1].dim_value = out.logits.shape[1]
        out_node.type.tensor_type.shape.dim[2].dim_value = out.logits.shape[2]
onnx.save(m, "wav2vec2_ko_base_3s_eager_op12.onnx")
print("Done. Structure matches EN (957 nodes, opset 12).")
```

### 실측 결과 (2026-03-18)

재변환 후 실제 양자화 + 디바이스 테스트를 수행했다.

#### Pegasus 시뮬레이션 (10 samples)

| 측정 | Old SDPA KO | **New Eager KO** | EN base-960h |
|------|------------|-----------------|-------------|
| **Argmax agreement (overall)** | 46.3% | **78.1% ± 9.5%** | ~85% |
| Min / Max | — | 66.4% / 90.6% | — |

→ 전체 argmax agreement **+31.8%p** 대폭 개선.

#### T527 NPU 디바이스 테스트

```
입력: [1, 48000] uint8, scale=0.0666, zp=131
출력: [1, 149, 56] uint8, scale=0.0690, zp=76
추론 시간: 415ms (3초 모델)
NB 크기: 72MB
```

- NPU vs FP32 argmax agreement: **89.9%**
- NPU vs uint8 sim agreement: **97.3%** (시뮬레이션 정확)

#### 핵심 한계: Non-blank 토큰 accuracy

```
FP32 non-pad frames: 13/149 (나머지 136은 [PAD])
Non-pad 프레임에서의 accuracy: 0%
```

FP32에서 99%+ 확신도로 한글 자모(`ㄸ`, `ㅓ`, `ㄴ`, `ㅌ`, `ㅐ` 등)를 출력하는 프레임이
uint8 양자화 후 **전부 `[PAD]`로 전환**됨.

| 디코딩 | 텍스트 |
|--------|-------|
| FP32 | `ㄸㅓㄴ ㅌㅐ ㅇㅣ ㅂㅇㅡㄹ` |
| uint8 NPU | `ㅔ ㅇ` |

→ 전체 agreement는 높아도 (FP32도 91% [PAD]), 실제 음성 정보가 있는 프레임에서 양자화 파괴.

### 결론

| 시나리오 | 예측 | **실측** |
|---------|------|---------|
| **최선** | 60-70% | **78.1%** (예측 초과) |
| **중간** | 50-55% | — |
| **최악** | 46% | — |

구조 재변환은 **예측을 초과하는 개선**을 가져왔다.
그러나 non-blank accuracy 0%는 **근본 원인(attention 분포 차이)이 지배적**임을 확인.
Eager opset12 ONNX를 기반으로 QAT(전략 A) 또는 Knowledge Distillation(전략 D2) 진행 필요.

---

## 9. 후속 작업

1. ~~즉시: eager opset 12로 재변환~~ ✅ 완료 (957 nodes, 377.9MB)
2. ~~양자화 + 시뮬레이션~~ ✅ 완료 (argmax agreement 78.1%)
3. ~~NB 생성 + 디바이스 테스트~~ ✅ 완료 (72MB NB, 415ms, non-blank accuracy 0%)
4. **QAT (전략 A)**: 재변환된 ONNX 기반으로 양자화 인식 학습 → attention 분포 강화
5. **Knowledge Distillation (전략 D2)**: Transformer teacher → CNN student
6. **참고**: 5초 모델(`INPUT_LENGTH=80000`)로도 변환 가능
