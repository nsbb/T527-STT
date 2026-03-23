# Wav2Vec2 영어→한국어 Fine-tuning 보고서

## 개요

영어 `facebook/wav2vec2-base-960h` 모델에서 시작하여 한국어 CTC fine-tuning.
기존 한국어 모델(Kkonjeong/wav2vec2-base-korean)은 T527 NPU uint8 양자화 시 CER 100% 실패.
영어 모델의 sharp logit 특성을 활용하여 uint8 양자화 생존하는 한국어 모델 학습.

---

## 배경: 왜 영어 모델에서 시작하는가

### 기존 한국어 모델 실패 원인

| | 영어 base-960h | 한국어 base-korean |
|---|---|---|
| Logit std | **8.39** | 1.95 |
| Top1-Top2 margin 최소 | **0.34** | 0.005 |
| uint8 step size | 0.08 | 0.05 |
| margin > step? | **✓ (0.34 > 0.08)** | ✗ (0.005 < 0.05) |
| T527 uint8 CER | **17.52%** | 100% |

한국어 모델의 logit margin이 uint8 step size보다 작아 양자화 후 argmax 결과 전부 뒤집힘.

### 전략

영어 모델(sharp logit)에서 시작 → LM head만 교체(32→56 한국어 자모) → 한국어 데이터로 CTC fine-tune → sharp logit 특성 유지하면서 한국어 학습.

---

## 학습 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 4070 Ti Super 16GB |
| Base 모델 | facebook/wav2vec2-base-960h (94.4M params, 12 layers) |
| 데이터 | Zeroth-Korean (22,263 train / 457 test, ~50시간) |
| Vocab | 56 한국어 자모 (19 초성 + 21 중성 + 11 종성 + 5 특수) |
| Loss | CTC |
| FP16 | ✓ |
| Effective batch | 8 × 4 (gradient_accumulation) = 32 |

---

## Attempt 1 — LR=1e-4, CNN freeze

| 항목 | 값 |
|------|-----|
| Epochs | 30 |
| LR | 1e-4 |
| Frozen | CNN feature encoder |
| Trainable | 90.2M / 94.4M |
| 결과 | **WER 100% — CTC 수렴 실패** |

CTC가 blank 출력에 고착. LR이 너무 높아 LM head(랜덤 초기화)가 불안정.
Loss: 10.66 → 3.37 (하강했으나 WER 개선 없음).

---

## Attempt 2 — LR=3e-5, Encoder L0-5 freeze

| 항목 | 값 |
|------|-----|
| Epochs | 50 |
| LR | 3e-5 |
| Warmup | 2000 steps |
| Frozen | CNN + Encoder layers 0-5 |
| Trainable | 47.7M / 94.4M |
| 학습 시간 | ~1시간 40분 |
| 결과 | **WER 54.18%** |
| Logit margin min | 0.0196 |

**핵심 변경**: LR을 3배 낮추고, warmup을 4배 늘리고, encoder 전반부(L0-5) freeze.

WER 하강 히스토리:
```
epoch  1  → WER 100%
epoch  3  → WER 99.02%  (blank 탈출 시작!)
epoch  5  → WER 80.20%
epoch 10  → WER 70.98%
epoch 20  → WER 61.39%
epoch 30  → WER 56.72%
epoch 50  → WER 54.18%
```

Loss: 10.66 → 3.48 → **1.35** → 0.56 (급격 하강).
Epoch 2~3에서 blank 탈출 — warmup 2000 steps가 핵심.

---

## Attempt 3 — LR=1e-5, 전체 unfreeze, cosine decay

| 항목 | 값 |
|------|-----|
| 시작점 | Attempt 2 checkpoint |
| Epochs | +50 (총 100) |
| LR | 1e-5 |
| Scheduler | Cosine |
| Frozen | CNN only (encoder 전체 unfreeze) |
| Trainable | 90.2M |
| 학습 시간 | ~1시간 40분 |
| 결과 | **WER 44.04%** |
| Logit margin min | 0.0018 |

```
epoch  5  → WER 50.73%
epoch 10  → WER 49.15%
epoch 20  → WER 46.45%
epoch 35  → WER 44.04%  (best)
epoch 50  → WER 44.48%  (약간 상승)
```

### T527 NPU uint8 테스트 (최초!)

ONNX export → Acuity KL uint8 → 72MB NB → T527 vpm_run:

```
ko_test_0001: blank=123/149 | "ㅂㅏㄹㅏㅁ ㅁㅔㄱㄱㅔ ㅇㅏㄹㄴㅔㄴㅇㅣ ㅁ"
ko_test_0002: blank=128/149 | "ㄴㅏㄴㅐ ㅇㅣㄷㅇㅡㄹ ㅋㅡㄹ"
ko_test_0003: blank=131/149 | "ㄱㅡㄹㅣ ㄱㅜ ㅇㅓㄱㅣ ㄴㅏㄴ"
```

**이전 CER 100% (전부 garbage) → 한국어 자모가 출력됨!**

---

## Attempt 4 — LR=5e-6, weight decay

| 항목 | 값 |
|------|-----|
| 시작점 | Attempt 3 checkpoint |
| Epochs | +30 (총 130) |
| LR | 5e-6 |
| Weight decay | 0.01 |
| 학습 시간 | ~1시간 |
| 결과 | **WER 42.06%** |
| Logit margin min | 0.0110 |

```
epoch  5  → WER 43.76%
epoch 15  → WER 42.70%
epoch 30  → WER 42.06%
```

### T527 NPU uint8 테스트

```
ko_test_0001: blank=127/149 | 바럼 말ㄷ ㅏㄹ내니 무
ko_test_0002: blank=125/149 | 나네 읻을 클ㄹㅆㅆ
ko_test_0005: blank=130/149 | 막끼녀 뚄 안
```

---

## Attempt 5 — LR=2e-6

| 항목 | 값 |
|------|-----|
| 시작점 | Attempt 4 checkpoint |
| Epochs | +50 (총 180) |
| LR | 2e-6 |
| 학습 시간 | ~1시간 40분 |
| 결과 | **WER 40.60%** |
| Logit margin min | **0.0365** |

```
epoch 10  → WER 41.09%
epoch 25  → WER 40.85%
epoch 40  → WER 40.72%
epoch 50  → WER 40.60%
```

### T527 NPU uint8 테스트 (10샘플)

| 샘플 | NPU 출력 (음절) | GT |
|------|-----------------|-----|
| ko_test_0000 | 묵도 굗은 가ㅏㅏㅏ | 몬터규는 자녀들이 사랑을... |
| ko_test_0001 | 가럼 며ㄸㄹ네니 ㅁ | 차 문이 종잇장처럼 얇지... |
| ko_test_0003 | **그리구 어기 나** | **그리고 이 나무는...** |
| ko_test_0004 | 각도 오던 아오 ㅂ그 | 평소 오전 아홉 시에서... |
| ko_test_0006 | **야구워된 여당 ㅁㅂ아** | **야권은 여당에서...** |
| ko_test_0007 | 국기들 보호ㅏㄱㄱㅇ | 독일을 보호하기 위하여... |
| ko_test_0008 | 멘틍 황은ㄱㅇ | 젠슨 황은 엔비디아에서... |
| ko_test_0009 | **이벋에 발꼋 뗐다다** | **이번에 발견된 화석은...** |

발음이 유사한 출력 다수 (야구워된↔야권은, 이벋에↔이번에, 그리구↔그리고).

---

## Attempt 6 — LR=1e-6, 100 epochs (진행 중)

| 항목 | 값 |
|------|-----|
| 시작점 | Attempt 5 checkpoint |
| Epochs | +100 (총 280) |
| LR | 1e-6 |
| Scheduler | Cosine |
| 현재 진행 | epoch 75/100 |
| 현재 WER | **39.50%** (best) |

```
epoch 10  → WER 40.01%
epoch 20  → WER 39.65%
epoch 40  → WER 39.65%
epoch 60  → WER 39.53%
epoch 72  → WER 39.50%  (best)
```

수렴 중. 100 epoch 완료 후 attempt7 (LR=5e-7) 자동 시작 예정.

---

## WER 추이 전체

```
WER(%)
100 ┤ ■ attempt1 (LR=1e-4, 실패)
    │
 55 ┤     ■ attempt2 (LR=3e-5)  ← blank 탈출!
    │
 44 ┤         ■ attempt3 (LR=1e-5)
 42 ┤           ■ attempt4 (LR=5e-6)
 41 ┤             ■ attempt5 (LR=2e-6)
 40 ┤               ■ attempt6 (LR=1e-6)  ← 현재
 39 ┤
    └──┬──┬──┬──┬──┬──→ attempt
       1  2  3  4  5  6
```

---

## Logit Margin 추이

| Attempt | WER | Logit std | Margin min | Margin mean | uint8 생존 |
|---------|-----|-----------|------------|-------------|-----------|
| 기존 base-korean | 100% | 1.95 | 0.0050 | 3.08 | ✗ |
| 2 | 54.18% | 2.55 | 0.0196 | 5.28 | ✗ |
| 3 | 44.04% | 4.03 | 0.0018 | 6.28 | ✗ |
| 4 | 42.06% | — | 0.0110 | 6.48 | ✗ |
| 5 | 40.60% | — | **0.0365** | 6.60 | ✗ (0.03 < 0.12) |
| 영어 base-960h | 17.52% | 8.39 | **0.3400** | 10.85 | **✓** |

Margin min이 개선 중이지만 uint8 step size (0.12) 대비 아직 부족.
→ **QAT (Quantization-Aware Training)** 필요.

---

## ONNX → NB 파이프라인

```bash
# 1. ONNX export
python3 -c "
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained('output/wav2vec2-ko-attemptN', attn_implementation='eager')
model.eval()
import torch
torch.onnx.export(model, torch.zeros(1,48000), 'output/model.onnx',
    input_names=['input_values'], output_names=['logits'], opset_version=12)
"

# 2. Acuity import + uint8 quantize + NB export (Docker)
docker run --rm -v $WORK:/work -v $ACUITY:/acuity:ro -v $VIVANTE:/vivante:ro t527-npu:v1.2 bash -c "
cd /acuity/bin
./pegasus import onnx --model /work/model.onnx --output-model /work/m.json --output-data /work/m.data
./pegasus quantize --model /work/m.json --model-data /work/m.data --device CPU \
  --with-input-meta /work/inputmeta.yml --model-quantize /work/m_kl.quantize \
  --quantizer asymmetric_affine --qtype uint8 --rebuild-all --algorithm kl_divergence
./pegasus export ovxlib --model /work/m.json --model-data /work/m.data \
  --dtype quantized --model-quantize /work/m_kl.quantize --with-input-meta /work/inputmeta.yml \
  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk \$VSIM --target-ide-project linux64 --batch-size 1 --output-path /work/nb/
"

# 3. T527 테스트 (vpm_run)
adb push nb/network_binary.nb /data/local/tmp/test/
adb shell "cd /data/local/tmp/test && LD_LIBRARY_PATH=/vendor/lib64 ./vpm_run_aarch64 -s sample.txt -b 0"
```

---

## 다음 단계

### 1. QAT (Quantization-Aware Training)

현재 FP32 학습 → 사후 uint8 양자화. QAT로 전환하면:
- 학습 중 fake quantize 시뮬레이션
- 모델이 uint8 step size에 맞게 logit margin을 키움
- margin min 0.0365 → 0.12+ 목표

### 2. 더 많은 데이터

현재 Zeroth-Korean 50시간. NAS 4356시간으로 학습 시 WER 대폭 개선 예상.

### 3. 입력 길이 확장

현재 3초(48000 samples). 5초로 확장하면 긴 발화 처리 가능.

---

## 파일 위치

| 파일 | 경로 |
|------|------|
| 영어 base weights | `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec_original/` |
| 학습 스크립트 | `ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/finetune/` |
| Zeroth-Korean data | `/home/nsbb/.cache/zeroth_korean/data/` (parquet 7개, 2.7GB) |
| Attempt 5 모델 | `finetune/output/wav2vec2-ko-attempt5/` |
| Attempt 5 ONNX | `finetune/output/wav2vec2_ko_attempt5_sim.onnx` (378MB) |
| Attempt 5 NB | `finetune/output/a5_nbg_nbg_unify/network_binary.nb` (72MB) |
| Attempt 6 모델 | `finetune/output/wav2vec2-ko-attempt6/` (학습 중) |

---

## Attempt 6 최종 결과 (100 epochs 완료)

| 항목 | 값 |
|------|-----|
| 시작점 | Attempt 5 checkpoint |
| Epochs | +100 (총 280) |
| LR | 1e-6 → 0 (cosine) |
| 학습 시간 | 6시간 13분 |
| **최종 WER** | **39.38%** |
| Logit std | 4.92 |
| Margin min | 0.0245 |
| Margin mean | 6.74 |
| Train loss | 0.2451 |

### FP32 디코딩 결과 (ko_test_0001)

```
GT:  차 문이 종잇장처럼 얇지 않으니 문 두께 스물 센티미터를 빼면 실제 승 하차 여유 공간은 스물 센티미터라는 계산이 나옵니다
FP32: 탈럼 야지 안으니 두문 두께 쓰물 센티미터를 빼메슬쩨 쓰짜 영의 공간에 스물 센티미터라네 기사
```

"스물 센티미터를", "공간", "센티미터라", "안으니", "두께" 등 핵심 단어 다수 일치.

### WER 수렴 히스토리

```
epoch 10  → WER 40.01%
epoch 30  → WER 39.90%
epoch 50  → WER 39.72%
epoch 72  → WER 39.50%
epoch 82  → WER 39.38% (best)
epoch 100 → WER 39.38% (수렴)
```

### Attempt 7 자동 시작됨

attempt6 완료 후 auto_continue.sh에 의해 attempt7 (LR=5e-7) 자동 시작.

---

## QAT (Quantization-Aware Training) 결과

### 설정

- attempt7 (WER 39.23%) 모델에서 시작
- FakeQuantize 50개 삽입 (encoder attention out_proj + lm_head)
- uint8 asymmetric affine 시뮬레이션
- 30 epochs, LR=1e-5, FP32 (fp16 비활성)

### WER 추이

```
epoch  1  WER 0.4412  (FakeQuantize 추가 직후 하락)
epoch 10  WER 0.4267
epoch 20  WER 0.4203
epoch 22  WER 0.4173  (best)
epoch 30  WER 0.4245  (overfitting)
```

### Logit Margin 비교

| 모델 | Margin mean | Margin min | WER |
|------|-------------|------------|-----|
| attempt7 (PTQ) | 6.74 | 0.0245 | 39.23% |
| **QAT** | **8.96** (+33%) | **0.0006** (악화) | 41.73% |
| 영어 base-960h | 10.85 | 0.34 | 17.52% |

### T527 NPU uint8 결과

| 샘플 | attempt6 NPU | QAT NPU |
|------|-------------|---------|
| ko_test_0003 | 그리구어기 나 | 그리구 어기 난 |
| ko_test_0006 | 약구워된 여단 | 야궈든 여당 |
| ko_test_0009 | 이 벋에 발껼께 | 이벋에 발껼 깨아 |

**결과 거의 동일.** QAT가 margin mean을 키웠지만, margin min은 오히려 악화. uint8 실제 결과에서 의미있는 개선 없음.

### 분석

QAT의 FakeQuantize가 대부분의 프레임에서 margin을 키웠으나 (mean 33% 개선), 소수의 경계 프레임에서 오히려 불안정해짐 (min 악화). T527 uint8 양자화는 min margin에 의해 결정되므로 개선 효과 없음.

### 더 나은 QAT 접근 필요

1. **더 많은 FakeQuantize 위치** — 현재 attention out_proj + lm_head만. 모든 Linear/Conv에 삽입
2. **NNCF 자동 QAT** — 수동 삽입 대신 NNCF가 최적 위치에 자동 삽입
3. **더 많은 데이터** — 50시간으로는 QAT 효과 제한적
4. **Margin min 최적화 loss** — CTC loss + margin penalty 추가
