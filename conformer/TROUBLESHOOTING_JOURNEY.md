# Conformer T527 NPU 배포 — 시행착오 전체 기록

"처음에 안 되던 모델을 어떻게 고쳐서 CER 10%까지 만들었는가"

---

## 결론 먼저

HuggingFace에서 한국어 Conformer CTC 모델을 받아서 T527 NPU에 올리려 했으나, **바로는 안 됐음.** 3차례 시도와 수십 번의 디버깅 끝에 여러 문제를 해결하고 최종 CER 10.02%를 달성.

| 시도 | 모델 | 결과 | 원인 |
|------|------|------|------|
| 1차 | EN Conformer small | FP32 garbage | graph surgery가 모델 파괴 |
| 2차 | SpeechBrain KO Conformer | NB export 실패 | 모델 너무 큼 (error 64768) |
| 3차 | cwwojin KO Conformer | garbage → **해결** | vocab 매핑 오류 + mel 전처리 차이 |
| 최종 | SungBeom KO Conformer | **CER 10.02%** | cwwojin 파이프라인 그대로 적용 |

---

## 1차 시도: 영어 Conformer small — graph surgery 실패

### 배경

NeMo `stt_en_conformer_ctc_small` (13.2M params). ONNX export 후 Acuity에서 import가 안 됨.

### 문제

Conformer의 relative positional encoding이 만드는 ONNX op:
- **Where op 48개** — attention mask 적용
- **Pad op 16개** — 상대 위치 행렬 생성 시 padding, constant_value가 빈 텐서

Acuity 6.12가 Pad의 빈 constant_value를 처리 못 하고 `IndexError: list index out of range`.

### 시도한 것

1. Where 48개 → Identity op으로 교체 (attention mask를 상수 true로 가정)
2. Pad 16개 → constant_value를 명시적 0.0으로 수정
3. onnxsim으로 3982 → 1647 nodes

### 결과

Acuity import 성공 → **NB 14MB, 74ms 추론.** 하지만 영어 WAV 넣어서 디코딩하니 **FP32에서 이미 garbage.**

```
GT:  CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS
NPU: erse uh s foring answer in and that theationsforver the to a carments
```

### 원인

Where op을 Identity로 바꾸면서 attention 연산이 깨짐. Where는 단순 mask가 아니라 relative position encoding의 핵심 연산이었음. graph surgery로 노드를 건드리면 **ONNX shape inference가 깨져서** 후속 연산이 전부 잘못된 tensor shape으로 계산됨.

### 교훈

**ONNX graph surgery는 위험. 가능하면 원본 export를 최대한 유지해야 함.**

---

## 2차 시도: SpeechBrain 한국어 Conformer — NB export 실패

### 배경

`speechbrain/asr-conformer-transformerlm-ksponspeech` (42.9M params, vocab 5000).

### 결과

```
ONNX: 134MB, 1438 nodes
Acuity import: 성공
Acuity quantize (KL uint8): 성공
Acuity export NB: error 64768 (실패)
```

### 원인

T527 NPU가 이 크기/구조의 모델을 NB로 컴파일 못 함. error 64768은 Acuity의 NPU 컴파일러가 모델을 NPU instruction으로 변환할 때 내부 한계 초과.

### 교훈

**Acuity import/quantize가 성공해도 NB export에서 실패할 수 있음.** 모델 크기만이 아니라 그래프 구조 (node 수, op 종류)도 영향.

---

## 3차 시도: cwwojin 한국어 Conformer — 삽질 끝에 성공

### 배경

`cwwojin/stt_kr_conformer_ctc_medium` (31.8M params, vocab 5001).

### 3-1. ONNX export

1차 시도의 교훈을 바탕으로 **graph surgery 하지 않고** NeMo `model.export()` 그대로 사용.

```python
# NeMo Docker에서
model.export("model.onnx")  # opset 16, dynamic shape, 4462 nodes
```

static shape 고정 + onnxsim + Pad fix만 수행:
- input shape: `[B, 80, T]` → `[1, 80, 301]`
- length input → 상수 301로 대체
- onnxsim: 4462 → 1905 nodes
- **Pad op 18개의 빈 constant_value → 0.0** (이것만 수정)
- **Where op 54개는 건드리지 않음** (Acuity가 알아서 처리)

### 3-2. Acuity 변환 성공

```
import: Error(0), Warning(1) — 성공
quantize (uint8 KL): Error(0), Warning(0) — 성공
export NB: Error(0), Warning(0) — 성공! (29MB)
```

### 3-3. T527 NPU 테스트 — garbage 출력

NB를 T527에 올려서 vpm_run → 한국어 토큰이 나오긴 하는데 의미없는 내용:

```
GT:  평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다
NPU: 남았없##번장난그거했없
```

### 3-4. FP32 ONNX도 garbage — 양자화 문제 아님

양자화 전 FP32 ONNX를 onnxruntime으로 돌려도 garbage → **양자화가 아니라 ONNX 자체 또는 입력이 문제.**

### 3-5. mel 전처리 의심

calibration mel을 librosa로 만들었는데, NeMo mel과 비교:

```
NeMo mel range:    [-1.75, 5.14], mean=0.0000
librosa mel range: [-2.24, 4.89], mean=-0.0048
```

다르다! → NeMo Docker에서 NeMo preprocessor로 mel 생성.

**NeMo mel 넣어도 여전히 garbage.** → mel 문제 아님.

### 3-6. 모델 자체가 문제인가? → transcribe()는 정상

```python
model.transcribe(["test.wav"])
# → "평소 오전 아홉 시에서 오후 일곱 시까지 일하면 하루 이만 원 정도를 번다"
```

NeMo 네이티브 추론은 완벽. ONNX만 문제.

### 3-7. RelPos attention 패치 시도 — 실패

Conformer의 `rel_shift()` 함수가 ONNX에서 잘못 변환된다고 가정하고, `torch.gather`로 pre-computed index 방식으로 패치:

```python
# rel_shift: pad(left=1) → reshape → drop first row
# 이걸 gather index로 대체
indices[i, j] = n - 1 - i + j  # 처음에 i + n - 1 - j로 잘못 계산 → 수정
```

패치 후 PyTorch와 ONNX 결과가 일치 (둘 다 같은 출력) → **ONNX export는 정확.** 근데 둘 다 garbage.

### 3-8. hook으로 단계별 비교 — 핵심 발견

transcribe()와 수동 forward()의 **모든 중간 결과를 hook으로 비교:**

| 비교 대상 | diff |
|----------|------|
| encoder 입력 (mel) | **0.000000** (완전 동일) |
| encoder 출력 | **0.000000** (완전 동일) |
| decoder 출력 (log_probs) | **0.000000** (완전 동일) |

**log_probs가 완전히 동일한데 디코딩 결과만 다름.**

이 순간 깨달음: **ONNX 모델은 처음부터 정상이었음. 디코딩이 잘못된 것.**

### 3-9. 진짜 원인: vocab.txt ≠ tokenizer ID

.nemo 파일 안에 들어있는 `vocab.txt`:
```
##s
the
a
##t
to
...
```

이건 SentencePiece vocab 파일의 내부 표현이고, **실제 tokenizer의 ID 매핑과 순서가 다름.**

```python
# 잘못된 방법 (vocab.txt 순서):
vocab[1] = "the"      # → garbage

# 올바른 방법 (tokenizer API):
tok.ids_to_tokens([1]) = ["▁그"]  # → 정상
```

**해결:** `tok.ids_to_tokens()`로 올바른 vocab 매핑 추출 → `vocab_correct.json` 생성.

### 3-10. 추가 발견: dither와 pad_to

`model.transcribe()` 소스 코드를 뜯어보니:

```python
# transcribe() 내부에서 자동으로 설정
self.preprocessor.featurizer.dither = 0.0   # 학습 시 1e-5 → 추론 시 0
self.preprocessor.featurizer.pad_to = 0     # mel frame padding 제거
```

이걸 안 하면 mel이 미세하게 달라짐. calibration mel 생성할 때도 이 설정 필수.

---

## 최종 파이프라인 (이 모든 삽질의 결과)

```
1. NeMo Docker에서 model.export() → ONNX (graph surgery 하지 않음!)
2. NeMo preprocessor(dither=0, pad_to=0)로 mel 생성
3. tokenizer.ids_to_tokens()로 올바른 vocab 추출
4. ONNX static shape 고정 + onnxsim
5. Pad op 18개만 수정 (Where 54개는 건드리지 않음)
6. Acuity import → quantize → export NB
7. T527 vpm_run → CER 10.02%
```

---

## 발견한 문제 요약

| # | 문제 | 증상 | 원인 | 해결 |
|---|------|------|------|------|
| 1 | graph surgery | FP32 garbage | Where/Identity 교체로 shape 깨짐 | 하지 않음 |
| 2 | NB export 실패 | error 64768 | 모델 구조가 NPU 컴파일러 한계 초과 | 더 작은 모델 사용 |
| 3 | vocab 매핑 | 디코딩 garbage | vocab.txt ≠ tokenizer ID 순서 | tok.ids_to_tokens() 사용 |
| 4 | mel 전처리 | FP32 garbage (librosa 사용 시) | librosa mel ≠ NeMo mel | NeMo preprocessor 사용 |
| 5 | dither/pad_to | mel 미세 차이 | 추론 시 dither=0, pad_to=0 필수 | transcribe() 소스 분석 |
| 6 | Pad op | Acuity import 실패 | constant_value 빈 텐서 | 명시적 0.0 설정 |
| 7 | gather index | 패치 후 garbage | i+n-1-j (잘못) → n-1-i+j (정확) | rel_shift 수학 분석 |

---

## 핵심 교훈 3가지

### 1. ONNX graph surgery는 최후의 수단

노드를 수동으로 제거/교체하면 shape inference가 깨짐. 가능하면 원본 export를 유지하고 Acuity가 자체 처리하게 맡기는 것이 안전. Pad의 빈 constant_value 같은 최소한의 수정만.

### 2. 모델이 garbage를 출력할 때, 모델 자체를 의심하기 전에 전처리/후처리를 먼저 확인

이번 경우 ONNX 모델은 처음부터 정상이었고, 문제는:
- vocab 매핑 (디코딩)
- mel 전처리 (입력)
- dither/pad_to 설정

모델 내부를 뜯어고치기(RelPos 패치 등) 전에 입출력부터 검증했어야 했음.

### 3. transcribe()의 소스 코드를 읽어라

NeMo의 `model.transcribe()`는 단순히 `forward()`를 호출하는 게 아님. 내부에서 dither=0, pad_to=0 설정, DataLoader 사용, 디코딩 전략 등 추가 작업을 함. 이걸 모르고 수동으로 forward() 호출하면 다른 결과가 나옴.
