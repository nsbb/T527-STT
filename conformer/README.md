# Korean Conformer CTC — T527 NPU uint8 테스트 결과

## 모델: cwwojin/stt_kr_conformer_ctc_medium

| 항목 | 값 |
|------|-----|
| HuggingFace | [cwwojin/stt_kr_conformer_ctc_medium](https://huggingface.co/cwwojin/stt_kr_conformer_ctc_medium) |
| 프레임워크 | NeMo (Conformer CTC + BPE) |
| 파라미터 | 31.8M (18 layers, d_model=256, 4 heads) |
| Vocab | 5001 (한국어 BPE 5000 + blank) |
| 학습 데이터 | AI Hub 한국어 |
| 입력 | mel spectrogram [1, 80, 301] (3초, 16kHz) |
| 출력 | log probs [1, 76, 5001] |

## T527 NPU 결과

| 항목 | 값 |
|------|-----|
| NB 크기 | **29MB** |
| 추론 시간 | **111ms** |
| 양자화 | uint8 asymmetric_affine KL divergence |
| CER (3초 vs 전체 GT) | 88.31% |

### 비교

| 모델 | NB | 추론 | CER | 비고 |
|------|-----|------|-----|------|
| **KR Conformer CTC** | **29MB** | **111ms** | **88.31%*** | 한국어 출력 동작 |
| KoCitrinet 300f | 62MB | 120ms | 44.44% | 운용중 |
| Wav2Vec2 KO fine-tune | 72MB | ~400ms | ~100% | 부분 동작 |
| NAS 80k base-korean | 77MB | 424ms | ~100% | 부분 동작 |

*3초 잘린 입력 vs 전체 GT(8~20초) 비교라서 높게 나옴. 실제 3초 구간만 비교하면 훨씬 낮을 것.

### 샘플 결과 (20개)

| # | GT (처음 50자) | NPU | CER |
|---|---|---|---|
| 00 | 몬터규는 자녀들이 사랑을 제대로 못 받고... | 뭔⁇ 그는 사⁇들이 사 | 90.0% |
| 01 | 차 문이 종잇장처럼 얇지 않으니... | ⁇처럼 야지⁇ 아니니니 조 문 두게 | 92.0% |
| 02 | 지난해 이들 크루즈관광객의... | 지나⁇는 이들⁇르주 관광장 계에 | 80.0% |
| 03 | 그리고 이 나무는 태즈메이니아... | ⁇ 이 나는 테지즈에 | 91.9% |
| 04 | 평소 오전 아홉 시 에서... | 편수 오전. 아홉 시에서 | 75.9% |
| 05 | 박준영 돈 안 되는 공익사건... | 아 그냥 형 돈 안 되는 공익⁇ | 90.6% |
| 06 | 야권은 여당에서 분명한 입장을... | ⁇⁇는 여⁇⁇ 분명⁇ 입자 | 87.1% |
| 07 | 독일을 보호하기 위하여... | ⁇기를⁇ 보호⁇기 위하요 | 89.6% |
| 08 | 젠슨 황은 엔비디아에서... | 제⁇슨학학은 V디⁇ 이 | 90.0% |
| 09 | 이번에 발견된 화석은... | ⁇ 발견 된 화석은 에 | 86.0% |
| 10 | 몽둥이와 함께 변 씨의... | 목둥 이 함께⁇시에 가 | 83.9% |
| 11 | 하지만 싫증 나면 버리면... | ⁇지만 말 실지 나면면⁇리면면 | 88.4% |
| 12 | 업계 관계자는 가격이 오르면... | 어깨 없지 관계기 자인 가격이이 오 | 93.4% |
| 13 | 지난달 사 일 고서점가인... | 지는들 사 일 일 고⁇⁇ | 91.5% |
| 14 | 경기 성남중원은 새누리당에서... | 변기⁇⁇ 중⁇ 세누이⁇ | 90.0% |
| 15 | 이날 박창진 사무장은... | 이⁇ 막 잘진인⁇모⁇은 | 93.2% |
| 16 | 국내 경기가 여전히 침체된... | 국내 내⁇기가 여⁇이 침체 내 | 84.4% |
| 17 | 애플 입장에선 이천 십 이 년... | 에⁇⁇ 예 이찬 | 94.4% |
| 18 | 임화섭 특파원 미국 내브래스카주... | 이 말⁇ 특 파 원 미국 내 | 85.7% |
| 19 | 지금은 대통령한테 줄을 대서... | 지금은 대통 형한테 줄을 돼서 | 75.0% |

### 주목할 출력

- `[04]` "편수 오전. 아홉 시에서" — GT "평소 오전 아홉 시 에서" (거의 정확)
- `[05]` "돈 안 되는 공익" — GT "돈 안 되는 공익사건" (핵심 키워드 정확)
- `[19]` "지금은 대통 형한테 줄을 돼서" — GT "지금은 대통령한테 줄을 대서" (거의 정확)
- `[09]` "발견 된 화석은" — GT "이번에 발견된 화석은" (핵심 단어 정확)

## ONNX export 해결 과정

### 문제

NeMo Conformer의 **RelPositionMultiHeadAttention**이 ONNX export 시 깨지는 현상:
- `rel_shift()`: pad(left=1) → reshape → drop first row → 이 조합이 ONNX에서 Where/Pad op으로 변환
- NeMo `model.export()`로 생성한 ONNX는 FP32에서도 garbage

### 원인 발견

`m.transcribe()`는 정상이지만 `m.forward()`는 garbage → 둘의 log_probs가 **완전 동일** (diff=0) → **디코딩 vocab 매핑 오류**였음!

- `vocab.txt` 파일 순서 ≠ SentencePiece tokenizer ID 순서
- `tok.ids_to_text()` 사용 시 정확한 결과
- 또한 `transcribe()`가 `dither=0.0`, `pad_to=0` 설정하는 것을 발견

### 해결

1. NeMo Docker에서 `model.export()` (opset 16, dynamic shape)
2. static shape [1, 80, 301] 고정 + length 상수화
3. onnxsim 으로 4462 → 1905 nodes
4. Pad op 18개 empty constant_value → 0.0 수정
5. NeMo preprocessor(dither=0, pad_to=0)로 calibration mel 생성
6. Acuity KL uint8 양자화 → **29MB NB**

### mel 전처리 주의

librosa mel ≠ NeMo mel! **반드시 NeMo preprocessor로 mel 생성해야 함.**
- NeMo mel range: [-1.75, 5.14]
- librosa mel range: [-2.24, 4.89]
- 두 결과의 차이가 모델 정확도에 치명적

## 파일 구조

```
conformer/
├── kr_cwwojin/
│   ├── stt_kr_conformer_ctc_medium.nemo    # 원본 모델 (123MB)
│   ├── model_nemo_export.onnx              # NeMo export (dynamic)
│   ├── model_acuity_v2.onnx                # static [1,80,301] + Pad fix (126MB)
│   ├── vocab_correct.json                  # 올바른 tokenizer ID→token 매핑
│   ├── nemo_calib/                         # NeMo preprocessor로 생성한 mel (20개)
│   ├── wksp_v2_nbg_unify/
│   │   ├── network_binary.nb              # T527 NPU NB (29MB)
│   │   └── nbg_meta.json
│   └── test_outputs_v2/                    # T527 NPU output .dat (20개)
├── kr_cwwojin_t527_results.csv             # 20샘플 결과 CSV
└── README.md                               # 이 문서
```

## 다음 단계

1. **full-length 테스트** — 슬라이딩 윈도우로 전체 음성 처리 시 CER 개선 예상
2. **SungBeom 모델 (122.5M)** — 더 큰 모델, NB 크기가 T527 한계 내인지 확인
3. **KoCitrinet과 앙상블** — Conformer + KoCitrinet 결합
4. **Android 앱 통합** — mel 전처리 + NB 추론 파이프라인
