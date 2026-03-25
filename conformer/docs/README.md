# Korean Conformer CTC — T527 NPU 한국어 음성인식

## 최종 결과

| 모델 | Params | Vocab | NB | 추론 | CER (100샘플) | 상태 |
|------|--------|-------|-----|------|------|------|
| **SungBeom Conformer** | **122.5M** | **2049 BPE** | **102MB** | **233ms/chunk** | **10.02%** | **최고 정확도** |
| cwwojin Conformer | 31.8M | 5001 BPE | 29MB | 111ms/chunk | 55.10% | 경량 |
| KoCitrinet 300f (비교) | ~10M | 2049 SP | 62MB | 120ms | 44.44% | 기존 운용 |

### SungBeom 양자화 비교

| 양자화 | NB | 추론/chunk | CER |
|--------|-----|----------|-----|
| **uint8 AA KL (권장)** | **102MB** | **233ms** | **10.59%** |
| uint8 AA MA | 102MB | 233ms | 10.79% |
| int16 DFP KL | 200MB | 564ms | 10.18% |

---

## 배경

T527 NPU에서 한국어 STT를 구현하기 위해 여러 모델을 시도:
- **Wav2Vec2**: Transformer 기반 → uint8 양자화 실패 (logit margin 부족)
- **KoCitrinet**: CNN 기반 → CER 44.44% (기존 최고)
- **Conformer**: CNN + Attention 하이브리드 → **3차 시도 끝에 CER 10.02% 달성**

Conformer는 처음에 ONNX export 문제, vocab 매핑 오류, mel 전처리 차이 등 여러 문제가 있었으나 하나씩 해결하여 최종 성공. 상세 과정은 [TROUBLESHOOTING_JOURNEY.md](TROUBLESHOOTING_JOURNEY.md) 참조.

---

## 폴더 구조

```
conformer/
├── docs/
│   ├── README.md                    # 이 문서
│   ├── SUNGBEOM_REPORT.md           # SungBeom 모델 상세 실험 보고서
│   ├── DEPLOYMENT_PARAMS.md         # 앱 배포용 파라미터 전체 정리
│   └── TROUBLESHOOTING_JOURNEY.md   # 시행착오 전체 기록 (3차 시도, 7개 문제)
├── configs/
│   ├── sungbeom_vocab_correct.json  # SungBeom tokenizer ID→token (2049)
│   ├── sungbeom_uint8_kl_nbg_meta.json
│   ├── sungbeom_uint8_ma_nbg_meta.json
│   ├── sungbeom_int16_dfp_nbg_meta.json
│   ├── sungbeom_inputmeta.yml       # Acuity 양자화 입력 설정
│   ├── sungbeom_dataset.txt         # calibration 파일 목록
│   ├── cwwojin_vocab_correct.json   # cwwojin tokenizer ID→token (5001)
│   ├── cwwojin_uint8_kl_nbg_meta.json
│   └── cwwojin_inputmeta.yml
├── scripts/
│   ├── fix_onnx_for_acuity.py       # ONNX → Acuity 호환 변환 (static shape + Pad fix)
│   ├── run_all_quant_test.py        # 3가지 양자화 × 100샘플 테스트
│   ├── patch_relpos.py              # RelPos 패치 시도 (참고용, 불필요였음)
│   ├── patch_relpos_v2.py           # RelPos 패치 v2 (참고용, 불필요였음)
│   ├── test_decode_compare.py       # transcribe() vs forward() 비교
│   └── test_transpose.py            # encoder output transpose 테스트
├── results/
│   ├── kr_sungbeom_uint8_kl_100.csv # SungBeom uint8 KL 100샘플 (CER 10.02%)
│   ├── kr_sungbeom_uint8_ma_100.csv # SungBeom uint8 MA 100샘플 (CER 10.08%)
│   ├── kr_sungbeom_int16_dfp_100.csv # SungBeom int16 DFP 100샘플 (CER 9.59%)
│   ├── kr_cwwojin_t527_full_100.csv # cwwojin 100샘플 sliding (CER 55.10%)
│   ├── kr_cwwojin_t527_full_results.csv # cwwojin 20샘플 sliding
│   └── kr_cwwojin_t527_results.csv  # cwwojin 20샘플 3초 truncated
└── (ai-sdk 쪽 — 대용량, git 미포함)
    kr_sungbeom/
    ├── stt_kr_conformer_ctc_medium.1.nemo  # 원본 모델 (468MB)
    ├── model.onnx                          # NeMo export ONNX
    ├── model_acuity.onnx                   # Acuity용 ONNX (477MB)
    ├── nemo_mel/                           # NeMo mel 100개
    ├── wksp_nbg_unify/network_binary.nb    # uint8 KL NB (102MB)
    ├── wksp_uint8_ma_nbg_unify/            # uint8 MA NB
    └── wksp_int16_nbg_unify/              # int16 DFP NB (200MB)
    kr_cwwojin/
    ├── stt_kr_conformer_ctc_medium.nemo    # 원본 모델 (123MB)
    ├── model_acuity_v2.onnx               # Acuity용 ONNX (126MB)
    └── wksp_v2_nbg_unify/network_binary.nb # uint8 KL NB (29MB)
```

---

## 문서 안내

| 문서 | 내용 |
|------|------|
| [SUNGBEOM_REPORT.md](SUNGBEOM_REPORT.md) | 모델 구조, 양자화 상세, CER 분석, 변환 파이프라인 전과정, 재현 방법 |
| [DEPLOYMENT_PARAMS.md](DEPLOYMENT_PARAMS.md) | 앱 배포에 필요한 모든 파라미터 (양자화, mel 전처리, CTC 디코딩, 슬라이딩 윈도우) |
| [TROUBLESHOOTING_JOURNEY.md](TROUBLESHOOTING_JOURNEY.md) | 1차~3차 시도, 발견한 7개 문제와 해결, 핵심 교훈 |

---

## 빠른 시작 (재현)

### 전제 조건

- NeMo Docker: `nvcr.io/nvidia/nemo:23.06`
- Acuity Docker: `t527-npu:v1.2` + Acuity 6.12.0 + VivanteIDE 5.7.2
- T527 디바이스 (adb + vpm_run_aarch64)

### 파이프라인

```bash
# 1. 모델 다운로드
aria2c -x16 -s16 --file-allocation=none \
  -o model.nemo \
  "https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium/resolve/main/stt_kr_conformer_ctc_medium.nemo"

# 2. NeMo Docker: ONNX export + vocab + mel
docker run --rm -v $WORK:/workspace nvcr.io/nvidia/nemo:23.06 python3 -c "
  import nemo.collections.asr as nemo_asr
  m = nemo_asr.models.EncDecCTCModelBPE.restore_from('model.nemo', map_location='cpu')
  m.eval()
  m.preprocessor.featurizer.dither = 0.0
  m.preprocessor.featurizer.pad_to = 0
  m.export('model.onnx')
  # + vocab 추출, mel 생성 (SUNGBEOM_REPORT.md 참조)
"

# 3. ONNX 변환
python3 scripts/fix_onnx_for_acuity.py model.onnx model_acuity.onnx

# 4. Acuity: import → quantize → export NB
#    (SUNGBEOM_REPORT.md 3-7~3-10 참조)

# 5. T527 테스트
python3 scripts/run_all_quant_test.py
```

---

## 핵심 교훈

1. **ONNX graph surgery 금지** — Where/Pad op을 수동 제거하면 모델 깨짐. Pad fix만 최소 수정.
2. **vocab.txt ≠ tokenizer ID** — NeMo BPE 모델은 `tok.ids_to_tokens()`로 vocab 추출 필수.
3. **librosa mel ≠ NeMo mel** — 반드시 NeMo preprocessor 사용. dither=0, pad_to=0 필수.
4. **Conformer = CNN+Attention** — 순수 Transformer(Wav2Vec2)는 uint8 실패, Conformer는 CNN 덕분에 uint8 동작.
