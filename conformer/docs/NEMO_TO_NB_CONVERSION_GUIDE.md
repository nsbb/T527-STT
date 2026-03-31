# .nemo → ONNX → NB 변환 가이드

**최종 업데이트:** 2026-03-31
**대상 모델:** SungBeom Conformer CTC Medium (122.5M params)
**대상 칩:** T527 Vivante NPU (VIP9000)

---

## 요약

```
.nemo → [NeMo Docker] → ONNX → [WSL 로컬] → acuity ONNX → [Docker + Acuity 6.12] → NB
```

**핵심 규칙:**
1. ONNX export는 **NeMo Docker**에서만
2. Acuity import/quantize/export는 **Docker(t527-npu:v1.2) + Acuity 6.12 binary + LD_LIBRARY_PATH**
3. NB export 시 optimize 플래그는 **`VIP9000NANOSI_PLUS_PID0X10000016`**

---

## 경로

| 항목 | 경로 |
|------|------|
| NeMo Docker | `nvcr.io/nvidia/nemo:23.06` |
| t527-npu Docker | `t527-npu:v1.2` |
| Acuity 6.12 binary | `/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0` |
| VivanteIDE 5.7.2 | `/home/nsbb/VeriSilicon/VivanteIDE5.7.2` |
| fix 스크립트 | `t527-stt/conformer/scripts/fix_onnx_for_acuity.py` |
| Conformer 작업 디렉토리 | `ai-sdk/models/conformer/kr_sungbeom/` |
| Calibration 데이터 | `ai-sdk/models/conformer/kr_sungbeom/calib/calib_*.npy` (10개) |
| inputmeta LID | `audio_signal_1507` (Conformer 고정) |

---

## Step 1: .nemo → ONNX

**반드시 NeMo Docker 사용. 로컬 NeMo(pip)로 하면 Acuity import 실패.**

```bash
WORK=<.nemo 파일이 있는 디렉토리 절대경로>

docker run --rm \
  -v $WORK:/workspace/model \
  nvcr.io/nvidia/nemo:23.06 \
  python3 -c "
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecCTCModelBPE.restore_from('/workspace/model/<MODEL>.nemo', map_location='cpu')
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0
    m.export('/workspace/model/<OUTPUT>.onnx')
  "
```

**`dither=0.0`, `pad_to=0` 필수** — 없으면 mel 전처리 불일치.

출력: `<OUTPUT>.onnx` (488MB, opset 16, 4462 nodes, dynamic shape)

---

## Step 2: ONNX fix for Acuity

WSL 로컬에서 실행.

```bash
python3 t527-stt/conformer/scripts/fix_onnx_for_acuity.py \
    <INPUT>.onnx <OUTPUT>_acuity.onnx --frames 301
```

수행 내용:
1. Dynamic shape → static `[1, 80, 301]`
2. `length` input → 상수 301로 교체
3. onnxsim: 4462 → 1905 nodes
4. Pad op 18개: 빈 constant_value → 명시적 0.0
5. Where op 54개: 건드리지 않음 (Acuity가 자체 처리)

출력: `<OUTPUT>_acuity.onnx` (455~477MB, 1905 nodes, static shape)

---

## Step 3: Acuity import

**⚠️ 반드시 Docker 안에서 + LD_LIBRARY_PATH 설정!**

```bash
WORK=/home/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0

docker run --rm \
  -v $WORK:/work \
  -v $ACUITY612:/acuity612:ro \
  t527-npu:v1.2 \
  bash -c "
    PEG=/acuity612/bin/pegasus
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    cd /work/<SUBDIR>
    \$PEG import onnx \
      --model <ACUITY_ONNX>.onnx \
      --output-model <NAME>.json \
      --output-data <NAME>.data
  "
```

성공 시: `Error(0), Warning(1)` + `<NAME>.json` + `<NAME>.data` 생성

**LD_LIBRARY_PATH 없으면:**
```
IndexError: list index out of range
```

---

## Step 4: Acuity quantize

```bash
docker run --rm \
  -v $WORK:/work \
  -v $ACUITY612:/acuity612:ro \
  t527-npu:v1.2 \
  bash -c "
    PEG=/acuity612/bin/pegasus
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    cd /work/<SUBDIR>
    \$PEG quantize \
      --model <NAME>.json --model-data <NAME>.data \
      --device CPU --with-input-meta /work/<INPUTMETA>.yml \
      --rebuild-all \
      --model-quantize <NAME>_uint8.quantize \
      --quantizer asymmetric_affine --qtype uint8 \
      --algorithm kl_divergence \
      --batch-size 1
  "
```

### inputmeta.yml 템플릿

```yaml
input_meta:
  databases:
  - path: /work/dataset.txt
    type: TEXT
    ports:
    - lid: audio_signal_1507
      category: frequency
      dtype: float32
      sparse: false
      layout: nchw
      shape:
      - 1
      - 80
      - 301
      fitting: scale
      preprocess:
        reverse_channel: false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm:
          - 0
          - 1
          - 2
      redirect_to_output: false
```

### dataset.txt

```
/work/calib/calib_0000.npy
/work/calib/calib_0001.npy
...
/work/calib/calib_0009.npy
```

calib 파일: mel spectrogram [1, 80, 301] float32 npy (NeMo preprocessor로 생성)

---

## Step 5: NB export

**⚠️ VivanteIDE 환경변수 전부 필요 + optimize 플래그 `VIP9000NANOSI_PLUS_PID0X10000016`**

```bash
VIVANTE57=/home/nsbb/VeriSilicon/VivanteIDE5.7.2

docker run --rm \
  -v $WORK:/work \
  -v $VIVANTE57:/vivante57:ro \
  -v $ACUITY612:/acuity612:ro \
  t527-npu:v1.2 \
  bash -c '
    VSIM=/vivante57/cmdtools/vsimulator
    COMMON=/vivante57/cmdtools/common
    export REAL_GCC=/usr/bin/gcc
    export VIVANTE_VIP_HOME=/vivante57
    export VIVANTE_SDK_DIR=$VSIM
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:$VSIM/lib:$COMMON/lib:$VSIM/lib/x64_linux:$VSIM/lib/x64_linux/vsim:$LD_LIBRARY_PATH
    export EXTRALFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$VSIM/lib -Wl,-rpath,$COMMON/lib -Wl,-rpath,$VSIM/lib/x64_linux -Wl,-rpath,$VSIM/lib/x64_linux/vsim"
    cd /acuity612/bin
    ./pegasus export ovxlib \
      --model /work/<SUBDIR>/<NAME>.json \
      --model-data /work/<SUBDIR>/<NAME>.data \
      --dtype quantized \
      --model-quantize /work/<SUBDIR>/<NAME>_uint8.quantize \
      --with-input-meta /work/<INPUTMETA>.yml \
      --pack-nbg-unify \
      --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
      --viv-sdk $VSIM \
      --target-ide-project linux64 \
      --batch-size 1 \
      --output-path /work/<SUBDIR>/wksp_nbg_unify/
  '
```

출력: `wksp_nbg_unify/network_binary.nb` (~102MB) + `nbg_meta.json`

**주의:** `--output-path`에 `_nbg_unify`를 넣으면 실제 출력은 `<path>_nbg_unify/`로 이중 접미사 됨.

---

## 실패 원인 총정리

| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | import `IndexError: list index out of range` | LD_LIBRARY_PATH 미설정 | Docker에서 `/acuity612/bin/lib` export |
| 2 | import `IndexError: list index out of range` | 로컬 NeMo(pip)로 ONNX export | NeMo Docker(`nvcr.io/nvidia/nemo:23.06`) 사용 |
| 3 | import `IndexError: list index out of range` | Pad op 빈 constant_value | `fix_onnx_for_acuity.py` 적용 |
| 4 | `create network 0 failed` (vpm_run) | optimize 플래그 잘못됨 | `VIP9000NANOSI_PLUS_PID0X10000016` |
| 5 | `Fatal model compilation error: 512` | VivanteIDE 환경변수 누락 | REAL_GCC, VIVANTE_VIP_HOME, EXTRALFLAGS |
| 6 | quantize `doesn't have a valid input meta` | inputmeta 경로/LID 틀림 | LID=`audio_signal_1507`, Docker 내부 경로 `/work/` |
| 7 | NB 크기 다름 (500KB 작음) | optimize 플래그 다름 | PID0X10000016 포함 |

---

## 변환 이력

| 모델 | 날짜 | NB 크기 | 비고 |
|------|------|--------|------|
| PTQ (원본) | 2026-03-25 | 106,729,472 | sb.json/sb.data |
| QAT ailab ep13 | 2026-03-26 | 106,727,936 | 자체 데이터 |
| QAT AIHub 100k final | 2026-03-27 | 106,727,936 | q100k.json/q100k.data |
| QAT AIHub full ep09 | 2026-03-31 | 106,709,248 | full_ep09.json/full_ep09.data |
