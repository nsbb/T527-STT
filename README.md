# T527 NPU STT Models

Allwinner T527 NPU (Vivante VIP9000NANOSI_PLUS) 용 음성인식(STT) 모델 모음.
모든 모델은 uint8/int8 양자화되어 `.nb` (Network Binary) 형태로 변환 완료.

## Models

| Model | Language | Architecture | CER | Inference | NB Size | Input |
|-------|----------|-------------|-----|-----------|---------|-------|
| [KoCitrinet](ko_citrinet/) | Korean | 1D Conv + SE (CTC) | 44.44% | 120ms | 62MB | 3s (300f) |
| [Wav2Vec2](wav2vec2/) | English | CNN + Transformer (CTC) | 17.52% | 715ms | 87MB | 5s |
| [Zipformer](zipformer/) | Korean | Zipformer (RNN-T) | TBD | TBD | 68MB | streaming |
| [CitriNet EN](citrinet_en/) | English | 1D Conv + SE (CTC) | TBD | TBD | 7MB | 3s |
| [DeepSpeech2](deepspeech2/) | English | RNN (CTC) | TBD | TBD | 56MB | variable |

## Hardware

- **SoC**: Allwinner T527 (ARM Cortex-A55)
- **NPU**: Vivante VIP9000NANOSI_PLUS (PID 0x10000016)
- **Driver**: VIPLite v0x00010d00
- **NPU Clock**: 696MHz, DRAM 1.2GHz

## Toolchain

- **Acuity Toolkit**: v6.12.0 (`pegasus` CLI)
- **VivanteIDE**: v5.7.2 (vsimulator for NBG generation)
- **Quantization**: uint8 asymmetric_affine, moving_average
- **Export**: `pegasus export ovxlib --pack-nbg-unify`

## NPU API

```c
#include <awnn_lib.h>

awnn_init();
ctx = awnn_create("model.nb");

unsigned char *inputs[1] = {quantized_input};
awnn_set_input_buffers(ctx, inputs);
awnn_run(ctx);
float **outputs = awnn_get_output_buffers(ctx);

awnn_destroy(ctx);
awnn_uninit();
```

## Device Testing (vpm_run)

```bash
adb push network_binary.nb /data/local/tmp/test/
adb push input_0.dat /data/local/tmp/test/
# sample.txt: [network]\n path \n[input]\n path \n[output]\n path
adb shell "cd /data/local/tmp/test && LD_LIBRARY_PATH=/vendor/lib64 ./vpm_run_aarch64 -s sample.txt -b 0"
adb pull /data/local/tmp/test/output_0.dat .
```

## Model Details

### KoCitrinet (Korean)
- **Source**: NVIDIA NeMo NGC → ONNX → uint8 NB
- **Input**: `[1, 80, 1, 300]` int8 mel-spectrogram (scale=0.02096, zp=-37)
- **Output**: `[1, 2049, 1, 38]` int8 → CTC greedy + SentencePiece decode
- **Performance**: CER 44.44% (330 Korean samples), 120ms/frame

### Wav2Vec2 (English)
- **Source**: facebook/wav2vec2-base-960h → ONNX → uint8 NB
- **Input**: `[1, 80000]` uint8 raw waveform (scale=0.002860, zp=137)
- **Output**: `[1, 249, 32]` uint8 → CTC greedy decode (32-class English alphabet)
- **Performance**: CER 17.52%, WER 27.38% (50 LibriSpeech test-clean), 715ms/5s
- **Quantization degradation**: +7.78%p CER vs ONNX FP32

### Zipformer (Korean)
- **Source**: sherpa-onnx-streaming-zipformer-korean-2024-06-16 → uint8 NB
- **Components**: Encoder (63MB) + Decoder (2.8MB) + Joiner (1.9MB)
- **Status**: NB conversion complete, device testing pending

### CitriNet EN (English)
- **Source**: NVIDIA NeMo CitriNet → ONNX → uint8 NB
- **Input**: mel-spectrogram, 3 seconds

### DeepSpeech2 (English)
- **Source**: TensorFlow → uint8 NB
- **Input**: spectrogram features
