# Deploying Korean Speech Recognition on a 2 TOPS Edge NPU: W8A8 Quantization-Aware Training for Conformer CTC

**Authors:** Huisoo Kim, Heewon Kim, Gunhee Lee

**Affiliation:** HDC LABS (nsbb@hdc-labs.com)

---

## Abstract

Deploying automatic speech recognition (ASR) on ultra-low-power edge neural processing units (NPUs) remains an open challenge, particularly for non-English languages where quantization-robust architectures are less studied. We present the first deployment of Korean ASR on a 2 TOPS edge NPU (VeriSilicon VIP9000) under strict W8A8 quantization constraints, where both weights and activations must be represented as unsigned 8-bit integers. We systematically evaluate six ASR model families and find that only Conformer CTC, a CNN-attention hybrid, survives W8A8 quantization — pure Transformer architectures (Wav2Vec2, Zipformer, HuBERT) all degrade to near-random output. We propose a quantization-aware training (QAT) method with margin-based loss that recovers 78% of the post-training quantization (PTQ) degradation, reducing CER from 15.59% (PTQ) to 11.01% on 18,000 public AIHub benchmark samples across six evaluation domains, and to 7.24% on 368 real-environment samples recorded through a wall-pad intercom in a noisy office setting, with a real-time factor (RTF) of 0.08–0.09. We further discover that the fixed-window inference strategy imposed by the NPU's static input constraint unexpectedly outperforms full-length FP32 server inference on noisy short utterances (CER 11.13% vs. 13.19%), attributable to CTC hallucination suppression in bounded-length inputs. To our knowledge, this is the first published Korean ASR system deployed on a sub-5 TOPS NPU.

---

## 1. Introduction

The proliferation of edge devices in smart home and IoT applications—wall-mounted intercoms, smart speakers, and embedded control panels—has created demand for on-device automatic speech recognition that operates without network connectivity. While cloud-based Korean ASR services have achieved competitive accuracy [1, 2], edge deployment on ultra-low-power NPUs remains largely unexplored for Korean.

Korean is an agglutinative language in which grammatical relations are expressed by attaching suffixes and particles to word stems, generating a surface vocabulary far larger than English. Achieving competitive accuracy for Korean requires BPE vocabularies of approximately 2,000 subword tokens, compared to roughly 30 tokens sufficient for character-level English systems [14]. This vocabulary size compounds the W8A8 quantization challenge directly: with 2,049 CTC output classes sharing 256 discrete levels in uint8, the quantization step size for output logits is approximately 0.19, meaning that rounding noise of a single step can flip the top-1 prediction to an incorrect token. By contrast, an English model with 32 classes has a step size of roughly 7.6—40× larger margin for the same bit-width.

Modern edge SoCs such as the Allwinner T527 integrate NPUs rated at 1–2 TOPS (tera operations per second), orders of magnitude below the compute available on server GPUs or high-end mobile NPUs (Apple ANE ~16 TOPS, Qualcomm Hexagon ~15 TOPS). Critically, these low-end NPUs impose **W8A8 quantization**—both model weights and activations must be represented as unsigned 8-bit integers—a far more restrictive constraint than the W4A16 or W8A16 schemes studied in recent ASR quantization literature [3, 4, 5]. Under W8A8, the 256 discrete levels must capture the full dynamic range of both convolution filters and attention logits, making Transformer-based architectures particularly vulnerable.

Despite growing interest in edge ASR, no prior work has demonstrated Korean speech recognition on a sub-5 TOPS NPU. The closest related work includes: (i) a KAIST study proposing a Conformer-LSTM architecture for Korean edge ASR, achieving 14.57% CER in FP32 simulation without actual hardware deployment [6]; (ii) Apple's on-device Conformer achieving WER 4.5% in English at RTF 0.19 on Apple Watch, using FP16 on the Apple Neural Engine [7]; and (iii) ENERZAi's 1.58-bit Whisper model achieving 6.45% CER on Korean, without reported NPU deployment metrics [8]. No prior work addresses the intersection of Korean ASR, W8A8 quantization, and actual edge NPU deployment.

In this work, we make the following contributions:

1. **Architecture survival analysis under W8A8:** We systematically evaluate six ASR model families on a 2 TOPS NPU and demonstrate that only the Conformer CTC architecture—a CNN-attention hybrid—survives W8A8 quantization, while pure Transformer architectures degrade catastrophically. We attribute this to the stabilizing effect of depthwise convolutions on post-attention activation distributions.

2. **QAT with MarginLoss for edge ASR:** We propose a quantization-aware training method using a margin-based auxiliary loss that recovers 78% of the PTQ degradation, achieving 9.30% average CER across 18,368 Korean speech samples on 11 evaluation sets. We further show that training on a modest 84-hour subset outperforms training on the full 4,356-hour dataset, revealing a FakeQuantize overfitting phenomenon.

3. **Fixed-window inference advantage:** We discover that the fixed-window inference strategy mandated by the NPU's static input shape unexpectedly outperforms full-length FP32 server inference on noisy short utterances, with CER 11.13% vs. 13.19% on a 3,000-sample low-quality telephone dataset, due to suppression of CTC hallucination in bounded-length inputs.

---

## 2. Related Work

### 2.1 Edge ASR Deployment

Gulati et al. [9] introduced the Conformer architecture combining convolution and self-attention modules, which has become the dominant ASR encoder. Several works have explored Conformer deployment on edge devices. Apple [7] deployed a streaming Conformer on the Apple Watch Series 7 Neural Engine, achieving WER 4.5% at RTF 0.19 in FP16. ARM [10] demonstrated an INT8 Conformer-S on the simulated Ethos-U85 NPU (up to 4 TOPS) for English LibriSpeech, but only reported FP32 baseline WER (6.43%) without quantized accuracy. These works target English on higher-compute hardware with less restrictive quantization (FP16 or W8A16).

### 2.2 ASR Model Quantization

Ding et al. [3] achieved near-lossless 4-bit Conformer quantization using native QAT on LibriSpeech (WER 2.1% vs. 2.0% FP32). Li et al. [4] extended this to 2-bit with co-training, and recent work [5] explores sub-1-bit stochastic quantization. Amazon [11] proposed a sub-8-bit general quantizer for RNN-T and Conformer models. However, all these approaches assume flexible precision (W4A16 or mixed-precision) and focus exclusively on English. The W8A8 constraint—where activations are also quantized to 8-bit—has received less attention in the ASR literature, despite being the native requirement of commodity edge NPUs.

### 2.3 Korean ASR

The Korean ASR landscape has been advanced by large-scale datasets including AIHub [12] and KsponSpeech [13]. Server-side Korean Conformer models have achieved CER 5.7% with Transformer LM rescoring [14] and CER 7.33% with CTC-only decoding [15]. The SungBeom Conformer CTC Medium [16], trained on 13,946 hours of AIHub data, provides a strong Korean CTC baseline. For edge deployment, Wang et al. [6] analyzed a 28.9M Conformer-LSTM for Korean but did not proceed to quantization or hardware deployment. ENERZAi [8] demonstrated 1.58-bit quantization of Whisper for Korean (CER 6.45%) but did not report RTF or actual NPU inference measurements.

---

## 3. Method

### 3.1 Target Hardware

We target the Allwinner T527 system-on-chip, which integrates an octa-core ARM Cortex-A55 CPU and a VeriSilicon VIP9000NANOSI_PLUS NPU. The NPU operates at 696 MHz with a peak throughput of 2 TOPS for INT8 operations. The hardware imposes the following constraints:

- **W8A8 quantization only**: Both weights and activations must be unsigned 8-bit integers (uint8) with asymmetric affine scaling. No mixed-precision or FP16 fallback is available.
- **Static input shape**: The compiled network binary (NB) file requires fixed tensor dimensions at compile time.
- **Model size limit**: NB files exceeding approximately 120 MB cause runtime failures.

Model conversion uses the Acuity Toolkit 6.12 (VeriSilicon's proprietary quantization and compilation tool) with the VivanteIDE 5.7.2 simulator backend for NB generation.

### 3.2 Model Architecture

We adopt the SungBeom Conformer CTC Medium [16], a 122.5M-parameter model with 18 Conformer layers. Each layer comprises a feed-forward module, multi-head self-attention with relative positional encoding (8 heads, d_model=512), a convolution module (depthwise convolution, kernel size 31), and a second feed-forward module in a macaron-style layout. The decoder is a linear CTC head over a 2,049-token BPE vocabulary (2,048 Korean subwords + 1 blank token).

The model was pre-trained on 13,946 hours of AIHub Korean speech data. We fix the input to a static shape of [1, 80, 301], corresponding to 80 Slaney-scale mel-frequency bins over 301 frames (approximately 3.01 seconds at 16 kHz with 10 ms frame shift). The output shape is [1, 76, 2049], reflecting a 4x subsampling factor from the convolutional frontend.

### 3.3 Model Conversion Pipeline

The conversion from the NeMo training framework [17] to the NPU binary follows four stages: (1) ONNX export with dither disabled and padding removed for inference; (2) ONNX graph surgery to fix the input shape to [1, 80, 301], simplifying from 4,462 to 1,905 nodes via onnxsim, and patching Pad operator constant values; (3) Acuity import and uint8 KL-divergence calibration using 100 representative mel-spectrogram samples from AIHub; (4) NB export with the VIP9000NANOSI_PLUS target profile. The resulting NB file is 102 MB.

### 3.4 Quantization-Aware Training

Standard PTQ via Acuity's KL-divergence calibration incurs substantial CER degradation under W8A8. We employ QAT to adapt the model weights to quantization noise prior to PTQ conversion.

**FakeQuantize placement.** We insert simulated uint8 quantization at three positions: (i) encoder input (mel features), (ii) encoder output, and (iii) decoder output (CTC logits). Each FakeQuantize module computes per-tensor asymmetric affine parameters dynamically during training and applies the straight-through estimator (STE) for gradient computation.

**MarginLoss.** Under uint8 quantization with 2,049 output classes, the quantization step size for the output logits is approximately 0.19 (full range / 255). To ensure that the top-1 prediction survives quantization-induced perturbation, we introduce an auxiliary margin loss:

$$\mathcal{L}_{\text{margin}} = \frac{1}{T} \sum_{t=1}^{T} \text{ReLU}(m - (z_{t,\text{top1}} - z_{t,\text{top2}}))$$

where $z_{t,\text{top1}}$ and $z_{t,\text{top2}}$ are the highest and second-highest logit values at time step $t$, and $m = 0.3$ is the target margin (approximately 1.5× the uint8 step size). The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{CTC}} + \lambda \mathcal{L}_{\text{margin}}, \quad \lambda = 0.1$$

**Training configuration.** We fine-tune on 100,000 randomly sampled utterances from AIHub (84.63 hours, seed=42), split into 95,000 training and 5,000 validation samples. We use AdamW with learning rate 1e-5, weight decay 0.01, cosine annealing schedule (eta_min=1e-7), batch size 16, gradient clipping at 1.0, and train for 10 epochs (59,380 steps) in FP32 precision. The convolutional subsampling frontend is frozen during QAT, as it shows negligible quantization sensitivity. Training completes in approximately 2 hours on a single NVIDIA RTX 6000 Ada GPU (48 GB).

### 3.5 Fixed-Window Inference

Due to the static input shape constraint, we employ a sliding window strategy for utterances longer than 3 seconds. The mel spectrogram is segmented into overlapping chunks of 301 frames with a stride of 250 frames (51-frame overlap). Each chunk is independently quantized to uint8, processed by the NPU, and the resulting logits are concatenated—taking the first 63 frames from all chunks except the last (which contributes all 76 frames). CTC greedy decoding (argmax, consecutive duplicate collapse, blank removal) is applied to the concatenated logits, followed by BPE detokenization.

For utterances shorter than 3 seconds, the mel spectrogram is zero-padded to 301 frames. This zero-padding is a key element discussed in Section 6.

---

## 4. Experimental Setup

### 4.1 Evaluation Data

We evaluate on 18,368 Korean speech samples spanning 11 datasets in two categories:

**Public benchmarks (AIHub, 18,000 samples).** Six datasets of 3,000 samples each: *eval_clean* and *eval_other* from the Korean speech evaluation partition; *007-low-quality* (telephone recordings with low SNR); *009-lectures* (Korean university lectures); *010-meetings* (multi-speaker meeting recordings); and *012-consultation* (customer service calls).

**Self-collected real-environment data (368 samples).** Five datasets recorded through a wall-pad intercom in a furnished model apartment: two speakers (KSK: 108, HJY: 107) at standard distance; and three spatial conditions at 2m (51 samples), 2m without HVAC noise (51), and 3m distance (51).

### 4.2 Baseline Architectures

We compare six model families under identical W8A8 quantization:

| Model | Architecture | Params | Type | Vocab |
|:------|:------------|-------:|:-----|------:|
| Conformer CTC Medium | CNN + Attention | 122.5M | Hybrid | 2,049 |
| Wav2Vec2 Base Korean | Transformer | 94.4M | Encoder | 56–1,912 |
| Zipformer | Transformer variant | 40M | Encoder | 5,000 |
| HuBERT Korean | Transformer | 96M | Encoder | 2,142 |
| KoCitrinet | CNN only | ~10M | CNN | 2,048 |
| DeepSpeech2 | RNN + CNN | ~30M | RNN | 2,338 |

### 4.3 Metrics

We report **Character Error Rate (CER)** computed on space-removed text using Levenshtein edit distance, following standard Korean ASR evaluation practice [14]. **Real-Time Factor (RTF)** is the ratio of inference time to audio duration, measured on the T527 NPU at 696 MHz.

---

## 5. Results

### 5.1 Architecture Survival under W8A8

Table 1 presents the W8A8 quantization results for all six architectures.

**Table 1.** ASR architecture comparison under W8A8 uint8 quantization on T527 NPU.

| Model | Params | NB Size | FP32 CER | uint8 CER | Inference | Status |
|:------|-------:|--------:|---------:|----------:|----------:|:-------|
| **Conformer CTC** | **122.5M** | **102 MB** | **~6–10%** | **10.59%** | **233 ms** | **Success** |
| KoCitrinet | ~10M | 62 MB | 8.44% | 44.44% | 120 ms | Degraded |
| Wav2Vec2 (EN) | 94.4M | 87 MB | 9.74% | 17.52% | 715 ms | Degraded |
| Wav2Vec2 (KO) | 94.4M | 77 MB | 9–18% | 92.83% | 424 ms | Failed |
| Zipformer | 40M | 63 MB | 16.2% | 100% | 50 ms | Failed |
| HuBERT (KO) | 96M | 76 MB | — | 100% | 423 ms | Failed |

Only the Conformer architecture produces usable recognition under W8A8. The pure Transformer models (Wav2Vec2, Zipformer, HuBERT) all degrade to near-random output. We attribute this to the role of depthwise convolution in each Conformer block: after multi-head self-attention computes query-key dot products that are sensitive to quantization noise, the subsequent depthwise convolution (kernel size 31) and batch normalization act as a stabilizing bottleneck that re-normalizes the activation distribution before the next attention layer. Without this convolutional regularization, small quantization errors in attention logits compound across layers, causing catastrophic output degradation.

KoCitrinet, a purely convolutional model, survives quantization (CER 44.44%) but lacks the attention mechanism needed for competitive accuracy. This supports the hypothesis that the CNN-attention hybrid structure of Conformer provides the optimal balance between quantization robustness (from convolutions) and modeling capacity (from attention).

### 5.2 PTQ vs. QAT Performance

Table 2 presents the full 11-dataset comparison across FP32 server inference, PTQ, and QAT.

**Table 2.** CER (%) comparison: FP32 (server, full-length), PTQ uint8 (device), and QAT uint8 (device) across 11 evaluation datasets. Bold indicates best device result.

| Category | Dataset | Samples | FP32 | PTQ | QAT | QAT Loss |
|:---------|:--------|--------:|-----:|----:|----:|---------:|
| Self | 7F_KSK | 108 | 1.67 | 21.20 | **2.58** | +0.91 |
| Self | modelhouse_nh | 51 | 1.42 | 7.49 | **3.11** | +1.69 |
| Self | 7F_HJY | 107 | 6.89 | 15.64 | **8.33** | +1.44 |
| Self | modelhouse_2m | 51 | 6.65 | 16.14 | **8.62** | +1.97 |
| Self | modelhouse_3m | 51 | 13.51 | 21.73 | **13.57** | +0.06 |
| **Self** | **Average** | **368** | **6.03** | **16.44** | **7.24** | **+1.21** |
| AIHub | 012-consultation | 3,000 | 4.49 | 10.03 | **7.46** | +2.97 |
| AIHub | eval_other | 3,000 | 7.35 | 13.30 | **10.41** | +3.06 |
| AIHub | eval_clean | 3,000 | 8.17 | 16.01 | **10.95** | +2.78 |
| AIHub | 007-low-quality | 3,000 | 13.19 | 19.01 | **11.13** | **-2.06** |
| AIHub | 009-lectures | 3,000 | 9.27 | 15.66 | **12.42** | +3.15 |
| AIHub | 010-meetings | 3,000 | 9.75 | 15.29 | **13.66** | +3.91 |
| **AIHub** | **Average** | **18,000** | **8.70** | **14.88** | **11.01** | **+2.31** |
| **All** | **Average** | **18,368** | **7.49** | **15.59** | **9.30** | **+1.81** |

QAT improves over PTQ on all 11 datasets without exception. The overall quantization loss is reduced from +8.10 percentage points (PTQ) to +1.81 percentage points (QAT), a **78% recovery**. On self-collected real-environment data—representative of the target wall-pad deployment—QAT achieves 7.24% CER, within 1.21 percentage points of the FP32 server baseline.

Notably, the 007-low-quality dataset is the only case where QAT *outperforms* FP32 (11.13% vs. 13.19%, a 2.06 percentage point advantage). We analyze this phenomenon in Section 6.

### 5.3 On-Device Performance

Table 3 reports inference latency and RTF on the T527 NPU.

**Table 3.** On-device performance by audio duration.

| Audio Duration | Chunks | Inference Time | RTF |
|---------------:|-------:|---------------:|----:|
| 3 s | 1 | 233 ms | 0.078 |
| 5 s | 2 | 466 ms | 0.093 |
| 10 s | 4 | 932 ms | 0.093 |
| 15 s | 6 | 1.40 s | 0.093 |
| 20 s | 8 | 1.86 s | 0.093 |

The system achieves RTF 0.078–0.093, corresponding to 11–13× real-time performance. Each 3-second chunk requires 233 ms of NPU inference, with mel-spectrogram computation adding approximately 15 ms on the ARM CPU. The 102 MB NB model loads in under 500 ms at startup.

### 5.4 QAT Training Data Size

A counterintuitive finding emerges from our data scaling experiments. Table 4 compares QAT with a 100k-sample subset (84.63 hours) versus the full AIHub dataset (4.09M samples, 4,356 hours).

**Table 4.** QAT training data size comparison.

| Config | Data | Hours | Steps | val_loss | Device CER |
|:-------|-----:|------:|------:|---------:|-----------:|
| 100k subset | 95k | 84.63 | 59,380 | 0.142 | **9.30%** |
| Full dataset | 4.09M | 4,356 | 2,557,560 | 0.069 | 14.81% |

Despite achieving a substantially lower validation loss (0.069 vs. 0.142), full-dataset QAT yields *worse* on-device CER than the 100k subset (14.81% vs. 9.30%). We attribute this to **FakeQuantize overfitting**: the dynamic per-tensor FakeQuantize used during training simulates a different quantization regime than the static KL-divergence calibration applied by Acuity during actual PTQ conversion. With 43× more gradient updates, the full-dataset model over-adapts to the training-time quantization noise distribution, losing generalization to the deployment-time quantizer. This finding echoes recent observations on catastrophic forgetting in QAT [18] and suggests that practitioners should limit QAT fine-tuning to a small fraction of the original pre-training budget.

---

## 6. Analysis: Fixed-Window INT8 vs. Full-Length FP32

### 6.1 Observation

Among 11 evaluation datasets, the 007-low-quality telephone dataset is the only case where the W8A8 INT8 model outperforms the FP32 server baseline (Table 2). Stratifying by utterance duration reveals the source of this reversal.

**Table 5.** CER (%) by utterance duration on 007-low-quality (3,000 samples).

| Duration | Samples | FP32 | INT8 QAT | Difference |
|---------:|--------:|-----:|---------:|-----------:|
| **≤ 3 s** | **1,012** | **17.85** | **8.65** | **-9.20** |
| 3–6 s | 1,332 | 11.77 | 12.55 | +0.78 |
| 6–10 s | 527 | 9.05 | 12.13 | +3.08 |
| 10–20 s | 123 | 8.31 | 11.93 | +3.62 |
| > 20 s | 6 | 7.45 | 11.95 | +4.50 |

The INT8 system's advantage is concentrated entirely in utterances of 3 seconds or shorter, where it achieves a 9.20 percentage point improvement. For longer utterances, FP32 performs as expected. The overall dataset-level reversal occurs because 007-low-quality contains 33.7% short utterances (1,012 / 3,000), sufficient to dominate the aggregate CER.

### 6.2 CTC Hallucination in Full-Length Inference

Examining the short (≤ 3 s) utterance subset reveals a striking pattern in hypothesis lengths.

**Table 6.** Output statistics for ≤ 3 s utterances in 007-low-quality.

| Metric | FP32 | INT8 QAT |
|:-------|-----:|---------:|
| Mean reference length (chars) | 8.8 | 8.8 |
| Mean hypothesis length (chars) | 20.2 | 8.7 |
| Hypothesis / reference ratio | **2.30×** | **0.99×** |
| Insertion-dominant samples (hyp > ref) | 73.6% | 5.3% |
| Exact match rate | 40.6% | 62.2% |
| Catastrophic errors (CER ≥ 80%) | 30 | 7 |

The FP32 server model produces hypotheses averaging 2.3× the reference length on short noisy utterances—a **CTC hallucination** phenomenon where the model assigns non-blank tokens to noise frames. Examples include: reference "무엇을도와드릴까요" (9 chars) producing FP32 hypothesis "뭐수어을샷" (CER 100%), while the INT8 system correctly outputs "무엇을 도와드릴까요" (CER 0%).

**Root cause.** In full-length inference, a short noisy utterance (e.g., 1.8 seconds, ~112 mel frames) is processed with its natural length. When the SNR is low, noise frames constitute a large fraction of the input, and the CTC decoder assigns tokens to noise—producing insertion-heavy hallucinated output. In fixed-window inference, the same audio is zero-padded to 301 frames, and the CTC decoder correctly assigns blank tokens to the padded (silent) region, while concentrating token predictions in the genuine speech region.

### 6.3 Robustness Verification

To verify that the advantage is not solely due to outlier catastrophic failures, we exclude all samples with CER ≥ 80% from both systems:

| Condition | Samples | FP32 CER | INT8 CER | Difference |
|:----------|--------:|---------:|---------:|-----------:|
| All samples | 3,000 | 13.19% | 11.13% | -2.06 |
| Both CER < 80% | 2,960 | 12.26% | 10.62% | **-1.64** |

Even after removing catastrophic failures, the INT8 fixed-window system maintains a 1.64 percentage point advantage, confirming that the effect is not driven solely by extreme outliers. A secondary contributing factor is the implicit regularization of uint8 quantization, which suppresses low-magnitude spurious activations caused by noise.

### 6.4 Cross-Dataset Validation

To confirm that the effect requires both low SNR and short duration, we compare with 012-consultation, another telephone dataset:

| Dataset | ≤ 3 s samples | FP32 CER (≤ 3 s) | INT8 CER (≤ 3 s) |
|:--------|-------------:|---------:|---------:|
| 007-low-quality | 1,012 | 17.85% | 8.65% |
| 012-consultation | 1,046 | 4.52% | 4.73% |

The consultation dataset, despite being telephone speech, has adequate SNR—FP32 achieves a normal 4.52% CER on short utterances, and the INT8 system shows no advantage. The fixed-window advantage manifests only when the combination of low SNR and short duration causes FP32 CTC to hallucinate.

---

## 7. Discussion

### 7.1 Practical Implications

Our results suggest that for ultra-low-power edge deployment (1–2 TOPS, W8A8), the Conformer CTC architecture with QAT is currently the only viable path for Korean ASR. The finding that fixed-window inference improves robustness on noisy short commands is particularly relevant for the target smart-home scenario, where typical user interactions are brief voice commands spoken near a wall-pad device. The architectural survival analysis provides a practical elimination criterion for practitioners evaluating ASR models for edge NPU deployment: if the target hardware requires W8A8, pure Transformer encoders should be ruled out.

### 7.2 QAT Recipe Guidelines

Our experiments yield three practical guidelines: (1) **Limit QAT duration** — 84 hours of data (59k steps) outperformed 4,356 hours (2.56M steps), suggesting QAT fine-tuning should remain within 1% of the original pre-training budget; (2) **Validate on target hardware** — validation loss is a poor proxy for device CER due to the mismatch between FakeQuantize and actual PTQ quantizers; (3) **Match calibration to training distribution** — calibration with 100 AIHub samples (matching the pre-training distribution) yielded 7.24% CER versus 15.0% with domain-specific but distribution-mismatched samples.

### 7.3 Limitations

Several limitations should be noted. First, evaluation is conducted on a single hardware platform (VeriSilicon VIP9000); generalization to other W8A8 NPUs requires further study. Second, the CER gap between QAT and FP32 remains significant for challenging domains (meetings: 13.66% vs. 9.75%; lectures: 12.42% vs. 9.27%). Third, we employ only CTC greedy decoding; language model integration or beam search could further improve accuracy but is constrained by the NPU's computational budget. Finally, the fixed 3-second window introduces overlap-boundary artifacts for long utterances, and the zero-padding strategy for short utterances adds unnecessary computation.

---

## 8. Conclusion

We present the first deployment of Korean speech recognition on a 2 TOPS edge NPU under W8A8 quantization constraints. Through systematic evaluation of six ASR architectures, we establish that Conformer CTC is uniquely suited to W8A8 quantization due to its CNN-attention hybrid structure. Our QAT method with MarginLoss recovers 78% of the PTQ degradation, achieving 11.01% CER on public AIHub benchmarks (18,000 samples) and 7.24% on real-environment deployment data (368 samples), at RTF 0.08–0.09. An unexpected finding is that the fixed-window inference constraint actually benefits noisy short-utterance recognition by suppressing CTC hallucination, outperforming FP32 full-length inference by 2.06 percentage points on low-quality telephone speech. These results demonstrate the feasibility of competitive Korean ASR on ultra-low-power edge hardware and provide practical guidelines for W8A8 quantization of speech models.

---

## References

[1] Return Zero, "Korean ASR Benchmark," github.com/rtzr/Awesome-Korean-Speech-Recognition, 2024.

[2] Naver Clova, "CLOVA Speech Recognition API," 2023.

[3] S. Ding, et al., "4-bit Conformer with Native Quantization Aware Training," in *Proc. INTERSPEECH*, 2022.

[4] Y. Li, et al., "Towards 2-bit Conformer Quantization: A Co-Training Approach," in *Proc. INTERSPEECH*, 2023.

[5] Z. Chen, et al., "Towards One-bit ASR: Stochastic Precision for Streaming Conformer," *arXiv:2505.21245*, 2025.

[6] H. Wang, et al., "Edge-Oriented Korean ASR Model Design Based on Conformer-LSTM," in *Proc. IEIE Fall Conference*, 2023.

[7] Y. Shangguan, et al., "On-Device Speech Recognition on Apple Watch," in *Proc. NAACL Industry Track*, 2024.

[8] ENERZAi, "Small Models, Big Heat: Conquering Korean ASR with Low-Bit Whisper," enerzai.com/resources/blog, 2025.

[9] A. Gulati, et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Proc. INTERSPEECH*, 2020.

[10] ARM, "End-to-End INT8 Conformer on Ethos-U85," developer.arm.com/blog, 2025.

[11] Amazon, "Sub-8-Bit Quantization for On-Device Speech Recognition: A Regularization-Free Approach," in *Proc. ASRU*, 2022.

[12] AI Hub, "Korean Speech Recognition Corpus," aihub.or.kr, 2020–2024.

[13] J. Bang, et al., "KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition," *Applied Sciences*, 2020.

[14] J. Koo, et al., "Korean End-to-End Speech Recognition Using Conformer," *Journal of the Acoustical Society of Korea*, vol. 40, no. 5, 2021.

[15] SpeechBrain, "Conformer-TransformerLM on KsponSpeech," huggingface.co/speechbrain, 2023.

[16] SungBeom, "stt_kr_conformer_ctc_medium," huggingface.co/SungBeom, 2024.

[17] NVIDIA, "NeMo: Neural Modules Toolkit," github.com/NVIDIA/NeMo, 2023.

[18] Z. Chen, et al., "Overcoming Forgetting Catastrophe in Quantization-Aware Training," in *Proc. ICCV*, 2023.

---

*[Figure 1: System overview — pipeline from NeMo training through ONNX conversion, Acuity quantization, and T527 NPU deployment with sliding window inference.]*

*[Figure 2: CER comparison across 11 evaluation datasets for FP32 (server), PTQ (device), and QAT (device). Grouped bar chart showing consistent QAT improvement over PTQ.]*

*[Figure 3: CER by utterance duration on 007-low-quality dataset. FP32 and INT8 QAT lines crossing at approximately 3 seconds, showing the fixed-window advantage region.]*
