#!/usr/bin/env python3
"""Wav2Vec2 Korean ONNX FP32 evaluation on wallpad testsets.
Generates per-sample CSV files for each testset."""

import os, sys, csv, json, time
import numpy as np

# Lazy imports to avoid LD_LIBRARY_PATH conflicts
def main():
    import onnxruntime as ort
    from scipy.io import wavfile
    from scipy.signal import resample

    import librosa

    ONNX_MODEL = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx"
    VOCAB_JSON = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/vocab.json"
    TESTSET_DIR = "/mnt/c/Users/nsbb/travail/STT/testset"
    OUTPUT_DIR = "/home/nsbb/travail/claude/T527/t527-stt/wav2vec2/base-korean"

    INPUT_SAMPLES = 48000  # 3 seconds @ 16kHz

    # Load vocab
    with open(VOCAB_JSON) as f:
        vocab = json.load(f)
    id2char = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("[PAD]", 53)

    # Load ONNX model
    print(f"Loading ONNX model: {os.path.basename(ONNX_MODEL)}")
    sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def load_audio(path, target_sr=16000):
        """Load audio file (wav/mp3/flac) and return float32 array at target_sr."""
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32)

    def normalize_wav2vec2(audio):
        """Wav2Vec2FeatureExtractor normalization."""
        mean = np.mean(audio)
        var = np.var(audio)
        return ((audio - mean) / np.sqrt(var + 1e-7)).astype(np.float32)

    def ctc_decode(logits, id2char, pad_id):
        """CTC greedy decode with jamo-to-syllable conversion."""
        tokens = np.argmax(logits, axis=-1)
        if tokens.ndim > 1:
            tokens = tokens[0]

        # Deduplicate + remove PAD
        deduped = []
        prev = -1
        for t in tokens:
            if t != prev:
                if t != pad_id:
                    deduped.append(t)
                prev = t

        # Convert to text
        text = ''.join(id2char.get(t, '') for t in deduped)

        # Jamo to syllable
        text = jamo_to_syllable(text)
        return text

    def jamo_to_syllable(text):
        """Convert decomposed jamo sequence to composed Korean syllables."""
        CHOSEONG = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
        JUNGSEONG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
        JONGSEONG = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'

        result = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in CHOSEONG:
                cho = CHOSEONG.index(c)
                if i + 1 < len(text) and text[i + 1] in JUNGSEONG:
                    jung = JUNGSEONG.index(text[i + 1])
                    jong = 0
                    if i + 2 < len(text) and text[i + 2] in JONGSEONG.strip():
                        # Check if next-next is jungseong (then current is not jongseong)
                        if i + 3 < len(text) and text[i + 3] in JUNGSEONG:
                            jong = 0
                        else:
                            jong = JONGSEONG.index(text[i + 2])
                            i += 1
                    syllable = chr(0xAC00 + cho * 21 * 28 + jung * 28 + jong)
                    result.append(syllable)
                    i += 2
                else:
                    result.append(c)
                    i += 1
            else:
                result.append(c)
                i += 1
        return ''.join(result)

    def compute_cer(ref, hyp):
        """Character Error Rate using edit distance."""
        ref = ref.replace(' ', '')
        hyp = hyp.replace(' ', '')
        if len(ref) == 0:
            return 1.0 if len(hyp) > 0 else 0.0

        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return d[len(ref)][len(hyp)] / len(ref)

    # Process each testset
    testsets = ["modelhouse_3m", "modelhouse_2m", "modelhouse_2m_noheater", "7F_KSK", "7F_HJY"]

    for testset_name in testsets:
        csv_path = os.path.join(TESTSET_DIR, f"{testset_name}.csv")
        audio_dir = os.path.join(TESTSET_DIR, testset_name)

        if not os.path.exists(csv_path):
            print(f"Skip {testset_name}: CSV not found")
            continue

        print(f"\n=== {testset_name} ===")

        # Read ground truth CSV
        samples = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['FileName'].split('\\')[-1]
                gt = row['gt']
                samples.append((filename, gt))

        results = []
        total_cer = 0
        for idx, (filename, gt) in enumerate(samples, 1):
            filepath = os.path.join(audio_dir, filename)
            if not os.path.exists(filepath):
                print(f"  [{idx}] Missing: {filename}")
                continue

            try:
                audio = load_audio(filepath)
                duration_sec = len(audio) / 16000

                # Pad/truncate to INPUT_SAMPLES
                if len(audio) > INPUT_SAMPLES:
                    audio = audio[:INPUT_SAMPLES]
                else:
                    audio = np.pad(audio, (0, INPUT_SAMPLES - len(audio)))

                # Normalize
                audio = normalize_wav2vec2(audio)

                # Inference
                t0 = time.time()
                logits = sess.run(None, {input_name: audio.reshape(1, -1)})[0]
                infer_ms = (time.time() - t0) * 1000

                # Decode
                pred = ctc_decode(logits, id2char, pad_id)
                cer = compute_cer(gt, pred)
                exact = "Y" if gt.replace(' ', '') == pred.replace(' ', '') else "N"

                results.append({
                    'idx': idx,
                    'wav': filename,
                    'gt': gt,
                    'pred': pred,
                    'cer': f"{cer:.4f}",
                    'exact_match': exact,
                    'duration_sec': f"{duration_sec:.2f}",
                    'infer_ms': f"{infer_ms:.1f}"
                })
                total_cer += cer

                if idx <= 3 or cer < 0.3:
                    print(f"  [{idx}] CER={cer:.2%} GT={gt} -> {pred}")
            except Exception as e:
                print(f"  [{idx}] Error {filename}: {e}")

        if results:
            avg_cer = total_cer / len(results)
            print(f"  Total: {len(results)} samples, Avg CER: {avg_cer:.2%}")

            # Write CSV
            out_csv = os.path.join(OUTPUT_DIR, f"test_results_{testset_name}_onnx_fp32.csv")
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['idx','wav','gt','pred','cer','exact_match','duration_sec','infer_ms'])
                writer.writeheader()
                writer.writerows(results)
            print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()
