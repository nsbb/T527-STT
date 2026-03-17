#!/usr/bin/env python3
"""Wav2Vec2 Korean PyTorch FP32 evaluation on Zeroth-Korean test set.
Uses variable-length input (not truncated to 3s)."""

import os, sys, csv, json, time
import numpy as np

def main():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from datasets import load_dataset

    OUTPUT_DIR = "/home/nsbb/travail/claude/T527/t527-stt/wav2vec2/base-korean"
    CACHE_DIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/zeroth_korean_cache"
    MAX_SAMPLES = 100

    # Load model
    print("Loading Kkonjeong/wav2vec2-base-korean...")
    processor = Wav2Vec2Processor.from_pretrained("Kkonjeong/wav2vec2-base-korean")
    model = Wav2Vec2ForCTC.from_pretrained("Kkonjeong/wav2vec2-base-korean")
    model.eval()

    # Load dataset
    print("Loading Zeroth-Korean test set...")
    ds = load_dataset("kresnik/zeroth_korean", split="test", cache_dir=CACHE_DIR)

    def compute_cer(ref, hyp):
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

    CHOSEONG = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
    JUNGSEONG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
    JONGSEONG = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'

    def jamo_to_syllable(text):
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

    results = []
    total_cer = 0

    for idx, sample in enumerate(ds):
        if idx >= MAX_SAMPLES:
            break

        audio = sample['audio']['array'].astype(np.float32)
        sr = sample['audio']['sampling_rate']
        gt = sample['text']
        duration_sec = len(audio) / sr

        # Process
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        t0 = time.time()
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        infer_ms = (time.time() - t0) * 1000

        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        pred_raw = processor.batch_decode(pred_ids)[0]
        pred = jamo_to_syllable(pred_raw)

        cer = compute_cer(gt, pred)
        exact = "Y" if gt.replace(' ', '') == pred.replace(' ', '') else "N"

        results.append({
            'idx': idx + 1,
            'wav': f"zeroth_test_{idx:04d}",
            'gt': gt,
            'pred': pred,
            'cer': f"{cer:.4f}",
            'exact_match': exact,
            'duration_sec': f"{duration_sec:.2f}",
            'infer_ms': f"{infer_ms:.1f}"
        })
        total_cer += cer

        if idx < 5 or cer == 0:
            print(f"  [{idx+1}] CER={cer:.2%} GT={gt[:40]} -> {pred[:40]}")

    avg_cer = total_cer / len(results)
    exact_count = sum(1 for r in results if r['exact_match'] == 'Y')
    print(f"\nTotal: {len(results)} samples, Avg CER: {avg_cer:.2%}, Exact: {exact_count}/{len(results)}")

    out_csv = os.path.join(OUTPUT_DIR, "test_results_zeroth_korean_pytorch_fp32.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['idx','wav','gt','pred','cer','exact_match','duration_sec','infer_ms'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
