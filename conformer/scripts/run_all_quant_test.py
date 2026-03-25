#!/usr/bin/env python3
"""3가지 양자화 (uint8 KL, uint8 MA, int16 DFP) × 100샘플 슬라이딩 윈도우 테스트."""
import numpy as np
import json
import subprocess
import csv
import time
import sys
import os

ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK = "/home/nsbb/travail/claude/T527/ai-sdk/models/conformer/kr_sungbeom"
MEL_DIR = f"{WORK}/nemo_mel"
VOCAB_PATH = f"{WORK}/vocab_correct.json"
GT_PATH = "/home/nsbb/travail/claude/T527/t527-stt/testset/base_korean/ground_truth.txt"
CSV_DIR = "/home/nsbb/travail/claude/T527/t527-stt/conformer"

vocab = json.load(open(VOCAB_PATH))
info = json.load(open(f"{MEL_DIR}/info.json"))
gt_map = {}
for line in open(GT_PATH):
    if line.startswith("#") or not line.strip(): continue
    parts = line.strip().split("\t")
    if len(parts) >= 2: gt_map[parts[0]] = parts[1]

WINDOW, STRIDE, SEQ_OUT = 301, 250, 76
STRIDE_OUT = int(SEQ_OUT * STRIDE / WINDOW)
BLANK_ID = 2048
VOCAB_SIZE = 2049

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if s1[i-1]==s2[j-1] else 1))
    return dp[m][n]

def decode_tokens(collapsed):
    text = ""
    for tid in collapsed:
        tk = vocab.get(str(tid), f"<{tid}>")
        if tk == "<unk>": tk = "\u2047"
        if tk.startswith("\u2581"): text += " " + tk[1:]
        else: text += tk
    return text.strip()

QUANT_CONFIGS = {
    "uint8_kl": {
        "nb": f"{WORK}/wksp_nbg_unify/network_binary.nb",
        "meta": f"{WORK}/wksp_nbg_unify/nbg_meta.json",
        "device_dir": "/data/local/tmp/kr_conf_sb",
        "dtype": "uint8",
    },
    "uint8_ma": {
        "nb": f"{WORK}/wksp_uint8_ma_nbg_unify/network_binary.nb",
        "meta": f"{WORK}/wksp_uint8_ma_nbg_unify/nbg_meta.json",
        "device_dir": "/data/local/tmp/kr_conf_sb_ma",
        "dtype": "uint8",
    },
    "int16_dfp": {
        "nb": f"{WORK}/wksp_int16_nbg_unify/network_binary.nb",
        "meta": f"{WORK}/wksp_int16_nbg_unify/nbg_meta.json",
        "device_dir": "/data/local/tmp/kr_conf_sb_int16",
        "dtype": "int16",
    },
}

def quantize_input(mel_chunk, meta, dtype):
    inp_q = list(meta["Inputs"].values())[0]["quantize"]
    if dtype == "uint8":
        s, z = inp_q["scale"], inp_q["zero_point"]
        return np.clip(np.round(mel_chunk / s) + z, 0, 255).astype(np.uint8)
    else:  # int16 DFP
        fl = inp_q["fl"]
        return np.clip(np.round(mel_chunk * (2**fl)), -32768, 32767).astype(np.int16)

def dequantize_output(raw_bytes, meta, dtype):
    out_q = list(meta["Outputs"].values())[0]["quantize"]
    if dtype == "uint8":
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        s, z = out_q["scale"], out_q["zero_point"]
        return (raw.astype(np.float32) - z) * s
    else:  # int16 DFP
        raw = np.frombuffer(raw_bytes, dtype=np.int16)
        fl = out_q["fl"]
        return raw.astype(np.float32) / (2**fl)

def run_test(qname, config):
    print(f"\n{'='*60}")
    print(f"=== {qname} ===")
    print(f"{'='*60}")

    meta = json.load(open(config["meta"]))
    dtype = config["dtype"]
    device_dir = config["device_dir"]

    # Push NB
    subprocess.run([ADB, "shell", f"mkdir -p {device_dir}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([ADB, "push", config["nb"], f"{device_dir}/network_binary.nb"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Sample file
    subprocess.run([ADB, "shell",
        f"printf '[network]\n{device_dir}/network_binary.nb\n[input]\n{device_dir}/chunk.dat\n[output]\n{device_dir}/chunk_out.dat\n' > {device_dir}/sample.txt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    results = []
    total_ed, total_gt_len, total_chunks = 0, 0, 0
    total_infer_us = 0

    for idx in range(100):
        mel_full = np.load(f"{MEL_DIR}/mel_{idx:04d}.npy")
        total_frames = mel_full.shape[2]
        dur = info[str(idx)]["duration"]
        gt = gt_map.get(f"ko_test_{idx:04d}.wav", "")

        # Split chunks
        chunks = []
        start = 0
        while start < total_frames:
            end = start + WINDOW
            chunk = mel_full[:, :, start:end]
            if chunk.shape[2] < WINDOW:
                chunk = np.pad(chunk, ((0,0),(0,0),(0, WINDOW - chunk.shape[2])))
            chunks.append(chunk)
            if end >= total_frames: break
            start += STRIDE

        all_logits = []
        sample_infer_us = 0
        for ci, chunk in enumerate(chunks):
            q = quantize_input(chunk, meta, dtype)
            q.tofile("/tmp/quant_chunk.dat")
            subprocess.run([ADB, "push", "/tmp/quant_chunk.dat", f"{device_dir}/chunk.dat"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            ret = subprocess.run(
                [ADB, "shell", f"cd {device_dir} && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0"],
                capture_output=True, text=True, timeout=30)

            # Parse inference time
            for line in ret.stdout.split('\n'):
                if 'run time for this network' in line:
                    try:
                        us = int(line.split(':')[1].strip().split()[0])
                        sample_infer_us += us
                    except: pass

            subprocess.run([ADB, "pull", f"{device_dir}/chunk_out.dat", "/tmp/quant_chunk_out.dat"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            total_chunks += 1

            raw_bytes = open("/tmp/quant_chunk_out.dat", "rb").read()
            logits = dequantize_output(raw_bytes, meta, dtype)
            logits = logits.reshape(SEQ_OUT, VOCAB_SIZE)
            all_logits.append(logits)

        total_infer_us += sample_infer_us

        # Merge
        merged = []
        for ci, logits in enumerate(all_logits):
            merged.append(logits[:STRIDE_OUT] if ci < len(all_logits)-1 else logits)
        merged_logits = np.concatenate(merged, axis=0)

        argmax = np.argmax(merged_logits, axis=1)
        blank = int(np.sum(argmax == BLANK_ID))
        collapsed = []
        prev = -1
        for tid in argmax:
            if tid != prev:
                if tid != BLANK_ID: collapsed.append(int(tid))
                prev = tid
        text = decode_tokens(collapsed)

        gt_clean = gt.replace(" ", "")
        npu_clean = text.replace(" ", "").replace("\u2047", "").replace(".", "")
        ed = edit_distance(npu_clean, gt_clean)
        cer = ed / max(len(gt_clean), 1) * 100
        total_ed += ed
        total_gt_len += len(gt_clean)

        infer_ms = round(sample_infer_us / 1000, 1)
        results.append({
            "id": idx, "wav": f"ko_test_{idx:04d}.wav", "duration": dur,
            "chunks": len(chunks), "infer_ms": infer_ms,
            "gt": gt, "npu": text,
            "blank_ratio": round(blank/max(len(argmax),1)*100,1),
            "gt_len": len(gt_clean), "npu_len": len(npu_clean),
            "edit_dist": ed, "cer": round(cer, 2)
        })

        if idx % 20 == 0:
            print(f"  [{idx:02d}] {dur}s {len(chunks)}ch {infer_ms}ms CER={cer:.1f}%", flush=True)

    avg_cer = total_ed / max(total_gt_len, 1) * 100
    avg_infer = total_infer_us / total_chunks / 1000 if total_chunks else 0
    print(f"\n  {qname}: CER={avg_cer:.2f}%, avg_infer={avg_infer:.0f}ms/chunk, {total_chunks} chunks")

    # Save CSV
    csv_path = f"{CSV_DIR}/kr_sungbeom_{qname}_100.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","wav","duration","chunks","infer_ms","gt","npu","blank_ratio","gt_len","npu_len","edit_dist","cer"])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_path}")
    return avg_cer, avg_infer

# Run all 3
summary = {}
for qname, config in QUANT_CONFIGS.items():
    cer, infer = run_test(qname, config)
    summary[qname] = {"cer": cer, "infer_ms": infer}

print(f"\n{'='*60}")
print(f"=== SUMMARY ===")
print(f"{'='*60}")
for qname, s in summary.items():
    print(f"  {qname:12s}: CER={s['cer']:.2f}%, infer={s['infer_ms']:.0f}ms/chunk")
