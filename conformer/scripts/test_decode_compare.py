"""Compare decoder output between transcribe() and manual forward."""
import nemo.collections.asr as nemo_asr
import torch, soundfile as sf, numpy as np, tarfile

m = nemo_asr.models.EncDecCTCModelBPE.restore_from("/workspace/model/stt_kr_conformer_ctc_medium.nemo", map_location="cpu")
m.eval()
m.preprocessor.featurizer.dither = 0.0
m.preprocessor.featurizer.pad_to = 0

tf = tarfile.open("/workspace/model/stt_kr_conformer_ctc_medium.nemo")
vf = [f for f in tf.getnames() if "vocab.txt" in f][0]
vocab = tf.extractfile(vf).read().decode().strip().split("\n")

# Hook to capture log_probs from transcribe
captured = {}
orig_forward = m.forward.__func__

def hooked_forward(self, **kwargs):
    result = orig_forward(self, **kwargs)
    captured["log_probs"] = result[0].clone()
    captured["enc_len"] = result[1].clone()
    return result

import types
m.forward = types.MethodType(hooked_forward, m)

# transcribe
texts = m.transcribe(["/workspace/testset/ko_test_0004.wav"])
lp_transcribe = captured["log_probs"]
el_transcribe = captured["enc_len"]
print(f"transcribe log_probs: {lp_transcribe.shape}")
print(f"transcribe text: {texts[0][:70]}")

# manual forward
audio, sr = sf.read("/workspace/testset/ko_test_0004.wav", dtype="float32")
at = torch.tensor(audio).unsqueeze(0)
with torch.no_grad():
    lp_manual, el_manual, _ = m.forward(input_signal=at, input_signal_length=torch.tensor([len(audio)]))
print(f"manual log_probs: {lp_manual.shape}")

# Compare
min_t = min(lp_transcribe.shape[1], lp_manual.shape[1])
diff = (lp_transcribe[:, :min_t, :] - lp_manual[:, :min_t, :]).abs()
print(f"log_probs diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

# Decode manual
argmax = lp_manual[0].argmax(dim=-1)[:el_manual[0]]
decoded = []
prev = -1
for tid in argmax:
    tid = tid.item()
    if tid != prev:
        if tid != 5000: decoded.append(tid)
        prev = tid
text = ""
for tid in decoded:
    tk = vocab[tid] if tid < len(vocab) else f"<{tid}>"
    if tk.startswith("\u2581"):
        text += " " + tk[1:]
    else:
        text += tk
print(f"manual text: {text.strip()[:70]}")
