"""Patch only rel_shift() to use gather instead of pad+reshape.
Keep everything else in the original NeMo code untouched.
"""
import nemo.collections.asr as nemo_asr
import nemo.collections.asr.parts.submodules.multi_head_attention as mha
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, soundfile as sf, tarfile, sys

m = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    '/workspace/model/stt_kr_conformer_ctc_medium.nemo', map_location='cpu')
m.eval()
enc, dec = m.encoder, m.decoder

# Vocab
t = tarfile.open('/workspace/model/stt_kr_conformer_ctc_medium.nemo')
vf = [f for f in t.getnames() if 'vocab.txt' in f][0]
vocab = t.extractfile(vf).read().decode().strip().split('\n')

SEQ_LEN = 76
POS_LEN = 2 * SEQ_LEN - 1  # 151

# Pre-compute rel_shift gather indices
# rel_shift: pad(left=1) → reshape → drop first row → reshape → truncate
# Result: indices[i, j] = SEQ_LEN - 1 - i + j
rel_indices = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.long)
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        rel_indices[i, j] = SEQ_LEN - 1 - i + j

# Monkey-patch rel_shift on ALL RelPositionMultiHeadAttention instances
def fixed_rel_shift(self, x):
    """Replace dynamic pad+reshape with pre-computed gather."""
    b, h, qlen, pos_len = x.size()
    idx = rel_indices.to(x.device).unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
    return torch.gather(x, 3, idx)

# Patch the method on the class
mha.RelPositionMultiHeadAttention.rel_shift = fixed_rel_shift
print("Patched rel_shift on class")

# Verify: original model inference
audio, sr = sf.read('/workspace/testset/ko_test_0004.wav', dtype='float32')
audio = audio[:48160]
if len(audio) < 48160: audio = np.pad(audio, (0, 48160 - len(audio)))
mel_t = torch.tensor(audio).unsqueeze(0)
with torch.no_grad():
    mel, _ = m.preprocessor(input_signal=mel_t, length=torch.tensor([mel_t.shape[1]]))
    if mel.shape[2] > 301: mel = mel[:, :, :301]
    elif mel.shape[2] < 301: mel = F.pad(mel, (0, 301 - mel.shape[2]))

    encoded, _ = enc(audio_signal=mel, length=torch.tensor([301]))
    logprobs = dec(encoder_output=encoded)
    argmax = logprobs[0].argmax(dim=-1)
    blank = (argmax == 5000).sum().item()

    decoded = []
    prev = -1
    for tid in argmax:
        tid = tid.item()
        if tid != prev:
            if tid != 5000: decoded.append(tid)
            prev = tid
    text = ''
    for tid in decoded:
        tk = vocab[tid] if tid < len(vocab) else f'<{tid}>'
        text += (' ' + tk[1:]) if tk.startswith('\u2581') else tk

print(f"PyTorch: blank={blank}/76: {text.strip()[:70]}")
print(f"GT: 평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다")
sys.stdout.flush()

# Also test sample 0
audio0, _ = sf.read('/workspace/testset/ko_test_0000.wav', dtype='float32')
audio0 = audio0[:48160]
if len(audio0) < 48160: audio0 = np.pad(audio0, (0, 48160 - len(audio0)))
with torch.no_grad():
    mel0, _ = m.preprocessor(input_signal=torch.tensor(audio0).unsqueeze(0), length=torch.tensor([len(audio0)]))
    if mel0.shape[2] > 301: mel0 = mel0[:, :, :301]
    elif mel0.shape[2] < 301: mel0 = F.pad(mel0, (0, 301 - mel0.shape[2]))
    encoded0, _ = enc(audio_signal=mel0, length=torch.tensor([301]))
    lp0 = dec(encoder_output=encoded0)
    am0 = lp0[0].argmax(dim=-1)
    bl0 = (am0 == 5000).sum().item()
    d0 = []
    prev = -1
    for tid in am0:
        tid = tid.item()
        if tid != prev:
            if tid != 5000: d0.append(tid)
            prev = tid
    t0 = ''
    for tid in d0:
        tk = vocab[tid] if tid < len(vocab) else f'<{tid}>'
        t0 += (' ' + tk[1:]) if tk.startswith('\u2581') else tk
print(f"[00] blank={bl0}/76: {t0.strip()[:70]}")
print(f"GT0: 몬터규는 자녀들이 사랑을 제대로 못 받고 크면 매우 심각한 결과가 초래된다는 결론을 내렸습니다")
sys.stdout.flush()

# ONNX export
class Wrapper(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    def forward(self, x):
        e, _ = self.enc(audio_signal=x, length=torch.tensor([x.shape[2]]))
        return self.dec(encoder_output=e)

w = Wrapper(enc, dec).eval()
try:
    torch.onnx.export(w, mel, '/workspace/model/model_fixed_relpos.onnx',
        input_names=['audio_signal'], output_names=['logprobs'],
        opset_version=17, do_constant_folding=True)
    print("ONNX export OK!")

    import onnxruntime as ort
    sess = ort.InferenceSession('/workspace/model/model_fixed_relpos.onnx')
    out = sess.run(None, {'audio_signal': mel.numpy()})[0]
    am_onnx = np.argmax(out[0], axis=1)
    bl_onnx = np.sum(am_onnx == 5000)
    d_onnx = []
    prev = -1
    for tid in am_onnx:
        if tid != prev:
            if tid != 5000: d_onnx.append(int(tid))
            prev = tid
    t_onnx = ''
    for tid in d_onnx:
        tk = vocab[tid] if tid < len(vocab) else f'<{tid}>'
        t_onnx += (' ' + tk[1:]) if tk.startswith('\u2581') else tk
    print(f"ONNX:    blank={bl_onnx}/76: {t_onnx.strip()[:70]}")

    import onnx
    mo = onnx.load('/workspace/model/model_fixed_relpos.onnx')
    print(f"ONNX nodes: {len(mo.graph.node)}")
    where_c = sum(1 for n in mo.graph.node if n.op_type == 'Where')
    pad_c = sum(1 for n in mo.graph.node if n.op_type == 'Pad')
    print(f"Where: {where_c}, Pad: {pad_c}")
except Exception as e:
    print(f"ONNX FAIL: {e}")
    import traceback; traceback.print_exc()
