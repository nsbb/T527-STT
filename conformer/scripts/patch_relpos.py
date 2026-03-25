import nemo.collections.asr as nemo_asr
import torch, torch.nn as nn, numpy as np, soundfile as sf, tarfile, sys

m = nemo_asr.models.EncDecCTCModelBPE.restore_from('/workspace/model/stt_kr_conformer_ctc_medium.nemo', map_location='cpu')
m.eval()
enc, dec = m.encoder, m.decoder

t = tarfile.open('/workspace/model/stt_kr_conformer_ctc_medium.nemo')
vf = [f for f in t.getnames() if 'vocab.txt' in f][0]
vocab = t.extractfile(vf).read().decode().strip().split('\n')

SEQ_LEN = 76

class FixedRelPosAttn(nn.Module):
    def __init__(self, orig, seq_len):
        super().__init__()
        self.linear_q = orig.linear_q
        self.linear_k = orig.linear_k
        self.linear_v = orig.linear_v
        self.linear_out = orig.linear_out
        self.linear_pos = orig.linear_pos
        self.pos_bias_u = orig.pos_bias_u
        self.pos_bias_v = orig.pos_bias_v
        self.h = orig.h
        self.d_k = orig.d_k
        n = seq_len
        idx = torch.zeros(n, n, dtype=torch.long)
        for i in range(n):
            for j in range(n):
                idx[i, j] = n - 1 - i + j
        self.register_buffer('rel_idx', idx)

    def forward(self, query, key, value, pos_emb, mask=None, cache=None):
        B = query.size(0)
        q = self.linear_q(query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(B, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(B, -1, self.h, self.d_k).transpose(1, 2)

        qu = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        qv = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        ac = torch.matmul(qu, k.transpose(-2, -1))
        bd_full = torch.matmul(qv, p.transpose(-2, -1))
        idx = self.rel_idx.unsqueeze(0).unsqueeze(0).expand(B, self.h, -1, -1)
        bd = torch.gather(bd_full, 3, idx)

        scores = (ac + bd) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linear_out(out)

for i, layer in enumerate(enc.layers):
    layer.self_attn = FixedRelPosAttn(layer.self_attn, SEQ_LEN)
print(f"Patched {len(enc.layers)} layers")

# Test
audio, sr = sf.read('/workspace/testset/ko_test_0004.wav', dtype='float32')
audio = audio[:48160]
if len(audio) < 48160: audio = np.pad(audio, (0, 48160 - len(audio)))
mel_t = torch.tensor(audio).unsqueeze(0)
with torch.no_grad():
    mel, _ = m.preprocessor(input_signal=mel_t, length=torch.tensor([mel_t.shape[1]]))
    if mel.shape[2] > 301: mel = mel[:, :, :301]
    elif mel.shape[2] < 301: mel = nn.functional.pad(mel, (0, 301 - mel.shape[2]))

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
    argmax2 = np.argmax(out[0], axis=1)
    blank2 = np.sum(argmax2 == 5000)
    decoded2 = []
    prev = -1
    for tid in argmax2:
        if tid != prev:
            if tid != 5000: decoded2.append(int(tid))
            prev = tid
    text2 = ''
    for tid in decoded2:
        tk = vocab[tid] if tid < len(vocab) else f'<{tid}>'
        text2 += (' ' + tk[1:]) if tk.startswith('\u2581') else tk
    print(f"ONNX:    blank={blank2}/76: {text2.strip()[:70]}")
except Exception as e:
    print(f"ONNX FAIL: {e}")
    import traceback; traceback.print_exc()
