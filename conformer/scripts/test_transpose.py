import nemo.collections.asr as nemo_asr
import torch, soundfile as sf, numpy as np, tarfile

m = nemo_asr.models.EncDecCTCModelBPE.restore_from("/workspace/model/stt_kr_conformer_ctc_medium.nemo", map_location="cpu")
m.eval()
m.preprocessor.featurizer.dither = 0.0
m.preprocessor.featurizer.pad_to = 0

tf = tarfile.open("/workspace/model/stt_kr_conformer_ctc_medium.nemo")
vf = [f for f in tf.getnames() if "vocab.txt" in f][0]
vocab = tf.extractfile(vf).read().decode().strip().split("\n")

audio, sr = sf.read("/workspace/testset/ko_test_0004.wav", dtype="float32")
at = torch.tensor(audio).unsqueeze(0)

with torch.no_grad():
    mel, ml = m.preprocessor(input_signal=at, length=torch.tensor([len(audio)]))
    enc, el = m.encoder(audio_signal=mel, length=ml)
    print(f"enc: {enc.shape}")

    # Test both: [B,D,T] and [B,T,D]
    for label, e in [("no_transpose", enc), ("transposed", enc.transpose(1,2))]:
        lp = m.decoder(encoder_output=e)
        argmax = lp[0].argmax(dim=-1)
        decoded = []
        prev = -1
        for tid in argmax[:el[0]]:
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
        print(f"{label}: {text.strip()[:80]}")

print("GT: 평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다")
