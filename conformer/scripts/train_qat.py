#!/usr/bin/env python3
"""
SungBeom Conformer CTC Medium — QAT (Quantization-Aware Training)

uint8 양자화 시뮬레이션을 학습에 삽입하여 양자화에 강건한 모델 학습.
FP32 CER 9.93% → uint8 CER 10.59% (0.66%p 손실) → QAT로 0%p에 가깝게.

Usage:
    # NeMo Docker에서 실행 (GPU 필요)
    docker run --gpus all --rm \
      -v $WORK:/workspace \
      nvcr.io/nvidia/nemo:23.06 \
      python3 /workspace/train_qat.py \
        --nemo-path /workspace/stt_kr_conformer_ctc_medium.1.nemo \
        --train-manifest /workspace/train_manifest.json \
        --val-manifest /workspace/val_manifest.json \
        --output-dir /workspace/qat_output \
        --epochs 10 --lr 1e-5 --batch-size 16
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver
import nemo.collections.asr as nemo_asr
from nemo.core.optim.lr_scheduler import CosineAnnealing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class FakeQuantizeFunction(torch.autograd.Function):
    """Per-tensor asymmetric uint8 fake quantization with STE."""
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmin, qmax = 0, 2**num_bits - 1
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(x, num_bits=8):
    """Apply fake quantization (forward: quantize+dequantize, backward: STE)."""
    if not x.requires_grad:
        return FakeQuantizeFunction.apply(x.float(), num_bits)
    return FakeQuantizeFunction.apply(x, num_bits)


class ConformerQATWrapper(pl.LightningModule):
    """
    NeMo Conformer CTC + FakeQuantize 삽입.

    FakeQuantize 위치:
    1. Encoder 입력 (mel spectrogram) — 입력 양자화 시뮬레이션
    2. Encoder 출력 — 중간 feature 양자화 시뮬레이션
    3. Decoder 출력 (logits) — 출력 양자화 시뮬레이션
    """
    def __init__(self, nemo_model, lr=1e-5, warmup_steps=500, num_bits=8):
        super().__init__()
        self.model = nemo_model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_bits = num_bits

        # Freeze CNN feature extractor (if exists) — 양자화 문제 없음
        # Conformer는 conv subsampling이 있음
        if hasattr(self.model.encoder, 'pre_encode'):
            for param in self.model.encoder.pre_encode.parameters():
                param.requires_grad = False

    def forward(self, input_signal, input_signal_length):
        # 1. Preprocessor
        processed_signal, processed_signal_length = self.model.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )

        # FQ on mel input
        processed_signal = fake_quantize(processed_signal, self.num_bits)

        # 2. Encoder
        encoded, encoded_len = self.model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        # FQ on encoder output
        encoded = fake_quantize(encoded, self.num_bits)

        # 3. Decoder
        log_probs = self.model.decoder(encoder_output=encoded)

        # FQ on logits (before log_softmax — decoder already applies log_softmax)
        # Actually decoder output is already log_softmax, so FQ here simulates output quantization
        log_probs = fake_quantize(log_probs, self.num_bits)

        return log_probs, encoded_len

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch

        log_probs, encoded_len = self.forward(signal, signal_len)

        # CTC Loss
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # [T, B, C]
            transcript,
            encoded_len,
            transcript_len,
            blank=self.model.decoder.num_classes_with_blank - 1,  # blank = last token
            reduction='mean',
            zero_infinity=True,
        )

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch

        log_probs, encoded_len = self.forward(signal, signal_len)

        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            transcript,
            encoded_len,
            transcript_len,
            blank=self.model.decoder.num_classes_with_blank - 1,
            reduction='mean',
            zero_infinity=True,
        )

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.lr * 0.01,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


class MarginQATWrapper(ConformerQATWrapper):
    """
    QAT + Margin Loss: CTC loss + margin penalty.

    Margin loss는 top1-top2 logit 차이를 최소 margin_target 이상으로 유지.
    → uint8 step size보다 margin이 커지도록 학습.
    """
    def __init__(self, nemo_model, lr=1e-5, warmup_steps=500, num_bits=8,
                 margin_target=0.3, margin_lambda=0.1):
        super().__init__(nemo_model, lr, warmup_steps, num_bits)
        self.margin_target = margin_target  # uint8 step ~0.2, 1.5배 여유
        self.margin_lambda = margin_lambda

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch

        log_probs, encoded_len = self.forward(signal, signal_len)

        # CTC Loss
        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            transcript,
            encoded_len,
            transcript_len,
            blank=self.model.decoder.num_classes_with_blank - 1,
            reduction='mean',
            zero_infinity=True,
        )

        # Margin Loss: top1 - top2 >= margin_target
        sorted_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
        margins = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]  # [B, T]
        margin_loss = F.relu(self.margin_target - margins).mean()

        loss = ctc_loss + self.margin_lambda * margin_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('ctc_loss', ctc_loss, prog_bar=True)
        self.log('margin_loss', margin_loss, prog_bar=True)
        self.log('margin_min', margins.min(), prog_bar=True)
        self.log('margin_mean', margins.mean(), prog_bar=True)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo-path', required=True, help='.nemo model path')
    parser.add_argument('--train-manifest', required=True, help='NeMo format train manifest JSON')
    parser.add_argument('--val-manifest', required=True, help='NeMo format val manifest JSON')
    parser.add_argument('--output-dir', default='./qat_output')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-margin-loss', action='store_true', help='Add margin loss to CTC')
    parser.add_argument('--margin-target', type=float, default=0.3)
    parser.add_argument('--margin-lambda', type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load NeMo model
    print(f"Loading model from {args.nemo_path}...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_path, map_location='cpu'
    )
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # Setup data
    model.setup_training_data(
        train_data_config={
            'manifest_filepath': args.train_manifest,
            'sample_rate': 16000,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'shuffle': True,
            'max_duration': 20.0,
            'min_duration': 0.5,
        }
    )
    model.setup_validation_data(
        val_data_config={
            'manifest_filepath': args.val_manifest,
            'sample_rate': 16000,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'shuffle': False,
        }
    )

    # Wrap with QAT
    if args.use_margin_loss:
        print(f"Using MarginQAT (margin_target={args.margin_target}, lambda={args.margin_lambda})")
        qat_model = MarginQATWrapper(
            model, lr=args.lr,
            margin_target=args.margin_target,
            margin_lambda=args.margin_lambda,
        )
    else:
        print("Using standard QAT (FakeQuantize only)")
        qat_model = ConformerQATWrapper(model, lr=args.lr)

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='conformer_qat_{epoch:02d}_{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        val_check_interval=0.5,  # validate every half epoch
        gradient_clip_val=1.0,
        precision=32,  # FP32 (fake quantize는 fp16과 비호환)
    )

    print(f"Starting QAT training: {args.epochs} epochs, lr={args.lr}")
    trainer.fit(qat_model, model.train_dataloader(), model.val_dataloader())

    # Save final model (strip FakeQuantize, save as standard NeMo)
    print("Saving QAT model...")
    model.save_to(os.path.join(args.output_dir, 'conformer_qat_final.nemo'))

    # Also export ONNX
    model.export(os.path.join(args.output_dir, 'conformer_qat_final.onnx'))
    print(f"Done! Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
