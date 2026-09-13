from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_audio_dataloaders
from models import LISA


# -----------------------------------------------------------------------------
# 1. Loss Functions & Metrics Helpers
# -----------------------------------------------------------------------------

class MultiResolutionSTFTLoss(nn.Module):
    """
    Optimized Multi-Resolution STFT Loss.
    Registers window buffers on GPU automatically to eliminate per-batch I/O overhead.
    """

    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        for i, w in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(w))

    def stft_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_stft = torch.abs(
            torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
        )
        y_stft = torch.abs(
            torch.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
        )

        sc_loss = torch.norm(y_stft - x_stft, p="fro") / (torch.norm(y_stft, p="fro") + 1e-8)
        log_mag_loss = F.l1_loss(torch.log(x_stft + 1e-8), torch.log(y_stft + 1e-8))

        return sc_loss, log_mag_loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        total_sc_loss = 0.0
        total_mag_loss = 0.0

        for i, (n_fft, hop, win) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")
            sc_l, mag_l = self.stft_loss(x, y, n_fft, hop, win, window)
            total_sc_loss += sc_l
            total_mag_loss += mag_l

        return (total_sc_loss / len(self.fft_sizes)) + (total_mag_loss / len(self.fft_sizes))


def compute_si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    alpha = torch.sum(pred * target, dim=-1, keepdim=True) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target
    noise = pred - target_scaled

    val = 10 * torch.log10(
        (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return val.mean().item()


# -----------------------------------------------------------------------------
# 2. Pipeline Verification Diagnostic
# -----------------------------------------------------------------------------

@torch.no_grad()
def verify_model_pipeline(
    model: nn.Module,
    dataloader: DataLoader,
    scale: float = 2.0,
    device: Optional[str] = None,
) -> bool:
    """
    Dry-run verification tool that inspects tensor dimensions across DataLoader,
    Model forward pass, Tensor alignment, and Loss computation.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================ Verification Check ({device}) ================")

    model = model.to(device)
    model.eval()

    # 1. Fetch Single Batch
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"❌ ERROR: Failed to load batch from DataLoader: {e}")
        return False

    if isinstance(batch, dict):
        lr_audio = batch["lr"].to(device)
        hr_audio = batch["hr"].to(device)
    else:
        lr_audio, hr_audio = batch[0].to(device), batch[1].to(device)

    print(f"[Input]  Low-Res (LR) Shape  : {list(lr_audio.shape)}")
    print(f"[Target] High-Res (HR) Shape : {list(hr_audio.shape)}")

    # 2. Forward Pass Test with scale argument
    try:
        sr_audio = model(lr_audio, scale)
    except Exception as e:
        print(f"❌ ERROR: Model forward pass failed: {e}")
        return False

    print(f"[Output] Super-Res (SR) Shape : {list(sr_audio.shape)}")

    # 3. Batch Size Check
    if sr_audio.shape[0] != lr_audio.shape[0]:
        print(f"❌ ERROR: Batch size mismatch! Input was {lr_audio.shape[0]}, but output is {sr_audio.shape[0]}")
        return False

    # 4. Temporal Sample Length Check
    expected_samples = hr_audio.shape[-1]
    actual_samples = sr_audio.shape[-1]

    if actual_samples != expected_samples:
        print(
            f"⚠️  NOTICE: Output sample length ({actual_samples}) does not match ground truth ({expected_samples}).\n"
            f"   Automatic cropping/padding via `_align_tensors()` will handle this during training."
        )
    else:
        print("✅ Sample length matches ground truth target exactly!")

    # 5. Dimension & Loss Computation Check
    try:
        if sr_audio.dim() == 2:
            sr_audio = sr_audio.unsqueeze(1)
        if hr_audio.dim() == 2:
            hr_audio = hr_audio.unsqueeze(1)

        target_len = hr_audio.shape[-1]
        curr_len = sr_audio.shape[-1]
        if curr_len > target_len:
            sr_audio = sr_audio[..., :target_len]
        elif curr_len < target_len:
            sr_audio = F.pad(sr_audio, (0, target_len - curr_len))

        l1_fn = nn.L1Loss()
        stft_fn = MultiResolutionSTFTLoss().to(device)

        l1_val = l1_fn(sr_audio, hr_audio)
        stft_val = stft_fn(sr_audio, hr_audio)
        total_loss = l1_val + 2.5 * stft_val

        print(
            f"✅ Loss Execution Successful! Test Loss: {total_loss.item():.4f} "
            f"(L1: {l1_val.item():.4f}, STFT: {stft_val.item():.4f})"
        )

    except Exception as e:
        print(f"❌ ERROR: Loss computation failed: {e}")
        return False

    print("================ Verification Passed! Ready for Training ================\n")
    return True


# -----------------------------------------------------------------------------
# 3. Main Trainer Class
# -----------------------------------------------------------------------------

class Trainer:
    """
    Modular PyTorch Trainer managing restoration, training loop, loss computation,
    validation metrics, and auto-checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scale: float = 2.0,
        lr: float = 2e-4,
        epochs: int = 100,
        save_dir: str = "saved_models",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scale = scale
        self.epochs = epochs

        self.model_name = getattr(model, "name", model.__class__.__name__)
        self.checkpoint_dir = Path(save_dir) / self.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.l1_loss_fn = nn.L1Loss()
        self.stft_loss_fn = MultiResolutionSTFTLoss().to(self.device)

        self.start_epoch = 1
        self.best_val_loss = float("inf")

        self._auto_resume_checkpoint()

    def _align_tensors(
        self, sr_audio: torch.Tensor, hr_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensures prediction and target match exact dimensions [B, 1, T] and sample length."""
        if sr_audio.dim() == 2:
            sr_audio = sr_audio.unsqueeze(1)
        if hr_audio.dim() == 2:
            hr_audio = hr_audio.unsqueeze(1)

        target_len = hr_audio.shape[-1]
        curr_len = sr_audio.shape[-1]

        if curr_len > target_len:
            sr_audio = sr_audio[..., :target_len]
        elif curr_len < target_len:
            sr_audio = F.pad(sr_audio, (0, target_len - curr_len))

        return sr_audio, hr_audio

    def _auto_resume_checkpoint(self) -> None:
        latest_ckpt_path = self.checkpoint_dir / "checkpoint_latest.pt"

        if latest_ckpt_path.exists():
            print(f"\n[Trainer] Found existing checkpoint at: {latest_ckpt_path}")
            print(f"[Trainer] Restoring model, optimizer, and scheduler state...")

            checkpoint = torch.load(latest_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"[Trainer] Resuming training starting from Epoch {self.start_epoch}\n")
        else:
            print(f"\n[Trainer] No prior checkpoint found in {self.checkpoint_dir}")
            print(f"[Trainer] Initializing new training run from Epoch 1\n")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "model_name": self.model_name,
        }

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint_data, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint_data, best_path)
            print(f" -> Saved new BEST checkpoint to: {best_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_stft = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)

        for batch in pbar:
            if isinstance(batch, dict):
                lr_audio = batch["lr"].to(self.device)
                hr_audio = batch["hr"].to(self.device)
            else:
                lr_audio, hr_audio = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass with scale argument
            sr_audio = self.model(lr_audio, self.scale)

            # Apply Shape & Temporal Length Safeguards
            sr_audio, hr_audio = self._align_tensors(sr_audio, hr_audio)

            # Compute Losses
            l1_loss = self.l1_loss_fn(sr_audio, hr_audio)
            stft_loss = self.stft_loss_fn(sr_audio, hr_audio)

            loss = l1_loss + 2.5 * stft_loss

            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L1": f"{l1_loss.item():.4f}",
                    "STFT": f"{stft_loss.item():.4f}",
                }
            )

        num_batches = len(self.train_loader)
        return {
            "loss": total_loss / num_batches,
            "l1_loss": total_l1 / num_batches,
            "stft_loss": total_stft / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_val_loss = 0.0
        total_si_sdr = 0.0

        pbar = tqdm(self.val_loader, desc="[Validation]", leave=False)

        for batch in pbar:
            if isinstance(batch, dict):
                lr_audio = batch["lr"].to(self.device)
                hr_audio = batch["hr"].to(self.device)
            else:
                lr_audio, hr_audio = batch[0].to(self.device), batch[1].to(self.device)

            # Forward Pass with scale argument
            sr_audio = self.model(lr_audio, self.scale)

            # Apply Shape & Temporal Length Safeguards
            sr_audio, hr_audio = self._align_tensors(sr_audio, hr_audio)

            l1_l = self.l1_loss_fn(sr_audio, hr_audio)
            stft_l = self.stft_loss_fn(sr_audio, hr_audio)
            val_loss = l1_l + 2.5 * stft_l

            si_sdr_val = compute_si_sdr(sr_audio, hr_audio)

            total_val_loss += val_loss.item()
            total_si_sdr += si_sdr_val

        num_batches = len(self.val_loader)
        return {
            "val_loss": total_val_loss / num_batches,
            "val_si_sdr": total_si_sdr / num_batches,
        }

    def train(self) -> None:
        if self.start_epoch > self.epochs:
            print(f"[Trainer] Model is already trained to Epoch {self.start_epoch - 1}. Exiting.")
            return

        print(f"=== Starting Training Loop: '{self.model_name}' on {self.device} ===")

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            current_val_loss = val_metrics.get("val_loss", train_metrics["loss"])

            if self.scheduler:
                self.scheduler.step(current_val_loss)

            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss

            self._save_checkpoint(epoch=epoch, is_best=is_best)

            log_str = (
                f"Epoch [{epoch}/{self.epochs}] | "
                f"Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1_loss']:.4f}, STFT: {train_metrics['stft_loss']:.4f})"
            )
            if val_metrics:
                log_str += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val SI-SDR: {val_metrics['val_si_sdr']:.2f} dB"
                )

            print(log_str)

        print(f"\n=== Training Complete for '{self.model_name}' ===")


# -----------------------------------------------------------------------------
# 4. Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    lsr = 8000
    hsr = 16000
    scale = float(hsr / lsr)  # 16000 / 8000 = 2.0

    # 1. Instantiate DataLoaders
    train_loader, val_loader, _ = get_audio_dataloaders(
        data_path="./data",
        batch_size=8,
        num_workers=4,
        lsr=lsr,
        hsr=hsr,
    )

    # 2. Instantiate Model from models.py
    model = LISA()
    model.name = "ProbabilisticLISA_v1"

    # 3. Pipeline Verification with scale
    is_valid = verify_model_pipeline(model, train_loader, scale=scale)

    # 4. Execute Training Loop
    if is_valid:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scale=scale,
            lr=2e-4,
            epochs=50,
            save_dir="./saved_models",
        )
        trainer.train()
    else:
        print("Training aborted due to pipeline verification errors.")