import os
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")


class AudioSRDiagnostics:
    """
    Diagnostic visual and audio utilities tailored for probabilistic 
    Audio Super-Resolution (AudioSR) architectures (e.g., Probabilistic LISA).
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    # -------------------------------------------------------------------------
    # Helper Transformations
    # -------------------------------------------------------------------------
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts PyTorch tensor to 1D or 2D numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().squeeze().numpy()

    def _compute_stft_db(self, audio: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes STFT magnitude in dB scale."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        stft_complex = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            return_complex=True
        )
        mag = torch.abs(stft_complex).squeeze(0).cpu().numpy()
        mag_db = 20 * np.log10(np.clip(mag, a_min=1e-8, a_max=None))
        
        freqs = np.linspace(0, self.sample_rate / 2, mag_db.shape[0])
        times = np.linspace(0, audio.shape[-1] / self.sample_rate, mag_db.shape[1])
        return mag_db, freqs, times

    def _save_wav(self, file_path: str, audio_tensor: torch.Tensor):
        """Helper to safely save audio using soundfile without relying on torchcodec/FFmpeg."""
        audio_np = self._to_numpy(audio_tensor)
        sf.write(file_path, audio_np, self.sample_rate)

    # -------------------------------------------------------------------------
    # 1. Waveform Ensemble & Uncertainty Plot
    # -------------------------------------------------------------------------
    def plot_waveform_ensemble(
        self,
        hr_audio: torch.Tensor,
        lr_audio: torch.Tensor,
        sr_ensemble: torch.Tensor,
        zoom_ms: Optional[Tuple[float, float]] = (100.0, 150.0),
        title: str = "Waveform Uncertainty & Transient Analysis"
    ) -> plt.Figure:
        hr = self._to_numpy(hr_audio)
        lr = self._to_numpy(lr_audio)
        ensemble = self._to_numpy(sr_ensemble) # [M, L]
        
        M, L = ensemble.shape
        time = np.linspace(0, L / self.sample_rate, L) * 1000.0 # ms
        
        mean_pred = np.mean(ensemble, axis=0)
        std_pred = np.std(ensemble, axis=0)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        
        # Panel 1: Full Waveform
        axes[0].plot(time, hr, color="black", alpha=0.6, label="Ground Truth (HR)", linewidth=1.0)
        axes[0].plot(time, mean_pred, color="crimson", alpha=0.8, label=r"Ensemble Mean ($\mu$)", linewidth=0.8)
        axes[0].fill_between(time, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                             color="crimson", alpha=0.25, label=r"Uncertainty ($\pm 2\sigma$)")
        axes[0].set_title(f"{title} - Full Overview")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Transient Zoom-In
        if zoom_ms is not None:
            mask = (time >= zoom_ms[0]) & (time <= zoom_ms[1])
            axes[1].plot(time[mask], hr[mask], color="black", linewidth=1.5, label="Ground Truth (HR)")
            axes[1].plot(time[mask], lr[mask], color="gray", linestyle=":", linewidth=1.2, label="Low Res (LR)")
            
            for m in range(min(M, 5)):
                axes[1].plot(time[mask], ensemble[m, mask], alpha=0.4, linewidth=0.8, label=f"Sample {m+1}" if m==0 else "")
            
            axes[1].plot(time[mask], mean_pred[mask], color="crimson", linewidth=1.2, label=r"Ensemble Mean ($\mu$)")
            axes[1].fill_between(time[mask], mean_pred[mask] - 2*std_pred[mask], mean_pred[mask] + 2*std_pred[mask], 
                                 color="crimson", alpha=0.2, label=r"$\pm 2\sigma$ Band")
            
            axes[1].set_title(f"Transient Zoom [{zoom_ms[0]}ms - {zoom_ms[1]}ms]")
            axes[1].set_xlabel("Time (ms)")
            axes[1].set_ylabel("Amplitude")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 2. Spectrogram & High-Frequency Cutoff Error Maps
    # -------------------------------------------------------------------------
    def plot_spectrogram_diagnostics(
        self,
        hr_audio: torch.Tensor,
        lr_audio: torch.Tensor,
        sr_ensemble: torch.Tensor,
        cutoff_freq: float = 4000.0
    ) -> plt.Figure:
        hr_db, freqs, times = self._compute_stft_db(hr_audio)
        lr_db, _, _ = self._compute_stft_db(lr_audio)
        
        ensemble_mags = []
        for m in range(sr_ensemble.shape[0]):
            mag_db, _, _ = self._compute_stft_db(sr_ensemble[m])
            ensemble_mags.append(mag_db)
        
        ensemble_mags = np.stack(ensemble_mags, axis=0) # [M, F, T]
        mean_db = np.mean(ensemble_mags, axis=0)
        var_db = np.var(ensemble_mags, axis=0)
        error_db = np.abs(hr_db - mean_db)

        fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        
        plots = [
            (hr_db, "Ground Truth (HR)", "magma", None),
            (lr_db, "Low-Resolution Input (LR)", "magma", None),
            (mean_db, r"Generated Ensemble Mean ($\mu_{SR}$)", "magma", None),
            (error_db, r"High-Frequency Absolute Error ($|STFT_{HR} - STFT_{\mu}|$)", "viridis", (0, 30)),
            (var_db, "Epistemic Uncertainty / STFT Variance Across Samples", "inferno", None)
        ]

        for idx, (data, title, cmap, vlim) in enumerate(plots):
            ax = axes[idx]
            vmin, vmax = vlim if vlim else (data.min(), data.max())
            im = ax.pcolormesh(times, freqs / 1000.0, data, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
            
            ax.axhline(cutoff_freq / 1000.0, color="cyan", linestyle="--", linewidth=1.2, label=f"Cutoff ({cutoff_freq/1000:.1f} kHz)")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel("Freq (kHz)")
            fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
            if idx == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 3. High-Frequency Artifact & Spectral Energy Diagnostics
    # -------------------------------------------------------------------------
    def plot_high_frequency_artifacts(
        self,
        hr_audio: torch.Tensor,
        sr_ensemble: torch.Tensor,
        cutoff_freq: float = 4000.0
    ) -> plt.Figure:
        hr_db, freqs, times = self._compute_stft_db(hr_audio)
        
        ensemble_dbs = []
        for m in range(sr_ensemble.shape[0]):
            db, _, _ = self._compute_stft_db(sr_ensemble[m])
            ensemble_dbs.append(db)
        
        ensemble_dbs = np.stack(ensemble_dbs, axis=0) # [M, F, T]
        mean_db = np.mean(ensemble_dbs, axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # 1. Average PSD Roll-Off
        hr_psd = np.mean(hr_db, axis=1)
        mean_psd = np.mean(mean_db, axis=1)
        
        axes[0].plot(freqs / 1000.0, hr_psd, color="black", label="Ground Truth (HR)")
        axes[0].plot(freqs / 1000.0, mean_psd, color="crimson", label="Generated Mean")
        for m in range(min(sr_ensemble.shape[0], 5)):
            axes[0].plot(freqs / 1000.0, np.mean(ensemble_dbs[m], axis=1), alpha=0.3, linewidth=0.8)
        
        axes[0].axvline(cutoff_freq / 1000.0, color="blue", linestyle="--", label="Cutoff Frequency")
        axes[0].set_title("Average Power Spectral Density (PSD)")
        axes[0].set_xlabel("Frequency (kHz)")
        axes[0].set_ylabel("Power (dB)")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. High-Band Energy Envelope
        high_band_mask = freqs >= cutoff_freq
        hr_hb_energy = np.sum(10**(hr_db[high_band_mask, :] / 20), axis=0)
        mean_hb_energy = np.sum(10**(mean_db[high_band_mask, :] / 20), axis=0)

        axes[1].plot(times, hr_hb_energy, color="black", alpha=0.7, label="HR High Band")
        axes[1].plot(times, mean_hb_energy, color="crimson", alpha=0.8, label="Generated High Band")
        axes[1].set_title(f"High-Band Energy Envelope (>{cutoff_freq/1000:.1f} kHz)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Linear Energy Sum")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Spectral Kurtosis in High Band
        high_band_mags = 10**(ensemble_dbs[:, high_band_mask, :] / 20)
        kurtosis = np.mean((high_band_mags - np.mean(high_band_mags, axis=0))**4, axis=0) / (np.var(high_band_mags, axis=0)**2 + 1e-8)
        avg_kurtosis_time = np.mean(kurtosis, axis=0)

        axes[2].plot(times, avg_kurtosis_time, color="teal", label="Spectral Kurtosis")
        axes[2].set_title("High-Band Ensemble Kurtosis (Artifact Detector)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Kurtosis Index")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 4. Probabilistic Calibration Plots (PIT & CRPS)
    # -------------------------------------------------------------------------
    def plot_probabilistic_calibration(
        self,
        hr_audio: torch.Tensor,
        sr_ensemble: torch.Tensor,
        pit_values: np.ndarray,
        crps_score: float
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 1. PIT Histogram
        axes[0].hist(pit_values.flatten(), bins=20, density=True, color="skyblue", edgecolor="black", alpha=0.7)
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Ideal Uniform U(0,1)")
        axes[0].set_title("Probability Integral Transform (PIT) Histogram")
        axes[0].set_xlabel("Cumulative Probability (u)")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        var_dev = np.var(pit_values) - (1.0 / 12.0)
        shape_status = "Under-dispersed (Mean Regressed)" if var_dev > 0.02 else "Calibrated" if abs(var_dev) <= 0.02 else "Over-dispersed (Too Noisy)"
        axes[0].text(0.05, 0.85, f"Status: {shape_status}\nVar Dev: {var_dev:+.4f}", 
                     transform=axes[0].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # 2. Time-Domain Pointwise CRPS Error
        hr = self._to_numpy(hr_audio)
        ensemble = self._to_numpy(sr_ensemble) # [M, L]
        
        term1 = np.mean(np.abs(ensemble - hr), axis=0)
        diffs = np.abs(ensemble[:, None, :] - ensemble[None, :, :])
        term2 = np.mean(diffs, axis=(0, 1)) / 2.0
        frame_crps = term1 - term2
        
        time = np.linspace(0, len(hr) / self.sample_rate, len(hr))
        axes[1].plot(time, frame_crps, color="darkviolet", alpha=0.7, label=f"CRPS (Mean: {crps_score:.4f})")
        axes[1].set_title("Time-Resolved Frame CRPS")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("CRPS Loss")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 5. Export Master Artifact Package
    # -------------------------------------------------------------------------
    def export_evaluation_artifacts(
        self,
        hr_audio: torch.Tensor,
        lr_audio: torch.Tensor,
        sr_ensemble: torch.Tensor,
        output_dir: str,
        sample_name: str = "sample_001",
        cutoff_freq: float = 4000.0,
        crps_score: float = 0.0,
        pit_values: Optional[np.ndarray] = None
    ) -> None:
        """
        Exports all WAV audio files and saves diagnostic PNG figures to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save Audio WAVs (Using self._save_wav / soundfile backend)
        self._save_wav(os.path.join(output_dir, f"{sample_name}_HR.wav"), hr_audio)
        self._save_wav(os.path.join(output_dir, f"{sample_name}_LR.wav"), lr_audio)
        
        mean_audio = sr_ensemble.mean(dim=0)
        self._save_wav(os.path.join(output_dir, f"{sample_name}_Pred_Mean.wav"), mean_audio)
        
        for m in range(min(sr_ensemble.shape[0], 3)):
            self._save_wav(os.path.join(output_dir, f"{sample_name}_Sample_{m+1}.wav"), sr_ensemble[m])

        # 2. Render & Save Plots
        fig_wave = self.plot_waveform_ensemble(hr_audio, lr_audio, sr_ensemble)
        fig_wave.savefig(os.path.join(output_dir, f"{sample_name}_waveform_ensemble.png"), dpi=200)
        plt.close(fig_wave)

        fig_spec = self.plot_spectrogram_diagnostics(hr_audio, lr_audio, sr_ensemble, cutoff_freq)
        fig_spec.savefig(os.path.join(output_dir, f"{sample_name}_spectrogram_diagnostics.png"), dpi=200)
        plt.close(fig_spec)

        fig_artifacts = self.plot_high_frequency_artifacts(hr_audio, sr_ensemble, cutoff_freq)
        fig_artifacts.savefig(os.path.join(output_dir, f"{sample_name}_hf_artifacts.png"), dpi=200)
        plt.close(fig_artifacts)

        if pit_values is not None:
            fig_calib = self.plot_probabilistic_calibration(hr_audio, sr_ensemble, pit_values, crps_score)
            fig_calib.savefig(os.path.join(output_dir, f"{sample_name}_calibration.png"), dpi=200)
            plt.close(fig_calib)

        print(f"Artifacts successfully exported to: {output_dir}")


if __name__ == "__main__":
    torch.manual_seed(42)
    sample_rate = 16000
    L = 16000 # 1 second
    M = 8     # Ensemble size
    cutoff_freq = 4000.0

    hr_audio = torch.randn(L)
    lr_audio = F.lowpass_biquad(hr_audio, sample_rate, cutoff_freq)
    
    sr_ensemble = hr_audio.unsqueeze(0).repeat(M, 1) + torch.randn(M, L) * 0.15

    diagnostics = AudioSRDiagnostics(sample_rate=sample_rate)
    u = (sr_ensemble <= hr_audio.unsqueeze(0)).float().cpu().numpy()

    diagnostics.export_evaluation_artifacts(
        hr_audio=hr_audio,
        lr_audio=lr_audio,
        sr_ensemble=sr_ensemble,
        output_dir="./eval_output",
        sample_name="test_speech_01",
        cutoff_freq=cutoff_freq,
        crps_score=0.452,
        pit_values=u
    )