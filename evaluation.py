import numpy as np
import torchaudio.functional as F
from visqol import VisqolApi
import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

class Evaluator: 
    def __init__(self):
        self.target_sr = 16000 
        mode = "speech" 
        self.visqol_api = VisqolApi()
        self.visqol_api.create(mode=mode)
        self.pesq_api = PerceptualEvaluationSpeechQuality(self.target_sr , 'wb')
    
    def match_length(self, ref, sr_pred):
        # used in case the lenghts of the degraded and source are not of same length
        min_len = min(len(ref), len(sr_pred))
        return ref[:min_len], sr_pred[:min_len]

    def sample_to_correct_rate(self, hr_audio, sr_audio, current_sr):
        if current_sr != self.target_sr:
            # we resample to 16k htz:
            hr_target = F.resample(hr_audio, orig_freq=current_sr, new_freq=self.target_sr)
            sr_target = F.resample(sr_audio, orig_freq=current_sr, new_freq=self.target_sr)
        else:
            hr_target = hr_audio
            sr_target = sr_audio
        return hr_target, sr_target

    def evaluate_pesq(self, hr_audio, sr_audio, current_sr):
        score = []
        for hr_target, sr_target  in zip(hr_audio, sr_audio):
            hr_target , sr_target = self.sample_to_correct_rate(hr_target, sr_target, current_sr)
            hr_target, sr_target = self.match_length(hr_target, sr_target)
            score.append(self.pesq_api(sr_target.cpu(), hr_target.cpu()))
        return sum(score)/len(score)

    def evaluate_visqol(self, hr_audio, sr_audio, current_sr):
        """
        this function takes in a single sample of audio and returns the score for that specific
        sample only. 
        hr_audio: torch_tensor [strictly 1 dimension so unsqueeze ](can exist on gpu)-> high resolution sample(single sample)
        sr_audio: torch_tensor -> super resolution sample
        current_sr : > the resolution of the hr_audio and sr_audio that we feed

        note: the evaluation metric works only best at 16k htz, which means that
        we need to resample to 16k htz if current_sr != 16k htz
        """
        visqol = []
        for hr_target , sr_target in zip(hr_audio, sr_audio):
            hr_target, sr_target = self.sample_to_correct_rate(hr_target, sr_target, current_sr)

            hr_target, sr_target = self.match_length(hr_target, sr_target)

            hr_target = hr_target.detach().cpu().numpy().astype(np.float64)
            sr_target = sr_target.detach().cpu().numpy().astype(np.float64)
            
            try:
                similarity_result = self.visqol_api.measure_from_arrays(hr_target, sr_target, sample_rate=self.target_sr)
                visqol.append(similarity_result.moslqo)
            except Exception as e:
                print("error in measuring visqol")
                visqol.append(float('nan'))
        return sum(visqol)/ len(visqol)

    def evaluate_crps(self, hr_audio, sr_audio_ensemble):
        """
        Computes the Continuous Ranked Probability Score (CRPS) over an ensemble.
        Lower is better.
        
        hr_audio: Tensor of shape [Batch, Length]
        sr_audio_ensemble: Tensor of shape [Batch, M, Length], where M is ensemble size.
                           (If [Batch, Length] is passed, it acts as MAE).
        """
        # Handle deterministic fallback (unsqueeze to add ensemble dimension M=1)
        if sr_audio_ensemble.dim() == 2:
            sr_audio_ensemble = sr_audio_ensemble.unsqueeze(1)
            
        hr_audio, sr_audio_ensemble = self.match_length(hr_audio, sr_audio_ensemble)
        
        # Ensure shapes align for broadcasting
        B, M, L = sr_audio_ensemble.shape
        hr_audio = hr_audio.unsqueeze(1) # [B, 1, L]
        
        # Term 1: E_F|Y - y| -> Mean absolute error between ensemble and ground truth
        term1 = torch.abs(sr_audio_ensemble - hr_audio).mean(dim=1) # [B, L]
        
        # Term 2: 1/2 * E_F|Y - Y'| -> Expected difference between independent ensemble samples
        # We broadcast to compute pairwise differences: [B, M, 1, L] - [B, 1, M, L]
        diffs = torch.abs(sr_audio_ensemble.unsqueeze(2) - sr_audio_ensemble.unsqueeze(1))
        term2 = diffs.mean(dim=(1, 2)) / 2.0 # [B, L]
        
        crps = term1 - term2 # [B, L]
        
        # Return the mean CRPS across the batch and time length
        return crps.mean().item()

    def evaluate_pit(self, hr_audio, sr_audio_ensemble):
        """
        Computes the Probability Integral Transform (PIT) Variance Deviation and KS-Stat.
        Returns a dictionary. A perfectly calibrated model has Variance Deviation close to 0.0.
        
        hr_audio: Tensor of shape [Batch, Length]
        sr_audio_ensemble: Tensor of shape [Batch, M, Length] (M must be > 1)
        """
        if sr_audio_ensemble.dim() == 2 or sr_audio_ensemble.shape[1] < 2:
            raise ValueError("PIT requires a probabilistic ensemble. sr_audio must be shape [Batch, M, Length] with M > 1.")
            
        hr_audio, sr_audio_ensemble = self.match_length(hr_audio, sr_audio_ensemble)
        hr_audio = hr_audio.unsqueeze(1) # [B, 1, L]
        
        # 1. Calculate PIT values (u): the proportion of generated samples <= ground truth
        # u will be a tensor of shape [B, L] with values between 0.0 and 1.0
        u = (sr_audio_ensemble <= hr_audio).float().mean(dim=1)
        
        # 2. Variance Deviation
        # A perfectly calibrated uniform distribution has a variance of 1/12 (~0.0833)
        u_var = torch.var(u, unbiased=False)
        target_var = 1.0 / 12.0
        var_deviation = (u_var - target_var).item()
        
        # 3. Kolmogorov-Smirnov (KS) Distance from Uniform Distribution
        # Flattens all u values, sorts them, and checks max deviation from a true uniform CDF
        u_sorted = torch.sort(u.flatten())[0]
        n = u_sorted.numel()
        cdf_uniform = torch.linspace(0, 1, steps=n, device=u.device)
        ks_stat = torch.max(torch.abs(u_sorted - cdf_uniform)).item()
        
        return {
            "pit_var_deviation": var_deviation,
            "pit_ks_stat": ks_stat
        }
    


if __name__ == "__main__":
    # Example usage directly from a PyTorch training/eval loop
    evaluator = Evaluator() 
    
    # Simulating the ground truth and model output tensors (e.g., at 8kHz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_hr = []
    dummy_sr = []
    for i in range(1, 4):
        dummy_hr_target = torch.randn(80000, device=device) # 10 seconds of 8kHz audio
        dummy_sr_pred = torch.randn(8000, device=device)  # Slightly longer due to padding
        dummy_hr.append(dummy_hr_target)
        dummy_sr.append(dummy_sr_pred)

    
    # Pass the tensors directly
    score = evaluator.evaluate_visqol(dummy_hr, dummy_sr, current_sr=8000)
    
    if score is not None:
        print(f"ViSQOL MOS-LQO Score: {score:.3f}")

    dummy_hr_batch = torch.stack(dummy_hr)
    dummy_sr_batch = torch.stack(dummy_sr)
    pesq_score = evaluator.evaluate_pesq(dummy_hr_batch, dummy_sr_batch, current_sr = 8000)
    print(f"the pesq score is :{pesq_score}")


    torch.manual_seed(42)  # For reproducibility
    evaluator = Evaluator()
    
    # 2. Define tensor dimensions
    B = 2      # Batch size
    M = 10     # Ensemble size (number of predictions per sample)
    L = 16000  # Audio length (e.g., 1 second at 16kHz)

    # 3. Create synthetic ground truth (hr_audio)
    # Simulating a standardized waveform (mean=0, std=1)
    hr_audio = torch.randn(B, L)

    # =========================================================
    # SCENARIO 1: Perfectly Calibrated Model
    # The model predicts the true distribution accurately (std=1)
    # =========================================================
    sr_calibrated = torch.randn(B, M, L)
    
    print("=== SCENARIO 1: Perfectly Calibrated Model ===")
    crps_calibrated = evaluator.evaluate_crps(hr_audio, sr_calibrated)
    pit_calibrated = evaluator.evaluate_pit(hr_audio, sr_calibrated)
    
    print(f"CRPS:              {crps_calibrated:.4f} (Lower is better)")
    print(f"PIT Var Deviation: {pit_calibrated['pit_var_deviation']:.4f} (Ideal: ~0.0)")
    print(f"PIT KS-Stat:       {pit_calibrated['pit_ks_stat']:.4f} (Ideal: ~0.0)\n")


    # =========================================================
    # SCENARIO 2: Model Regressing to the Mean
    # The model outputs tiny variance around the mean (0). 
    # This simulates over-smoothed, muffled audio generation.
    # =========================================================
    sr_regressed = torch.randn(B, M, L) * 0.1  # Highly suppressed variance
    
    print("=== SCENARIO 2: Mean-Regressed Model ===")
    crps_regressed = evaluator.evaluate_crps(hr_audio, sr_regressed)
    pit_regressed = evaluator.evaluate_pit(hr_audio, sr_regressed)
    
    print(f"CRPS:              {crps_regressed:.4f}")
    print(f"PIT Var Deviation: {pit_regressed['pit_var_deviation']:.4f} (Positive = Under-dispersed)")
    print(f"PIT KS-Stat:       {pit_regressed['pit_ks_stat']:.4f} (High = Poor calibration)\n")


    # =========================================================
    # SCENARIO 3: Over-dispersed Model
    # The model generates too much random noise (std=3).
    # =========================================================
    sr_noisy = torch.randn(B, M, L) * 3.0
    
    print("=== SCENARIO 3: Over-dispersed Model (Too much noise) ===")
    crps_noisy = evaluator.evaluate_crps(hr_audio, sr_noisy)
    pit_noisy = evaluator.evaluate_pit(hr_audio, sr_noisy)
    
    print(f"CRPS:              {crps_noisy:.4f}")
    print(f"PIT Var Deviation: {pit_noisy['pit_var_deviation']:.4f} (Negative = Over-dispersed)")
    print(f"PIT KS-Stat:       {pit_noisy['pit_ks_stat']:.4f}")