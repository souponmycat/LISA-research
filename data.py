import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchaudio.functional import resample


def get_leaf_files(path: str, enders: Tuple[str, ...] = (".mp4", ".wav", ".flac", ".mp3")) -> List[str]:
    """Recursively scans directory for supported audio files without external packages."""
    audio_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(enders):
                audio_files.append(os.path.join(root, file))
    return audio_files


class AudioCorpus(Dataset):
    """
    PyTorch Dataset for Audio Super-Resolution.
    Handles loading, mono-conversion, resampling, and random cropping/padding.
    """

    def __init__(
        self,
        path: str = "./data",
        limit: Optional[int] = None,
        lsr: int = 8000,
        hsr: int = 16000,
        clip_sec: float = 2.0,
        trunc: bool = True,
        file_list: Optional[List[str]] = None,
        mono: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if file_list is None:
            self.files = get_leaf_files(path=path)
            np.random.seed(seed)
            np.random.shuffle(self.files)

            if limit is not None:
                self.files = self.files[: min(limit, len(self.files))]
        else:
            self.files = file_list

        self.lsr = lsr
        self.hsr = hsr
        self.clip_sec = clip_sec
        self.trunc = trunc
        self.mono = mono

        self.target_hr_samples = int(self.hsr * self.clip_sec)
        self.target_lr_samples = int(self.lsr * self.clip_sec)

    def __len__(self) -> int:
        return len(self.files)

    def _pad_or_crop(self, waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        curr_len = waveform.shape[-1]
        if curr_len > target_len:
            max_start = curr_len - target_len
            start = torch.randint(0, max_start + 1, (1,)).item()
            return waveform[:, start : start + target_len]
        elif curr_len < target_len:
            pad_amount = target_len - curr_len
            return torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[index]

        # 1. Load Audio
        data, sr = sf.read(file_path, dtype="float32")
        hrw = torch.from_numpy(data)

        if hrw.ndim == 1:
            hrw = hrw.unsqueeze(0)
        else:
            hrw = hrw.transpose(0, 1)

        # 2. Convert to Mono
        if self.mono and hrw.shape[0] > 1:
            hrw = torch.mean(hrw, dim=0, keepdim=True)

        # 3. Resample to High Sampling Rate
        if sr != self.hsr:
            hrw = resample(hrw, sr, self.hsr)

        # 4. Truncate/Pad High-Res Audio
        if self.trunc:
            hrw = self._pad_or_crop(hrw, self.target_hr_samples)

        # 5. Resample to Low Sampling Rate
        lrw = resample(hrw, self.hsr, self.lsr)

        # 6. Ensure Low-Res Exact Sample Length
        if self.trunc:
            lrw = self._pad_or_crop(lrw, self.target_lr_samples)

        return {"lr": lrw, "hr": hrw}


def get_audio_dataloaders(
    data_path: str = "./data",
    batch_size: int = 32,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    limit_files: Optional[int] = None,
    lsr: int = 8000,
    hsr: int = 16000,
    clip_sec: float = 2.0,
    num_eval_samples: int = 12,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    generator = torch.Generator().manual_seed(seed)

    full_dataset = AudioCorpus(
        path=data_path,
        limit=limit_files,
        lsr=lsr,
        hsr=hsr,
        clip_sec=clip_sec,
        trunc=True,
        seed=seed,
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No audio files found in: {data_path}")

    tr_len = int(train_ratio * len(full_dataset))
    val_len = len(full_dataset) - tr_len
    tr_set, val_set = random_split(full_dataset, [tr_len, val_len], generator=generator)

    use_persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    tr_loader = DataLoader(
        tr_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch,
    )

    val_indices = val_set.indices[: min(num_eval_samples, len(val_set))]
    val_eval_files = [full_dataset.files[i] for i in val_indices]

    val_eval_set = AudioCorpus(
        file_list=val_eval_files,
        lsr=lsr,
        hsr=hsr,
        clip_sec=clip_sec,
        trunc=False,
        seed=seed,
    )

    val_eval_loader = DataLoader(
        val_eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return tr_loader, val_loader, val_eval_loader