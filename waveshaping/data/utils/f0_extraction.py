from functools import partial
from typing import Callable, Optional, Sequence, Union

import gin
import librosa
import numpy as np
import torch
import torchcrepe

from ...utils import apply
from .upsampling import linear_interpolation

CREPE_WINDOW_LENGTH = 1024


@gin.configurable
def extract_f0_with_crepe(
    audio: np.ndarray,
    sample_rate: float,
    hop_length: int = 128,
    minimum_frequency: float = 50.0,
    maximum_frequency: float = 2000.0,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    # convert to torch tensor with channel dimension (necessary for CREPE)
    audio = torch.tensor(audio).unsqueeze(0)
    f0, confidence = torchcrepe.predict(
        audio,
        sample_rate,
        hop_length,
        minimum_frequency,
        maximum_frequency,
        "full" if full_model else "tiny",
        batch_size=batch_size,
        device=device,
        decoder=torchcrepe.decode.viterbi,
        # decoder=torchcrepe.decode.weighted_argmax,
        return_harmonicity=True,
    )

    f0, confidence = f0.squeeze().numpy(), confidence.squeeze().numpy()

    if interpolate_fn:
        f0 = interpolate_fn(f0, CREPE_WINDOW_LENGTH, hop_length, original_length=audio.shape[-1])
        confidence = interpolate_fn(
            confidence,
            CREPE_WINDOW_LENGTH,
            hop_length,
            original_length=audio.shape[-1],
        )

    return f0, confidence


@gin.configurable
def extract_f0_with_pyin(
    audio: np.ndarray,
    sample_rate: float,
    minimum_frequency: float = 65.0,  # recommended minimum freq from librosa docs
    maximum_frequency: float = 2093.0,  # recommended maximum freq from librosa docs
    frame_length: int = 1024,
    hop_length: int = 128,
    fill_na: Optional[float] = None,
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    f0, _, voiced_prob = librosa.pyin(
        audio,
        sr=sample_rate,
        fmin=minimum_frequency,
        fmax=maximum_frequency,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=fill_na,
    )

    if interpolate_fn:
        f0 = interpolate_fn(f0, frame_length, hop_length, original_length=audio.shape[-1])
        voiced_prob = interpolate_fn(
            voiced_prob,
            frame_length,
            hop_length,
            original_length=audio.shape[-1],
        )

    return f0, voiced_prob
