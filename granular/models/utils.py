import math

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def compute_kld(mu, logvar):
    # TODO: add weighting of M/N = latent/input sizes
    mu = torch.flatten(mu, start_dim=1)
    logvar = torch.flatten(logvar, start_dim=1)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


def safe_log(x, eps=1e-7):
    return torch.log(x + eps)


def noise_filtering(filter_coeffs, filter_window):
    N = filter_coeffs.shape[0]
    l_grain = (filter_coeffs.shape[1] - 1) * 2
    dtype = filter_coeffs.dtype
    # create impulse response
    filter_coeffs = torch.complex(filter_coeffs, torch.zeros_like(filter_coeffs))
    filter_ir = torch.fft.irfft(filter_coeffs)
    filter_ir = filter_ir * filter_window.unsqueeze(0).repeat(N, 1)
    filter_ir = torch.fft.fftshift(filter_ir, dim=-1)
    # convolve with noise signal
    noise = torch.rand(N, l_grain, dtype=dtype, device=filter_coeffs.device) * 2 - 1
    S_noise = torch.fft.rfft(noise, dim=1)
    S_filter = torch.fft.rfft(filter_ir, dim=1)
    S = torch.mul(S_noise, S_filter)
    audio = torch.fft.irfft(S)
    return audio


class SpectralDistances(nn.Module):
    def __init__(
        self,
        stft_scales=[2048, 1024, 512, 256, 128],
        mel_scales=[2048, 1024],
        spec_power=1,
        mel_dist=True,
        log_dist=0,
        sr=16000,
        device="cpu",
    ):
        super(SpectralDistances, self).__init__()
        self.stft_scales = stft_scales
        self.mel_scales = mel_scales
        self.mel_dist = mel_dist
        self.log_dist = log_dist
        T_spec = []
        for scale in stft_scales:
            T_spec.append(
                Spectrogram(n_fft=scale, hop_length=scale // 4, window_fn=torch.hann_window, power=spec_power).to(
                    device
                )
            )
        self.T_spec = T_spec
        if mel_dist:
            print("\n*** training with MelSpectrogram distance")
            T_mel = []
            for scale in mel_scales:
                T_mel.append(
                    MelSpectrogram(
                        n_fft=scale,
                        hop_length=scale // 4,
                        window_fn=torch.hann_window,
                        sample_rate=sr,
                        f_min=50.0,
                        n_mels=scale // 4,
                        power=spec_power,
                    ).to(device)
                )
            self.T_mel = T_mel

    def forward(self, x_inp, x_tar):
        loss = 0
        n_scales = 0
        for i, scale in enumerate(self.stft_scales):
            S_inp, S_tar = self.T_spec[i](x_inp), self.T_spec[i](x_tar)
            stft_dist = (S_inp - S_tar).abs().mean()
            loss = loss + stft_dist
            n_scales += 1
            if self.log_dist > 0:
                loss = loss + (safe_log(S_inp) - safe_log(S_tar)).abs().mean() * self.log_dist
                n_scales += self.log_dist
        if self.mel_dist:
            for i, scale in enumerate(self.mel_scales):
                M_inp, M_tar = self.T_mel[i](x_inp), self.T_mel[i](x_tar)
                mel_dist = (M_inp - M_tar).abs().mean()
                loss = loss + mel_dist
                n_scales += 1
                if self.log_dist > 0:
                    loss = loss + (safe_log(M_inp) - safe_log(M_tar)).abs().mean() * self.log_dist
                    n_scales += self.log_dist
        return loss / n_scales


def envelope_distance(x_inp, x_tar, n_fft=1024, log=True):
    env_inp = torch.stft(x_inp, n_fft, hop_length=n_fft // 4, onesided=True, return_complex=False)
    env_inp = torch.mean(env_inp[:, :, :, 0] ** 2 + env_inp[:, :, :, 1] ** 2, 1)
    env_tar = torch.stft(x_tar, n_fft, hop_length=n_fft // 4, onesided=True, return_complex=False)
    env_tar = torch.mean(env_tar[:, :, :, 0] ** 2 + env_tar[:, :, :, 1] ** 2, 1)
    if log:
        env_inp, env_tar = safe_log(env_inp), safe_log(env_tar)
    return (env_inp - env_tar).abs().mean()
