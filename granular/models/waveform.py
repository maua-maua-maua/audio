import numpy as np
import pytorch_lightning as pl
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F

from .layers import LinearBlock, ResidualConv, StridedConv
from .utils import SpectralDistances, compute_kld, envelope_distance, mod_sigmoid, noise_filtering, reparametrize


class WaveformModel(pl.LightningModule):
    def __init__(
        self,
        z_dim,
        h_dim,
        kernel_size,
        channels,
        n_convs,
        stride,
        n_linears,
        n_grains,
        hop_size,
        normalize_ola,
        pp_chans,
        pp_ker,
        l_grain=2048,
        sr=16000,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0

        # fixed parameters

        self.tar_l = int((n_grains + 3) / 4 * l_grain)
        self.filter_size = l_grain // 2 + 1
        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)), requires_grad=False)

        ola_window = signal.hann(l_grain, sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains, 1).type(torch.float32)
        ola_windows[0, : l_grain // 2] = ola_window[
            l_grain // 2
        ]  # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1, l_grain // 2 :] = ola_window[l_grain // 2]
        self.ola_windows = nn.Parameter(ola_windows, requires_grad=False)

        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1), requires_grad=False)

        self.ola_folder = nn.Fold((self.tar_l, 1), (l_grain, 1), stride=(hop_size, 1))
        if normalize_ola:
            unfolder = nn.Unfold((l_grain, 1), stride=(hop_size, 1))
            input_ones = torch.ones(1, 1, self.tar_l, 1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor, requires_grad=False)

        # encoder parameters

        encoder_convs = [
            nn.Sequential(StridedConv(kernel_size, 1, channels, stride), ResidualConv(channels, n_blocks=3))
        ]
        encoder_convs += [
            nn.Sequential(StridedConv(kernel_size, channels, channels, stride), ResidualConv(channels, n_blocks=3))
            for i in range(1, n_convs)
        ]
        self.encoder_convs = nn.ModuleList(encoder_convs)

        self.flatten_size = int(channels * l_grain / (stride ** n_convs))
        self.encoder_linears = nn.Sequential(LinearBlock(self.flatten_size, h_dim), LinearBlock(h_dim, z_dim))
        self.mu = nn.Linear(z_dim, z_dim)
        self.logvar = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.Hardtanh(min_val=-5.0, max_val=5.0)
        )  # clipping to avoid numerical instabilities

        # decoder parameters

        decoder_linears = [LinearBlock(z_dim, h_dim)]
        decoder_linears += [LinearBlock(h_dim, h_dim) for i in range(1, n_linears)]
        decoder_linears += [nn.Linear(h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

        self.post_pro = nn.Sequential(nn.Conv1d(pp_chans, 1, pp_ker, padding=pp_ker // 2), nn.Softsign())

    def init_SpectralDistances(
        self,
        stft_scales=[2048, 1024, 512, 256, 128],
        mel_scales=[2048, 1024],
        spec_power=1,
        mel_dist=True,
        log_dist=0.0,
        env_dist=0,
        device="cpu",
    ):
        self.spec_dist = SpectralDistances(
            stft_scales=stft_scales,
            mel_scales=mel_scales,
            spec_power=spec_power,
            mel_dist=mel_dist,
            log_dist=log_dist,
            sr=self.hparams.sr,
            device=device,
        )
        self.env_dist = env_dist

    def init_beta(self, max_steps, tar_beta, beta_steps=1000):
        if self.continue_train:
            self.tar_beta = tar_beta
            self.beta = tar_beta
            print("\n*** setting fixed beta of ", self.beta)
        else:
            self.max_steps = max_steps
            self.tar_beta = tar_beta
            self.beta_steps = beta_steps  # number of warmup steps over half max_steps
            self.warmup_start = int(0.1 * max_steps)
            self.beta_step_size = max(1, int(max_steps / 2 / beta_steps))
            self.beta_step_val = tar_beta / beta_steps
            self.beta = 0
            print("\n*** setting beta warmup from 0 to ", tar_beta)

    def encode(self, x, print_shapes=False):
        # slicing input mini-batch of shape [bs,tar_l]
        mb_grains = F.conv1d(x.unsqueeze(1), self.slice_kernel, stride=self.hparams.hop_size, groups=1, bias=None)
        mb_grains = mb_grains.permute(0, 2, 1)
        # windowing input mb_grains of shape [bs,n_grains,l_grain]
        bs = mb_grains.shape[0]
        mb_grains = mb_grains * (self.ola_windows.unsqueeze(0).repeat(bs, 1, 1))
        mb_grains = mb_grains.reshape(bs * self.hparams.n_grains, self.hparams.l_grain).unsqueeze(1)
        if print_shapes:
            print("input batch size", mb_grains.shape)
        # mb_grains of shape [bs*n_grains,1,l_grain]
        for conv in self.encoder_convs:
            mb_grains = conv(mb_grains)
            if print_shapes:
                print("output conv size", mb_grains.shape)
        mb_grains = mb_grains.view(-1, self.flatten_size)
        if print_shapes:
            print("flatten size", mb_grains.shape)
        h = self.encoder_linears(mb_grains)
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = reparametrize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar}

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):
        filter_coeffs = self.decoder_linears(z)
        # filter_coeffs of shape [bs*n_grains,filter_size]
        filter_coeffs = mod_sigmoid(filter_coeffs)
        audio = noise_filtering(filter_coeffs, self.filter_window)
        # windowing from audio of shape [bs*n_grains,l_grain]
        if n_grains is None:
            audio = audio.reshape(-1, self.hparams.n_grains, self.hparams.l_grain)
        else:
            audio = audio.reshape(-1, n_grains, self.hparams.l_grain)
        bs = audio.shape[0]
        if ola_windows is None:
            audio = audio * (self.ola_windows.unsqueeze(0).repeat(bs, 1, 1))
        else:
            audio = audio * (ola_windows.unsqueeze(0).repeat(bs, 1, 1))
        # overlap-add
        if ola_folder is None:
            audio_sum = self.ola_folder(audio.permute(0, 2, 1)).squeeze()
        else:
            audio_sum = ola_folder(audio.permute(0, 2, 1)).squeeze()
        if self.hparams.normalize_ola:
            if ola_divisor is None:
                audio_sum = audio_sum / self.ola_divisor.unsqueeze(0).repeat(bs, 1)
            else:
                audio_sum = audio_sum / ola_divisor.unsqueeze(0).repeat(bs, 1)
        # post-processing of audio_sum of shape [bs,tar_l]
        audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1, self.hparams.pp_chans, 1)).squeeze(1)
        return audio_sum

    def forward(self, audio, sampling=True):
        encoder_outputs = self.encode(audio)
        if sampling:
            audio_output = self.decode(encoder_outputs["z"])
        else:
            audio_output = self.decode(encoder_outputs["mu"])
        return audio_output, encoder_outputs

    def compute_losses(self, batch, beta):
        audio, labels = batch
        audio = audio.to(self.device)
        # forward
        audio_output, encoder_outputs = self.forward(audio, sampling=True)
        # compute losses
        spec_loss = self.spec_dist(audio_output, audio)
        if beta > 0:
            kld_loss = compute_kld(encoder_outputs["mu"], encoder_outputs["logvar"]) * beta
        else:
            kld_loss = 0
        if self.env_dist > 0:
            env_loss = envelope_distance(audio_output, audio, n_fft=1024, log=True) * self.env_dist
        else:
            env_loss = 0
        return {"spec_loss": spec_loss, "kld_loss": kld_loss, "env_loss": env_loss}

    def training_step(self, batch, batch_idx):
        if not self.continue_train:
            if (self.acc_iter + 1) % self.beta_step_size == 0:
                if self.acc_iter < self.warmup_start:
                    self.beta = 0
                elif self.beta < self.tar_beta:
                    self.beta += self.beta_step_val
                    self.beta = np.min([self.beta, self.tar_beta])
                else:
                    self.beta = self.tar_beta
        losses = self.compute_losses(batch, self.beta)
        spec_loss, kld_loss, env_loss = losses["spec_loss"], losses["kld_loss"], losses["env_loss"]
        loss = spec_loss + kld_loss + env_loss
        self.log("waveform_train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_train_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_train_kld_loss", kld_loss, on_step=False, on_epoch=True)
        self.log("waveform_beta_kld", self.beta, on_step=False, on_epoch=True)
        if self.env_dist > 0:
            self.log("waveform_train_env_loss", env_loss)
        self.acc_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch, self.beta)
        spec_loss, kld_loss, env_loss = losses["spec_loss"], losses["kld_loss"], losses["env_loss"]
        loss = spec_loss + kld_loss + env_loss
        self.log("waveform_test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_test_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_test_kld_loss", kld_loss, on_step=False, on_epoch=True)
        if self.env_dist > 0:
            self.log("waveform_test_env_loss", env_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.continue_train:
            print("*** setting fixed learning rate of", self.hparams.learning_rate)
            return opt
        else:
            lr_decay = 1e-2
            lr_scale = np.exp(np.log(lr_decay) / self.max_steps)
            print(
                "*** setting exponential decay of learning rate with factor and final value to",
                lr_scale,
                self.hparams.learning_rate * lr_scale ** self.max_steps,
            )
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(opt, lr_scale, verbose=False),
                "interval": "step",
            }
            # TODO: lr_scheduler may be best by stepping every epoch for ExponentialLR ?
            return [opt], [scheduler]

    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.env_dist > 0:
            training_losses = ["spec_loss", "env_loss", "kld_loss"]
        else:
            training_losses = ["spec_loss", "kld_loss"]
        for loss in training_losses:
            print("\n*** " + loss + " initial gradient check")
            if loss != "kld_loss":
                losses = self.compute_losses(batch, 0.0)
            else:
                losses = self.compute_losses(batch, self.tar_beta)
            opt.zero_grad()
            losses[loss].backward()
            tot_grad = 0
            named_p = self.named_parameters()
            for name, param in named_p:
                if param.grad is not None:
                    sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                    if sum_abs_paramgrad == 0:
                        print(name, "sum_abs_paramgrad==0")
                    else:
                        tot_grad += sum_abs_paramgrad
                else:
                    print(name, "param.grad is None")
            print("tot_grad = ", tot_grad)
        opt.zero_grad()
