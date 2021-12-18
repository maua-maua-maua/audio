import pytorch_lightning as pl
import torch
from granular.models import LatentModel, SpectralDistances, WaveformModel, compute_kld, envelope_distance
from torch.nn import functional as F


class HierarchicalModel(pl.LightningModule):
    def __init__(
        self,
        w_ckpt_file="w_ckpt_file",
        w_yaml_file="w_yaml_file",
        l_ckpt_file="l_ckpt_file",
        l_yaml_file="l_yaml_file",
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0

        self.w_model = WaveformModel.load_from_checkpoint(
            checkpoint_path=w_ckpt_file, hparams_file=w_yaml_file, map_location="cpu"
        )
        self.l_model = LatentModel.load_from_checkpoint(
            checkpoint_path=l_ckpt_file, hparams_file=l_yaml_file, map_location="cpu"
        )

    def init_SpectralDistances(self, w_config, device="cpu"):
        self.spec_dist = SpectralDistances(
            stft_scales=w_config["stft_scales"],
            mel_scales=w_config["mel_scales"],
            spec_power=w_config["spec_power"],
            mel_dist=w_config["mel_dist"],
            log_dist=w_config["log_dist"],
            sr=self.w_model.hparams.sr,
            device=device,
        )
        self.env_dist = w_config["env_dist"]

    def init_beta(self, w_args, l_args, w_beta=0.0, l_beta=0.0):
        if w_beta == 0.0:
            self.w_beta = w_args["tar_beta"]
        else:
            self.w_beta = w_beta
        if l_beta == 0.0:
            self.l_beta = l_args["tar_beta"]
        else:
            self.l_beta = l_beta
        print("\n*** setting fixed beta for bottom and top KLD of ", self.w_beta, self.l_beta)

    def encode(self, x, sampling=True):
        w_encoder_outputs = self.w_model.encode(x)
        if sampling:
            l_encoder_outputs = self.l_model.encode(
                w_encoder_outputs["z"].reshape(-1, self.l_model.hparams.n_grains, self.l_model.hparams.z_dim)
            )
        else:
            l_encoder_outputs = self.l_model.encode(
                w_encoder_outputs["mu"].reshape(-1, self.l_model.hparams.n_grains, self.l_model.hparams.z_dim)
            )
        return l_encoder_outputs, w_encoder_outputs

    def decode(self, e, conds):
        z_hat = self.l_model.decode(e, conds).reshape(-1, self.l_model.hparams.z_dim)
        audio_output = self.w_model.decode(z_hat)
        return audio_output, z_hat

    def forward(self, audio, conds, sampling=True):
        l_encoder_outputs, w_encoder_outputs = self.encode(audio, sampling=sampling)
        if sampling:
            audio_output, z_hat = self.decode(l_encoder_outputs["e"], conds)
        else:
            audio_output, z_hat = self.decode(l_encoder_outputs["mu"], conds)
        return audio_output, z_hat, l_encoder_outputs, w_encoder_outputs

    def compute_losses(self, batch):
        audio, labels = batch
        audio, labels = audio.to(self.device), labels.to(self.device)
        # forward
        audio_output, z_hat, l_encoder_outputs, w_encoder_outputs = self.forward(audio, labels, sampling=True)
        # compute losses
        spec_loss = self.spec_dist(audio_output, audio)
        w_kld_loss = compute_kld(w_encoder_outputs["mu"], w_encoder_outputs["logvar"]) * self.w_beta
        if self.env_dist > 0:
            env_loss = envelope_distance(audio_output, audio, n_fft=1024, log=True) * self.env_dist
        else:
            env_loss = 0
        l_kld_loss = compute_kld(l_encoder_outputs["mu"], l_encoder_outputs["logvar"]) * self.l_beta
        l_rec_loss = F.mse_loss(z_hat, w_encoder_outputs["z"])
        return {
            "spec_loss": spec_loss,
            "w_kld_loss": w_kld_loss,
            "env_loss": env_loss,
            "l_rec_loss": l_rec_loss,
            "l_kld_loss": l_kld_loss,
        }

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        spec_loss, w_kld_loss, env_loss, l_rec_loss, l_kld_loss = (
            losses["spec_loss"],
            losses["w_kld_loss"],
            losses["env_loss"],
            losses["l_rec_loss"],
            losses["l_kld_loss"],
        )
        loss = spec_loss + w_kld_loss + env_loss + l_rec_loss + l_kld_loss
        self.log("train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_train_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_train_kld_loss", w_kld_loss, on_step=False, on_epoch=True)
        if self.env_dist > 0:
            self.log("waveform_train_env_loss", env_loss)
        self.log("latent_train_rec_loss", l_rec_loss, on_step=False, on_epoch=True)
        self.log("latent_train_kld_loss", l_kld_loss, on_step=False, on_epoch=True)
        self.acc_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        spec_loss, w_kld_loss, env_loss, l_rec_loss, l_kld_loss = (
            losses["spec_loss"],
            losses["w_kld_loss"],
            losses["env_loss"],
            losses["l_rec_loss"],
            losses["l_kld_loss"],
        )
        loss = spec_loss + w_kld_loss + env_loss + l_rec_loss + l_kld_loss
        self.log("test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_test_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_test_kld_loss", w_kld_loss, on_step=False, on_epoch=True)
        if self.env_dist > 0:
            self.log("waveform_test_env_loss", env_loss)
        self.log("latent_test_rec_loss", l_rec_loss, on_step=False, on_epoch=True)
        self.log("latent_test_kld_loss", l_kld_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        print("*** setting fixed learning rate of", self.hparams.learning_rate)
        return opt

    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.env_dist > 0:
            training_losses = ["spec_loss", "w_kld_loss", "env_loss", "l_rec_loss", "l_kld_loss"]
        else:
            training_losses = ["spec_loss", "w_kld_loss", "l_rec_loss", "l_kld_loss"]
        for loss in training_losses:
            print("\n*** " + loss + " initial gradient check")
            losses = self.compute_losses(batch)
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
