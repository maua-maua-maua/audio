import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from .layers import LinearBlock
from .utils import compute_kld, reparametrize


class LatentModel(pl.LightningModule):
    def __init__(
        self, e_dim, z_dim, h_dim, n_linears, rnn_type, n_RNN, n_grains, classes, conditional, learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0

        # when len(classes)>1 and conditional=True then the decoder receives a one-hot vector of size len(classes)
        # TODO: replace one-hot condition with FiLM modulation ? or add Fader-Network regularization to the encoder
        if conditional is True and len(classes) > 1:
            self.n_conds = len(classes)
            print("\n*** training latent VAE with class conditioning over", classes)
        else:
            self.n_conds = 0

        # encoder parameters

        encoder_z = [LinearBlock(z_dim, h_dim, norm="LN")]
        encoder_z += [LinearBlock(h_dim, h_dim, norm="LN") for i in range(1, n_linears)]
        self.encoder_z = nn.Sequential(*encoder_z)

        if rnn_type == "GRU":
            self.encoder_rnn = nn.GRU(h_dim, h_dim, num_layers=n_RNN, batch_first=True)
        if rnn_type == "LSTM":
            self.encoder_rnn = nn.LSTM(h_dim, h_dim, num_layers=n_RNN, batch_first=True)

        encoder_e = [LinearBlock(h_dim, h_dim) for i in range(1, n_linears)]
        encoder_e += [LinearBlock(h_dim, e_dim)]
        self.encoder_e = nn.Sequential(*encoder_e)

        self.mu = nn.Linear(e_dim, e_dim)
        self.logvar = nn.Sequential(
            nn.Linear(e_dim, e_dim), nn.Hardtanh(min_val=-5.0, max_val=5.0)
        )  # clipping to avoid numerical instabilities

        # decoder parameters

        decoder_e = [LinearBlock(e_dim + self.n_conds, h_dim)]  # global conditioning before the RNN
        decoder_e += [LinearBlock(h_dim, h_dim) for i in range(1, n_linears)]
        self.decoder_e = nn.Sequential(*decoder_e)

        if rnn_type == "GRU":
            self.decoder_rnn = nn.GRU(h_dim, h_dim, num_layers=n_RNN, batch_first=True)
        if rnn_type == "LSTM":
            self.decoder_rnn = nn.LSTM(h_dim, h_dim, num_layers=n_RNN, batch_first=True)

        decoder_z = [LinearBlock(h_dim + self.n_conds, h_dim, norm="LN")]  # granular conditioning after the RNN
        decoder_z += [LinearBlock(h_dim, h_dim, norm="LN") for i in range(1, n_linears)]
        decoder_z += [nn.Linear(h_dim, z_dim)]
        self.decoder_z = nn.Sequential(*decoder_z)

    def init_beta(self, max_steps, tar_beta, beta_steps=1000):
        self.max_steps = max_steps
        self.tar_beta = tar_beta
        self.beta_steps = beta_steps  # number of warmup steps over half max_steps
        self.warmup_start = int(0.1 * max_steps)
        self.beta_step_size = max(1, int(max_steps / 2 / beta_steps))
        self.beta_step_val = tar_beta / beta_steps
        self.beta = 0
        print("\n*** setting beta warmup from 0 to ", tar_beta)

    def encode(self, z):
        # input series of latents of shape [bs,n_grains,z_dim]
        h = self.encoder_z(z)
        _, h_n = self.encoder_rnn(h)
        if self.hparams.rnn_type == "LSTM":
            h_n = h_n[0]  # we ommit the additional LSTM cell state
            # using the last cell state to init the decoder prevents from random sampling (without encoder outputs)
        h = self.encoder_e(h_n[-1, :, :])
        mu = self.mu(h)
        logvar = self.logvar(h)
        e = reparametrize(mu, logvar)
        return {"e": e, "mu": mu, "logvar": logvar}

    def decode(self, e, conds):
        # input embedding of shape [N,e_dim] and conds of shape [N] (long)
        if self.n_conds > 0:
            conds = F.one_hot(conds, num_classes=self.n_conds)
            e = torch.cat((e, conds), 1)
        h = self.decoder_e(e)
        h = h.unsqueeze(1).repeat(1, self.hparams.n_grains, 1).contiguous()
        # otherwise could use an auto-regressive approach if mean seaking
        # e.g. https://stackoverflow.com/questions/65205506/lstm-autoencoder-problems
        h, _ = self.decoder_rnn(h)
        if self.n_conds > 0:
            conds = conds.unsqueeze(1).repeat(1, self.hparams.n_grains, 1).contiguous()
            h = torch.cat((h, conds), 2)
        z = self.decoder_z(h)
        return z

    def forward(self, z, conds, sampling=True):
        encoder_outputs = self.encode(z)
        if sampling:
            z_hat = self.decode(encoder_outputs["e"], conds)
        else:
            z_hat = self.decode(encoder_outputs["mu"], conds)
        return z_hat, encoder_outputs

    def compute_losses(self, batch, beta):
        z, conds = batch
        z, conds = z.to(self.device), conds.to(self.device)
        # forward
        z_hat, encoder_outputs = self.forward(z, conds, sampling=True)
        # compute losses
        rec_loss = F.mse_loss(z_hat, z)  # we train with a deterministic output
        # TODO: compare with gaussian output and KLD distance ?
        if beta > 0:
            kld_loss = compute_kld(encoder_outputs["mu"], encoder_outputs["logvar"]) * beta
        else:
            kld_loss = 0
        return {"rec_loss": rec_loss, "kld_loss": kld_loss}

    def training_step(self, batch, batch_idx):
        if (self.acc_iter + 1) % self.beta_step_size == 0:
            if self.acc_iter < self.warmup_start:
                self.beta = 0
            elif self.beta < self.tar_beta:
                self.beta += self.beta_step_val
                self.beta = np.min([self.beta, self.tar_beta])
            else:
                self.beta = self.tar_beta
        losses = self.compute_losses(batch, self.beta)
        rec_loss, kld_loss = losses["rec_loss"], losses["kld_loss"]
        loss = rec_loss + kld_loss
        self.log("latent_train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("latent_train_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("latent_train_kld_loss", kld_loss, on_step=False, on_epoch=True)
        self.log("latent_beta_kld", self.beta, on_step=False, on_epoch=True)
        self.acc_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch, self.beta)
        rec_loss, kld_loss = losses["rec_loss"], losses["kld_loss"]
        loss = rec_loss + kld_loss
        self.log("latent_test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("latent_test_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("latent_test_kld_loss", kld_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
        return [opt], [scheduler]

    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        for loss in ["rec_loss", "kld_loss"]:
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
