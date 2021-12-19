#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:17:53 2021

@author: adrienbitton
"""

import glob
import os
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from .models import WaveformModel
from .utils_stage1 import export_audio_reconstructions, export_latents, make_audio_dataloaders, plot_latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stage1(
    data_dir: str,
    classes=[],
    name=None,
    continue_train=False,
    batch_size=24,
    learning_rate=0.0002,
    max_steps=300000,
    num_workers=2,
    gpus=1,
    precision=32,
    profiler=False,
    out_dir="modelzoo",
    tar_beta=0.003,
    beta_steps=500,
    tar_l=1.1,
    # waveform model config
    amplitude_norm=False,
    channels=128,
    env_dist=0,
    h_dim=512,
    kernel_size=9,
    l_grain=2048,
    log_dist=0.0,
    mel_dist=True,
    mel_scales=[2048, 1024],
    n_convs=3,
    n_linears=3,
    normalize_ola=True,
    pp_chans=5,
    pp_ker=65,
    silent_reject=[0.2, 0.2],
    spec_power=1,
    sr=22050,
    stft_scales=[2048, 1024, 512, 256],
    stride=4,
    z_dim=128,
):

    if name is None:
        name = "granular_waveform_" + Path(data_dir).stem

    args = dict(
        classes=sorted(classes),
        data_dir=data_dir,
        name=name,
        continue_train=continue_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_workers=num_workers,
        gpus=gpus,
        precision=precision,
        profiler=profiler,
        out_dir=out_dir,
        tar_beta=tar_beta,
        beta_steps=beta_steps,
        tar_l=tar_l,
        w_config=dict(
            amplitude_norm=amplitude_norm,
            channels=channels,
            env_dist=env_dist,
            h_dim=h_dim,
            kernel_size=kernel_size,
            l_grain=l_grain,
            log_dist=log_dist,
            mel_dist=mel_dist,
            mel_scales=mel_scales,
            n_convs=n_convs,
            n_linears=n_linears,
            normalize_ola=normalize_ola,
            pp_chans=pp_chans,
            pp_ker=pp_ker,
            silent_reject=silent_reject,
            spec_power=spec_power,
            sr=sr,
            stft_scales=stft_scales,
            stride=stride,
            z_dim=z_dim,
        ),
    )

    if continue_train:
        ckpt_file = sorted(glob.glob(os.path.join(out_dir, name, "checkpoints", "*.ckpt")))[-1]
        yaml_file = os.path.join(out_dir, name, "hparams.yaml")
        name = name + "_continue"
        # take care of setting the learning rate and beta kld to the target end of training values
        lr_decay = 1e-2
        learning_rate = learning_rate * lr_decay
        print("\n*** training continuation for ", name)
        print("from ckpt_file,yaml_file =", ckpt_file, yaml_file)

    default_root_dir = os.path.join(out_dir, name)
    print("writing outputs into default_root_dir", default_root_dir)

    # lighting is writting output files in default_root_dir/lightning_logs/version_0/
    tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

    ###############################################################################
    ## STAGE 1: training waveform VAE
    ###############################################################################

    print("\n*** STAGE 1: training waveform VAE")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_steps=max_steps,
        check_val_every_n_epoch=1,
        gpus=gpus,
        precision=precision,
        benchmark=True,
        default_root_dir=default_root_dir,
        profiler=profiler,
        progress_bar_refresh_rate=50,
        callbacks=[lr_monitor],
    )

    # ------------
    # data
    # ------------

    print("\n*** loading data")

    train_dataloader, test_dataloader, tar_l, n_grains, l_grain, hop_size, classes = make_audio_dataloaders(
        data_dir,
        classes,
        sr,
        silent_reject,
        amplitude_norm,
        batch_size,
        tar_l=tar_l,
        l_grain=l_grain,
        high_pass_freq=50.0,
        num_workers=num_workers,
    )

    # ------------
    # model
    # ------------

    print("\n*** building model")

    if continue_train:
        w_model = WaveformModel.load_from_checkpoint(
            checkpoint_path=ckpt_file, hparams_file=yaml_file, map_location="cpu", learning_rate=learning_rate
        )
    else:
        w_model = WaveformModel(
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
            l_grain=l_grain,
            sr=sr,
            learning_rate=learning_rate,
        )
    w_model.continue_train = continue_train
    w_model.to(device)
    w_model.init_beta(max_steps, tar_beta, beta_steps=beta_steps)
    w_model.init_SpectralDistances(
        stft_scales=stft_scales,
        mel_scales=mel_scales,
        spec_power=spec_power,
        mel_dist=mel_dist,
        log_dist=log_dist,
        env_dist=env_dist,
        device=device,
    )  # TODO: it seems that scale=512 creates empty mel filterbank ?
    w_model.export_dir = os.path.join(tmp_dir, "exports")  # to write export files

    print("model running on device", w_model.device)
    print("model hyper-parameters\n", w_model.hparams)

    w_model.train()
    for batch in train_dataloader:
        break
    w_model.gradient_check(batch)  # TODO: callback at beginning and end of training ?

    # ------------
    # training
    # ------------

    print("\n*** start of STAGE 1 training")

    time.sleep(10)

    trainer.fit(w_model, train_dataloader, test_dataloader)

    print("\n*** end of STAGE 1 training after #iter = ", w_model.acc_iter)

    # ------------
    # export
    # ------------

    w_model.to(device)
    w_model.eval()

    print("\n*** exporting audio reconstructions")

    for batch in train_dataloader:
        break
    export_audio_reconstructions(w_model, batch, trainset=True)
    for batch in test_dataloader:
        break
    export_audio_reconstructions(w_model, batch, trainset=False)

    print("\n*** exporting latent projections")

    train_latents, train_labels, test_latents, test_labels = export_latents(w_model, train_dataloader, test_dataloader)

    plot_latents(train_latents, train_labels, test_latents, test_labels, classes, w_model.export_dir)

    # ------------
    # misc.
    # ------------

    np.save(os.path.join(tmp_dir, "config.npy"), args)
    shutil.move(tmp_dir, os.path.join(args["out_dir"]))
    shutil.rmtree(default_root_dir)
    os.rename(os.path.join(args["out_dir"], "version_0"), default_root_dir)
