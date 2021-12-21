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

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from .models import LatentModel, WaveformModel
from .utils_stage1 import export_latents, make_audio_dataloaders
from .utils_stage2 import (
    export_embedding_to_audio_reconstructions,
    export_embeddings,
    export_random_samples,
    make_latent_dataloaders,
    plot_embeddings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stage2(
    data_dir: str,
    name=None,
    waveform_name=None,
    batch_size=32,
    learning_rate=0.0002,
    max_steps=500000,
    tar_beta=0.01,
    beta_steps=500,
    conditional=False,
    num_workers=4,
    gpus=1,
    precision=32,
    profiler=False,
    out_dir="modelzoo/",
    # latent model config
    e_dim=256,
    h_dim=512,
    n_RNN=1,
    n_linears=2,
    rnn_type="LSTM",
):

    if name is None:
        name = "granular_latent_" + Path(data_dir).stem
    if waveform_name is None:
        waveform_name = "granular_waveform_" + Path(data_dir).stem

    name = waveform_name + "__" + name

    args = dict(
        name=name,
        waveform_name=waveform_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tar_beta=tar_beta,
        beta_steps=beta_steps,
        conditional=conditional,
        num_workers=num_workers,
        gpus=gpus,
        precision=precision,
        profiler=profiler,
        out_dir=out_dir,
        l_config=dict(
            e_dim=e_dim,
            h_dim=h_dim,
            n_RNN=n_RNN,
            n_linears=n_linears,
            rnn_type=rnn_type,
        ),
    )

    default_root_dir = os.path.join(out_dir, name)
    print("writing outputs into default_root_dir", default_root_dir)

    # lighting is writting output files in default_root_dir/lightning_logs/version_0/
    tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

    ###############################################################################
    ## STAGE 1: loading configuration and parameters of waveform VAE + creating dataset of latent projections
    ###############################################################################

    print("\n*** loading of pretrained waveform VAE from", os.path.join(out_dir, waveform_name))

    w_args = np.load(os.path.join(out_dir, waveform_name, "config.npy"), allow_pickle=True).item()

    print("\n*** loading audio data")

    train_dataloader, test_dataloader, tar_l, n_grains, l_grain, hop_size, classes = make_audio_dataloaders(
        w_args["data_dir"],
        w_args["classes"],
        w_args["w_config"]["sr"],
        w_args["w_config"]["silent_reject"],
        w_args["w_config"]["amplitude_norm"],
        batch_size,
        tar_l=w_args["tar_l"],
        l_grain=w_args["w_config"]["l_grain"],
        high_pass_freq=50.0,
        num_workers=num_workers,
    )

    print("\n*** restoring waveform model checkpoint")

    ckpt_file = sorted(glob.glob(os.path.join(out_dir, waveform_name, "checkpoints", "*.ckpt")))[-1]
    yaml_file = os.path.join(out_dir, waveform_name, "hparams.yaml")
    w_model = WaveformModel.load_from_checkpoint(checkpoint_path=ckpt_file, hparams_file=yaml_file, map_location="cpu")

    w_model.to(device)
    w_model.eval()
    w_model.freeze()

    # ------------
    # data
    # ------------

    print("\n*** exporting latent projections")

    train_latents, train_labels, test_latents, test_labels = export_latents(w_model, train_dataloader, test_dataloader)

    print("\n*** preparing latent dataloaders")

    train_latentloader, test_latentloader = make_latent_dataloaders(
        train_latents, train_labels, test_latents, test_labels, batch_size, num_workers=num_workers
    )

    ###############################################################################
    ## STAGE 2: training latent VAE
    ###############################################################################

    print("\n*** STAGE 2: training latent VAE")

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
    # model
    # ------------

    print("\n*** building model")

    l_model = LatentModel(
        e_dim,
        w_args["w_config"]["z_dim"],
        h_dim,
        n_linears,
        rnn_type,
        n_RNN,
        n_grains,
        classes,
        conditional,
        learning_rate=learning_rate,
    )
    l_model.to(device)
    l_model.init_beta(max_steps, tar_beta, beta_steps=beta_steps)
    l_model.export_dir = os.path.join(tmp_dir, "exports")  # to write export files

    print("model running on device", l_model.device)
    print("model hyper-parameters", l_model.hparams)

    l_model.train()
    for batch in train_latentloader:
        break
    l_model.gradient_check(batch)

    # ------------
    # training
    # ------------

    print("\n*** start of STAGE 2 training")

    time.sleep(10)

    trainer.fit(l_model, train_latentloader, test_latentloader)

    print("\n*** end of STAGE 2 training after #iter = ", l_model.acc_iter)

    # ------------
    # export
    # ------------

    l_model.to(device)
    l_model.eval()

    print("\n*** exporting embedding to audio reconstructions")

    for batch in train_latentloader:
        break
    export_embedding_to_audio_reconstructions(l_model, w_model, batch, trainset=True)
    for batch in test_latentloader:
        break
    export_embedding_to_audio_reconstructions(l_model, w_model, batch, trainset=False)

    print("\n*** exporting random samples embedding to audio")

    export_random_samples(l_model, w_model, l_model.export_dir, n_samples=10)

    print("\n*** plotting embedding projections")

    train_latents, train_labels, test_latents, test_labels = export_embeddings(
        l_model, train_latentloader, test_latentloader
    )

    plot_embeddings(train_latents, train_labels, test_latents, test_labels, classes, l_model.export_dir)

    # ------------
    # misc.
    # ------------

    np.save(os.path.join(tmp_dir, "config.npy"), args)
    shutil.move(tmp_dir, os.path.join(args["out_dir"]))
    shutil.rmtree(default_root_dir)
    os.rename(os.path.join(args["out_dir"], "version_0"), default_root_dir)
