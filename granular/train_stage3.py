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

from .models import HierarchicalModel
from .utils_stage1 import make_audio_dataloaders
from .utils_stage2 import plot_embeddings
from .utils_stage3 import export_audio_to_embeddings, export_hierarchical_audio_reconstructions, export_random_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stage3(
    data_dir: str,
    latent_name=None,
    waveform_name=None,
    batch_size=16,
    learning_rate=2e-6,  # here is the fixed learning rate at the end of the decay of the sub-network pretraining
    w_beta=0.0,
    l_beta=0.0,
    max_steps=100000,
    num_workers=2,
    gpus=1,
    precision=32,
    profiler=False,
    out_dir="modelzoo",
):
    if latent_name is None:
        latent_name = "granular_latent_" + Path(data_dir).stem
    if waveform_name is None:
        waveform_name = "granular_waveform_" + Path(data_dir).stem

    args = dict(
        data_dir=data_dir,
        latent_name=latent_name,
        waveform_name=waveform_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        w_beta=w_beta,
        l_beta=l_beta,
        max_steps=max_steps,
        num_workers=num_workers,
        gpus=gpus,
        precision=precision,
        profiler=profiler,
        out_dir=out_dir,
    )

    name = waveform_name + "__" + latent_name + "__finetuned"
    if w_beta > 0.0:
        name += "_wbeta" + str(w_beta)
    if l_beta > 0.0:
        name += "_lbeta" + str(l_beta)
    latent_name = waveform_name + "__" + latent_name

    default_root_dir = os.path.join(out_dir, name)
    print("writing outputs into default_root_dir", default_root_dir)

    # lighting is writting output files in default_root_dir/lightning_logs/version_0/
    tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

    ###############################################################################
    ## STAGE 1 & 2: loading configuration aof waveform and latent VAEs + creating audio dataset
    ###############################################################################

    print("\n*** loading of pretrained waveform VAE from", os.path.join(out_dir, waveform_name))

    w_args = np.load(os.path.join(out_dir, waveform_name, "config.npy"), allow_pickle=True).item()

    w_ckpt_file = sorted(glob.glob(os.path.join(out_dir, waveform_name, "checkpoints", "*.ckpt")))[-1]
    w_yaml_file = os.path.join(out_dir, waveform_name, "hparams.yaml")

    print("\n*** loading of pretrained latent VAE from", os.path.join(out_dir, latent_name))

    l_args = np.load(os.path.join(out_dir, latent_name, "config.npy"), allow_pickle=True).item()
    l_ckpt_file = sorted(glob.glob(os.path.join(out_dir, latent_name, "checkpoints", "*.ckpt")))[-1]
    l_yaml_file = os.path.join(out_dir, latent_name, "hparams.yaml")

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

    ###############################################################################
    ## STAGE 2: training latent VAE
    ###############################################################################

    print("\n*** STAGE 3: fine-tuning of waveform and latent VAEs")

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

    model = HierarchicalModel(
        w_ckpt_file=w_ckpt_file,
        w_yaml_file=w_yaml_file,
        l_ckpt_file=l_ckpt_file,
        l_yaml_file=l_yaml_file,
        learning_rate=learning_rate,
    )
    model.to(device)
    model.init_beta(w_args, l_args, w_beta=w_beta, l_beta=l_beta)
    model.init_SpectralDistances(w_args["w_config"], device=device)
    model.export_dir = os.path.join(tmp_dir, "exports")  # to write export files

    print("model running on device", model.device)
    print("model hyper-parameters", model.hparams)

    model.train()
    for batch in train_dataloader:
        break
    model.gradient_check(batch)

    # ------------
    # training
    # ------------

    print("\n*** start of STAGE 3 training")

    time.sleep(10)

    trainer.fit(model, train_dataloader, test_dataloader)

    print("\n*** end of STAGE 3 training after #iter = ", model.acc_iter)

    # ------------
    # export
    # ------------

    model.to(device)
    model.eval()

    print("\n*** exporting hierarchical audio reconstructions")

    for batch in train_dataloader:
        break
    export_hierarchical_audio_reconstructions(model, batch, trainset=True)
    for batch in test_dataloader:
        break
    export_hierarchical_audio_reconstructions(model, batch, trainset=False)

    print("\n*** exporting random samples embedding to audio")

    export_random_samples(model, model.export_dir, n_samples=10)

    print("\n*** plotting embedding projections")

    train_latents, train_labels, test_latents, test_labels = export_audio_to_embeddings(
        model, train_dataloader, test_dataloader
    )

    plot_embeddings(train_latents, train_labels, test_latents, test_labels, classes, model.export_dir)

    # ------------
    # misc.
    # ------------

    np.save(os.path.join(tmp_dir, "config.npy"), args)
    shutil.move(tmp_dir, os.path.join(args["out_dir"]))
    shutil.rmtree(default_root_dir)
    os.rename(os.path.join(args["out_dir"], "version_0"), default_root_dir)
