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
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from models import hierarchical_model
from utils_stage1 import make_audio_dataloaders
from utils_stage2 import plot_embeddings
from utils_stage3 import export_audio_to_embeddings, export_hierarchical_audio_reconstructions, export_random_samples

if __name__ == "__main__":
    pl.seed_everything(1234)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curr_dir = os.getcwd()

    # ------------
    # hyper-parameters and trainer
    # ------------

    parser = ArgumentParser()
    parser.add_argument("--latent_name", default=None, type=str)
    parser.add_argument("--waveform_name", default=None, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--learning_rate", default=2e-6, type=float
    )  # here is the fixed learning rate at the end of the decay of the sub-network pretraining
    parser.add_argument("--w_beta", default=0.0, type=float)
    parser.add_argument("--l_beta", default=0.0, type=float)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--out_dir", default="outputs", type=str)
    args = parser.parse_args()

    if args.latent_name is None:
        args.latent_name = "latent_" + Path(args.data_dir).stem
    if args.waveform_name is None:
        args.waveform_name = "waveform_" + Path(args.data_dir).stem

    args.name = args.waveform_name + "__" + args.latent_name + "__finetuned"
    if args.w_beta > 0.0:
        args.name += "_wbeta" + str(args.w_beta)
    if args.l_beta > 0.0:
        args.name += "_lbeta" + str(args.l_beta)
    args.latent_name = args.waveform_name + "__" + args.latent_name

    default_root_dir = os.path.join(curr_dir, args.out_dir, args.name)
    print("writing outputs into default_root_dir", default_root_dir)

    # lighting is writting output files in default_root_dir/lightning_logs/version_0/
    tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

    ###############################################################################
    ## STAGE 1 & 2: loading configuration aof waveform and latent VAEs + creating audio dataset
    ###############################################################################

    print("\n*** loading of pretrained waveform VAE from", os.path.join(curr_dir, args.out_dir, args.waveform_name))

    w_args = np.load(os.path.join(curr_dir, args.out_dir, args.waveform_name, "argparse.npy"), allow_pickle=True).item()
    from train_stage1 import w_config

    w_ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.waveform_name, "checkpoints", "*.ckpt")))[
        -1
    ]
    w_yaml_file = os.path.join(curr_dir, args.out_dir, args.waveform_name, "hparams.yaml")

    print("\n*** loading of pretrained latent VAE from", os.path.join(curr_dir, args.out_dir, args.latent_name))

    l_args = np.load(os.path.join(curr_dir, args.out_dir, args.latent_name, "argparse.npy"), allow_pickle=True).item()
    l_ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.latent_name, "checkpoints", "*.ckpt")))[-1]
    l_yaml_file = os.path.join(curr_dir, args.out_dir, args.latent_name, "hparams.yaml")

    print("\n*** loading audio data")

    train_dataloader, test_dataloader, tar_l, n_grains, l_grain, hop_size, classes = make_audio_dataloaders(
        w_args["data_dir"],
        w_args["classes"],
        w_config["sr"],
        w_config["silent_reject"],
        w_config["amplitude_norm"],
        args.batch_size,
        tar_l=w_args["tar_l"],
        l_grain=w_config["l_grain"],
        high_pass_freq=50.0,
        num_workers=args.num_workers,
    )

    ###############################################################################
    ## STAGE 2: training latent VAE
    ###############################################################################

    print("\n*** STAGE 3: fine-tuning of waveform and latent VAEs")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        check_val_every_n_epoch=1,
        gpus=args.gpus,
        precision=args.precision,
        benchmark=True,
        default_root_dir=default_root_dir,
        profiler=args.profiler,
        progress_bar_refresh_rate=50,
        callbacks=[lr_monitor],
    )

    # ------------
    # model
    # ------------

    print("\n*** building model")

    model = hierarchical_model(
        w_ckpt_file=w_ckpt_file,
        w_yaml_file=w_yaml_file,
        l_ckpt_file=l_ckpt_file,
        l_yaml_file=l_yaml_file,
        learning_rate=args.learning_rate,
    )
    model.to(device)
    model.init_beta(w_args, l_args, w_beta=args.w_beta, l_beta=args.l_beta)
    model.init_SpectralDistances(w_config, device=device)
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

    args = vars(args)
    args["classes"] = classes  # make sure the classes are saved in the sorted order used for training

    np.save(os.path.join(tmp_dir, "argparse.npy"), args)
    shutil.move(tmp_dir, os.path.join(curr_dir, args["out_dir"]))
    shutil.rmtree(default_root_dir)
    os.rename(os.path.join(curr_dir, args["out_dir"], "version_0"), default_root_dir)

    # tensorboard --logdir
