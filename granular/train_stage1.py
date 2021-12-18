#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:17:53 2021

@author: adrienbitton

tested on python 3.7
python3.7 -m venv venv_tmp
source venv_tmp/bin/activate
pip install -r requirements.txt
"""


import glob
import os
import shutil
from argparse import ArgumentParser

import matplotlib
import pytorch_lightning as pl
import torch
from models import waveform_model
from pytorch_lightning.callbacks import LearningRateMonitor
from utils_stage1 import export_audio_reconstructions, export_latents, make_audio_dataloaders, plot_latents

matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.use("Agg")  # for the server
import json
import time

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    pl.seed_everything(1234)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curr_dir = os.getcwd()

    # ------------
    # hyper-parameters and trainer
    # ------------

    parser = ArgumentParser()
    parser.add_argument("--mname", default="test", type=str)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--data_dir", default="./a_dataset/", type=str)
    parser.add_argument("--classes", nargs="+", default=["Clap", "Hat", "Kick", "Snare"])
    parser.add_argument("--tar_beta", default=0.01, type=float)
    parser.add_argument("--beta_steps", default=500, type=int)
    parser.add_argument("--tar_l", default=1.1, type=float)
    args = parser.parse_args()

    if args.continue_train:
        ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.mname, "checkpoints", "*.ckpt")))[-1]
        yaml_file = os.path.join(curr_dir, args.out_dir, args.mname, "hparams.yaml")
        args.mname = args.mname + "_continue"
        # take care of setting the learning rate and beta kld to the target end of training values
        lr_decay = 1e-2
        args.learning_rate = args.learning_rate * lr_decay
        print("\n*** training continuation for ", args.mname)
        print("from ckpt_file,yaml_file =", ckpt_file, yaml_file)

    w_config = {
        "amplitude_norm": False,
        "channels": 128,
        "env_dist": 0,
        "h_dim": 512,
        "kernel_size": 9,
        "l_grain": 2048,
        "log_dist": 0.0,
        "mel_dist": True,
        "mel_scales": [2048, 1024],
        "n_convs": 3,
        "n_linears": 3,
        "normalize_ola": True,
        "pp_chans": 5,
        "pp_ker": 65,
        "silent_reject": [0.2, 0.2],
        "spec_power": 1,
        "sr": 22050,
        "stft_scales": [2048, 1024, 512, 256],
        "stride": 4,
        "z_dim": 128,
    }

    default_root_dir = os.path.join(curr_dir, args.out_dir, args.mname)
    print("writing outputs into default_root_dir", default_root_dir)

    # lighting is writting output files in default_root_dir/lightning_logs/version_0/
    tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

    ###############################################################################
    ## STAGE 1: training waveform VAE
    ###############################################################################

    print("\n*** STAGE 1: training waveform VAE")

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
    # data
    # ------------

    print("\n*** loading data")

    train_dataloader, test_dataloader, tar_l, n_grains, l_grain, hop_size, classes = make_audio_dataloaders(
        args.data_dir,
        args.classes,
        w_config["sr"],
        w_config["silent_reject"],
        w_config["amplitude_norm"],
        args.batch_size,
        tar_l=args.tar_l,
        l_grain=w_config["l_grain"],
        high_pass_freq=50.0,
        num_workers=args.num_workers,
    )

    # ------------
    # model
    # ------------

    print("\n*** building model")

    if args.continue_train:
        w_model = waveform_model.load_from_checkpoint(
            checkpoint_path=ckpt_file, hparams_file=yaml_file, map_location="cpu", learning_rate=args.learning_rate
        )
    else:
        w_model = waveform_model(
            w_config["z_dim"],
            w_config["h_dim"],
            w_config["kernel_size"],
            w_config["channels"],
            w_config["n_convs"],
            w_config["stride"],
            w_config["n_linears"],
            n_grains,
            hop_size,
            w_config["normalize_ola"],
            w_config["pp_chans"],
            w_config["pp_ker"],
            l_grain=l_grain,
            sr=w_config["sr"],
            learning_rate=args.learning_rate,
        )
    w_model.continue_train = args.continue_train
    w_model.to(device)
    w_model.init_beta(args.max_steps, args.tar_beta, beta_steps=args.beta_steps)
    w_model.init_spectral_distances(
        stft_scales=w_config["stft_scales"],
        mel_scales=w_config["mel_scales"],
        spec_power=w_config["spec_power"],
        mel_dist=w_config["mel_dist"],
        log_dist=w_config["log_dist"],
        env_dist=w_config["env_dist"],
        device=device,
    )  # TODO: it seems that scale=512 creates empty mel filterbank ?
    w_model.export_dir = os.path.join(tmp_dir, "exports")  # to write export files

    print("model running on device", w_model.device)
    print("model hyper-parameters", w_model.hparams)

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

    args = vars(args)
    args["classes"] = classes  # make sure the classes are saved in the sorted order used for training

    np.save(os.path.join(tmp_dir, "argparse.npy"), args)
    shutil.move(tmp_dir, os.path.join(curr_dir, args["out_dir"]))
    shutil.rmtree(default_root_dir)
    os.rename(os.path.join(curr_dir, args["out_dir"], "version_0"), default_root_dir)

    # tensorboard --logdir
