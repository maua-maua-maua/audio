import argparse
import os
from math import floor, sqrt
from pathlib import Path

import librosa as rosa
import numpy as np
from numpy.lib.shape_base import dsplit
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write as write_wav
from torch.utils.data import DataLoader
from tqdm import tqdm
from udls.transforms import Dequantize

dequantize = Dequantize(16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dirpath(path):
    return os.path.abspath(os.path.dirname(path))


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, sample_rate, dequantize) -> None:
        super().__init__()
        self.files = rosa.util.find_files(input_dir)
        self.sample_rate = sample_rate
        self.dequantize = dequantize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        x, sr = rosa.load(file, sr=self.sample_rate)
        if self.dequantize:
            x = dequantize(x)
        return file, x.astype(np.float32)


@torch.inference_mode()
def model_blend(ckpts):
    raves = []
    for ckpt in ckpts:
        raves.append(torch.jit.load(ckpt))
    rave_dict = raves[0].state_dict()
    dtypes = [v.dtype for v in rave_dict.values()]
    n = 1
    for i, r in enumerate(raves[1:]):
        for k, v in r.state_dict().items():
            if "decoder.synth" in k:
                if i == 0:
                    v0 = rave_dict[k]
                    rave_dict[k] = n * v0 / (len(raves) + n - 1)
                rave_dict[k] += v / (len(raves) + n - 1)
    rave_dict = {k: v.to(dtype) for dtype, (k, v) in zip(dtypes, rave_dict.items())}
    del raves

    rave = torch.jit.load(ckpts[0])
    rave.load_state_dict(rave_dict)
    model_name = "_".join([Path(c).stem for c in ckpts])
    return rave, model_name


@torch.inference_mode()
def autoencode(loader, rave, model_name, output_dir):
    for (file,), audio in tqdm(loader, desc=f"Writing autoencoded wavs in {output_dir}"):
        encoded = rave.encode(audio.to(device).unsqueeze(1))
        decoded = rave.decode(encoded)
        write_wav(f"{output_dir}/{Path(file).stem}_{model_name}.wav", 44100, decoded.squeeze().numpy())


def group_by_longest_prefix(iterable):
    """
    given a sorted list of strings, group by longest common prefix
    https://stackoverflow.com/a/11263791
    """

    def common_count(t0, t1):
        """returns the length of the longest common prefix"""
        for i, pair in enumerate(zip(t0, t1)):
            if pair[0] != pair[1]:
                return i
        return i

    longest = 0
    out = []

    for t in iterable:
        if out:  # if there are previous entries

            # determine length of prefix in common with previous line
            common = common_count(t, out[-1])

            # if the current entry has a shorted prefix, output previous
            # entries as a group then start a new group
            if common < longest:
                yield out
                longest = 0
                out = []
            # otherwise, just update the target prefix length
            else:
                longest = common

        # add the current entry to the group
        out.append(t)

    # return remaining entries as the last group
    if out:
        yield out


import random

np.set_printoptions(precision=2, edgeitems=30, linewidth=100000)


def slice(start=None, stop=None, step=1):
    return np.vectorize(lambda x: x[:, :, :, start:stop:step], otypes=[str])


@torch.inference_mode()
def interpolate(loader, rave, model_name, output_dir, density=15, batch_size=32):
    encodings = {
        Path(file).stem: rave.encode(audio.to(device).unsqueeze(1)).cpu()
        for (file,), audio in tqdm(loader, desc="Encoding...")
    }
    groups = list(group_by_longest_prefix(encodings.keys()))
    group_side_len = min([floor(sqrt(len(g))) for g in groups])  # for now just equally-sized square groups supported
    groups = [np.array(random.sample(g, group_side_len**2)).reshape(group_side_len, group_side_len) for g in groups]
    main_side_len = floor(sqrt(len(groups)))
    groups = np.array(groups).reshape(main_side_len, main_side_len, group_side_len, group_side_len)
    groups = groups.transpose(0, 2, 1, 3).reshape(main_side_len * group_side_len, main_side_len * group_side_len)

    print("\nGrid layout:")
    padding = max(len(f) for f in encodings.keys()) + 2
    for row in groups:
        for file in row:
            print(file.ljust(padding), end="")
        print()
    print()

    enc_ex = list(encodings.values())[0]
    latent_dim = enc_ex.shape[-2]
    length = min(e.shape[-1] for e in encodings.values())
    blend_weight = torch.linspace(0, 1, density).reshape(-1, 1, 1).to(enc_ex)

    tl, tr, bl, br = (
        torch.tensor([[[1, 0, 0, 0]]]).to(enc_ex),
        torch.tensor([[[0, 1, 0, 0]]]).to(enc_ex),
        torch.tensor([[[0, 0, 1, 0]]]).to(enc_ex),
        torch.tensor([[[0, 0, 0, 1]]]).to(enc_ex),
    )
    row_t = tl * (1 - blend_weight) + tr * blend_weight
    row_b = bl * (1 - blend_weight) + br * blend_weight
    weights = row_t[None] * (1 - blend_weight[..., None]) + row_b[None] * blend_weight[..., None]
    weights = np.vectorize(lambda x: np.format_float_positional(x, unique=False, precision=2))(
        weights.squeeze().cpu().numpy()
    )
    weights = np.char.array(weights)

    output_grid, output_names = [[]], [[]]
    with tqdm(total=(groups.shape[0] - 1) * (groups.shape[1] - 1), desc="Generating interpolation grid...") as pbar:
        for y, row in enumerate(groups[:-1]):
            for x, file in enumerate(row[:-1]):

                names = [groups[i, j] for i, j in [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]]
                tl, tr, bl, br = [encodings[name][..., :length] for name in names]

                blend_y = blend_weight if y == len(groups) - 2 else blend_weight[:-1]
                blend_x = blend_weight if x == len(groups) - 2 else blend_weight[:-1]

                row_t = tl * (1 - blend_x) + tr * blend_x  # D, latent_dim, length
                row_b = bl * (1 - blend_x) + br * blend_x  # D, latent_dim, length
                grid = row_t[None] * (1 - blend_y[..., None]) + row_b[None] * blend_y[..., None]  # D, D, ld, le
                output_grid[y].append(grid)
                output_names[y].append(
                    weights[: len(blend_y), : len(blend_x)] + np.char.array([[["*"]]]) + np.char.array(names)
                )
                pbar.update()
            output_grid.append([])
            output_names.append([])
    output_grid = torch.cat([torch.cat(grid_row, axis=1) for grid_row in output_grid[:-1]])
    output_names = np.concatenate([np.concatenate(weight_row, axis=1) for weight_row in output_names[:-1]])

    decoded = torch.cat(
        [
            rave.decode(batch.to(device)).to("cpu")
            for batch in tqdm(
                output_grid.reshape(-1, latent_dim, length).split(batch_size), desc="Decoding interpolated grid..."
            )
        ]
    )
    with tqdm(total=output_grid.shape[0] * output_grid.shape[1], desc="Writing wavs...") as pbar:
        for y in range(output_grid.shape[0]):
            for x in range(output_grid.shape[1]):
                out_file = f"{output_dir}/{y}_{x}_{'_'.join(output_names[y, x])}_{model_name}.wav"
                write_wav(out_file, 44100, decoded[y * output_grid.shape[1] + x].squeeze().cpu().numpy())
                pbar.update()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpts", nargs="*", type=str, help="Checkpoint file of RAVE model to use")
    parser.add_argument("-i", "--input_dir", type=str, help="Directory containing audio files to process")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to write result audio files", default=dirpath(__file__) + "/../output/")
    parser.add_argument("-sr", "--sample_rate", default=44100, type=int, help="Sampling rate model was trained with")
    parser.add_argument("-d", "--dequantize", action='store_true', help="Whether a little bit of random noise should be added to audio before encoding")
    parser.add_argument("-ae", "--autoencode", action='store_true', help="Simple autoencoding of samples")
    parser.add_argument("-int", "--interpolate", action='store_true', help="Interpolation grid of samples")
    args = parser.parse_args()
    # fmt: on

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = InferenceDataset(input_dir=args.input_dir, sample_rate=args.sample_rate, dequantize=args.dequantize)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8)

    if len(args.ckpts) > 1:
        rave, model_name = model_blend(args.ckpts)
    else:
        rave = torch.jit.load(args.ckpts[0])
        model_name = Path(args.ckpts[0]).stem
    rave = rave.to(device)

    if args.autoencode:
        autoencode(loader, rave, model_name, args.output_dir)
    elif args.interpolate:
        interpolate(loader, rave, model_name, args.output_dir)
