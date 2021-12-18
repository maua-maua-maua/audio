import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from torch.utils.data import DataLoader
from udls import SimpleDataset, simple_audio_preprocess
from udls.transforms import Compose, Dequantize, RandomCrop

with torch.inference_mode():
    rave = torch.jit.load("rave_20k_bass_dnb_samples_last-v3.ts")

    dataset = SimpleDataset(
        "/home/hans/trainsets/rave/20k_bass_dnb_samples/rave",
        "/home/hans/datasets/music-samples/train",
        preprocess_function=simple_audio_preprocess(41000, 2 * 65536),
        split_set="full",
        transforms=Compose([RandomCrop(65536), Dequantize(16), lambda x: x.astype(np.float32)]),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=8)

    batch = next(iter(loader)).unsqueeze(1)
    print(batch.shape)

    encoded = rave.encode(batch)
    print(encoded.shape)

    decoded = rave.decode(encoded)
    print(decoded.shape)

    write_wav("/home/hans/datasets/music-samples/generations/original.wav", 41000, batch.squeeze().flatten().numpy())
    write_wav("/home/hans/datasets/music-samples/generations/decoded.wav", 41000, decoded.squeeze().flatten().numpy())
