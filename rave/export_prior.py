import torch

torch.set_grad_enabled(False)
import logging
import math
from glob import glob
from os import path

import torch.nn as nn
from cached_conv import use_buffer_conv
from effortless_config import Config
from termcolor import colored

logging.basicConfig(level=logging.INFO, format=colored("[%(relativeCreated).2f] ", "green") + "%(message)s")

logging.info("exporting model")


class args(Config):
    RUN = None
    NAME = "latent"


args.parse_args()
use_buffer_conv(True)

from cached_conv import *
from prior.model import Model

from rave.core import search_for_run


class TraceModel(nn.Module):
    def __init__(self, pretrained: Model):
        super().__init__()
        data_size = pretrained.data_size

        self.data_size = data_size
        self.pretrained = pretrained

        x = torch.zeros(1, 1, 2 ** 14)
        z = self.pretrained.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

        self.register_buffer(
            "forward_params",
            torch.tensor([1, ratio, data_size, ratio]),
        )

        self.pretrained.synth = None

        self.register_buffer(
            "previous_step",
            self.pretrained.quantized_normal.encode(torch.zeros(1, data_size, 1)),
        )

        self.pre_diag_cache = CachedPadding1d(data_size - 1)
        self.pre_diag_cache(z)
        self.pre_diag_cache = torch.jit.script(self.pre_diag_cache)

    def step_forward(self, temp):
        # PREDICT NEXT STEP
        x = self.pretrained.forward(self.previous_step)
        x = x / temp
        x = self.pretrained.post_process_prediction(x, argmax=False)
        self.previous_step.copy_(x.clone())

        # DECODE AND SHIFT PREDICTION
        x = self.pretrained.quantized_normal.decode(x)
        x = self.pre_diag_cache(x)
        x = self.pretrained.diagonal_shift.inverse(x)

        return x

    def forward(self, temp: torch.Tensor):
        x = torch.zeros(
            temp.shape[0],
            self.data_size,
            temp.shape[-1],
        ).to(temp)

        temp = temp.mean(-1, keepdim=True)
        temp = nn.functional.softplus(temp) / math.log(2)

        for i in range(x.shape[-1]):
            x[..., i : i + 1] = self.step_forward(temp)

        return x


logging.info("loading model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
model = Model.load_from_checkpoint(RUN, strict=False).eval()

logging.info("warmup forward pass")

x = torch.zeros(1, 1, 2 ** 17)
x = model.encode(x)
x = torch.zeros_like(x)
x = model.quantized_normal.encode(model.diagonal_shift(x))
x = x[..., -1:]
model(x)

logging.info("scripting cached modules")
n_cache = 0

cached_modules = [
    CachedConv1d,
]

for n, m in model.named_modules():
    if any(list(map(lambda c: isinstance(m, c), cached_modules))):
        m.script_cache()
        n_cache += 1

logging.info(f"{n_cache} cached modules found and scripted")

logging.info("script model")
model = TraceModel(model)
model = torch.jit.script(model)

logging.info("save model")
model.save(f"prior_{args.NAME}.ts")
