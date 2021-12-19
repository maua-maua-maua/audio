# examples of creating model configurations
from .hierarchical import HierarchicalModel
from .latent import LatentModel
from .layers import *
from .utils import *
from .waveform import WaveformModel

"""
import json
l_config = dict()
l_config["e_dim"] = 256
l_config["rnn_type"] = "LSTM"
l_config["n_RNN"] = 1
l_config["h_dim"] = 512
l_config["n_linears"] = 2
with open("./configs/l_E256_1LSTM.json", 'w') as f:
    json.dump(l_config, f, sort_keys=True, indent=4)
"""

"""
import json
w_config = dict()
w_config["sr"] = 22050
w_config["l_grain"] = 2048
w_config["silent_reject"] = [0.2,0.2] # first value is minimum peak amplitude, second is minimum non-silent length ratio to target length (0=False)
# or simpler rejection sampling e.g. https://github.com/NVIDIA/waveglow/issues/155#issuecomment-531029586 ?
w_config["amplitude_norm"] = False # amplitude normalization of training files
w_config["normalize_ola"] = True # normalization of the overlap-add output of the model
w_config["mel_dist"] = True
w_config["log_dist"] = 0. # scale factor of the log-magnitude distances (0=False)
w_config["spec_power"] = 1
w_config["env_dist"] = 0 # scale factor of the envelope distance (0=False)
w_config["stft_scales"] = [2048, 1024, 512, 256]
w_config["mel_scales"] = [2048, 1024]

w_config["z_dim"] = 128
w_config["h_dim"] = 512
w_config["kernel_size"] = 9
w_config["channels"] = 128
w_config["n_convs"] = 3
w_config["stride"] = 4
w_config["n_linears"] = 3
w_config["pp_chans"] = 5
w_config["pp_ker"] = 65
with open("./configs/w_22k_L2048_Reject0202_normola.json", 'w') as f:
    json.dump(w_config, f, sort_keys=True, indent=4)
"""
