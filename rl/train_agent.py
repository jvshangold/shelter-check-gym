import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from tax_env.env import TaxEnv

# hyperparameters

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256 # number of cells in each layer i.e. output dim
lr = 3e-4
max_grad_norm = 1.0

# Data collection parameters

frames_per_batch = 1000
# consider upping this later
total_frames = 50000

# PPO parameters

sub_batch_size = 64
num_epochs = 10
clip_epsilon = (
    0.2
)
gamme = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# defining our tax environment

base_env = TaxEnv()


