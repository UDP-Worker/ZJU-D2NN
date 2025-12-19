import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
import torch.utils as ptu

"""
Train a neural network or use a normal function to describe the propagating process
from SLM1 to SLM2. Also train a neural network or use a normal function to describe
the propagating process from SLM2 to camera.

To train these two neural network, we can utilize programmable SLM2.
"""
class FreePropagation(nn.Module):
    def __init__(
            self,
            distance: int,
            learning_rate: float
    ):
        super().__init__()

        self.distance = distance

        parameters = (distance,)
        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

    def forward(self, inputE: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward process with free propagation length
        raise NotImplementedError

class LenPropagation(nn.Module):

