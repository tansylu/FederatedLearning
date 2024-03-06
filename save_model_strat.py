from collections import OrderedDict
from typing import List, Tuple
from flwr.server.client_proxy import ClientProxy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import FitRes, Metrics, Parameters
from flwr_datasets import FederatedDataset

from cnn_structure import Net

class SaveModelStrategy(fl.server.strategy.FedAvg):
    net = Net()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy | FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Parameters | dict[str, bool | bytes | float | int | str] | None]:
        param, metrics = super().aggregate_fit(server_round, results, failures)
        self.net.load_state_dict(
            {
                k: torch.Tensor(v)
                for k, v in zip(
                    self.net.state_dict().keys(),
                    fl.common.parameters_to_ndarrays(param),
                )
            }
        )

        return param, metrics

    pass
