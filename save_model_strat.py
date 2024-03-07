from typing import List, Tuple, Union
from flwr.server.client_proxy import ClientProxy

import torch

import flwr as fl
from flwr.common import FitRes, Parameters

from cnn_structure import Net


class SaveModelStrategy(fl.server.strategy.FedAvg):
    net = Net()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Union[ClientProxy, FitRes]]],
        failures: List[Tuple[Union[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Union[Parameters, dict[str, Union[bool, bytes, float, int, str]]], None]:
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