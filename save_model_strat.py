from typing import List, Tuple, Union
from flwr.server.client_proxy import ClientProxy
import torch
import flwr as fl
from flwr.common import FitRes, Parameters
from cnn_structure import Net

# Define a custom strategy for federated learning
class SaveModelStrategy(fl.server.strategy.FedAvg):
    # Initialize the neural network model
    net = Net()

    # Override the aggregate_fit method of the FedAvg strategy
    def aggregate_fit(
        self,
        server_round: int,  # The current round number
        results: List[Tuple[Union[ClientProxy, FitRes]]],  # The results from the clients
        failures: List[Tuple[Union[ClientProxy, FitRes], BaseException]],  # Any failures that occurred
    ) -> Tuple[Union[Parameters, dict[str, Union[bool, bytes, float, int, str]]], None]:
        # Call the original aggregate_fit method and get the aggregated parameters and metrics
        param, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Load the aggregated parameters into the neural network model
        self.net.load_state_dict(
            {
                # Convert the parameters to PyTorch tensors
                k: torch.Tensor(v)
                for k, v in zip(
                    # Get the keys from the current state of the model
                    self.net.state_dict().keys(),
                    # Convert the parameters to numpy arrays
                    fl.common.parameters_to_ndarrays(param),
                )
            }
        )

        # Return the aggregated parameters and metrics
        return param, metrics