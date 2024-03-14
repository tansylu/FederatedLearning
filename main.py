from typing import List, Tuple


import flwr as fl
from flwr.common import Metrics
from cnn_structure import FlowerClient, Net, test, train
from constants import BATCH_SIZE, DEVICE, NUM_CLIENTS, NUM_ROUNDS
from dataset_loader import load_datasets
from save_model_strat import SaveModelStrategy
from show_result import save_result


trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, BATCH_SIZE)


## CNN

trainloader = trainloaders[0]
valloader = valloaders[0]
# net = Net().to(DEVICE)


# for epoch in range(5):
#     train(net, trainloader, 1)
#     loss, accuracy = test(net, valloader)
#     print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

# loss, accuracy = test(net, testloader)
# print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


## Federated learning


def gen_client(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader).to_client()


# Create FedAvg strategy
# TODO: Research what other strategy we can use

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 5, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Create FedAvg strategy
# FedAvgM is average with momentum. This makes it so that, with the
# additional momentum needed makes it better when the data in the diferent
# submodels differs more, since it makes it so that the server model is
# less likely to get stuck between 2 clusters of good results (like, if
# you graph the models according to their config, and there are 2 clusters
# of good models, there is the risk of the avg getting the server stuck
# in the middle)

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=gen_client,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,
)

fb = next(iter(testloader))

save_result(fb, strategy.net)
