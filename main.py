from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from cnn_structure import FlowerClient, Net, get_parameters, test, train
from constants import BATCH_SIZE, DEVICE, NUM_CLIENTS
from dataset_loader import load_datasets

print("Loading datasets...")
trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, BATCH_SIZE)
print("Datasets loaded.")

# ## CNN

# trainloader = trainloaders[0]
# valloader = valloaders[0]
# net = Net().to(DEVICE)

# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# for epoch in range(5):
#     print(f"Starting epoch {epoch+1}...")
#     train(net, trainloader, 1)
#     print(f"Finished training for epoch {epoch+1}. Starting testing...")
#     loss, accuracy, labels, predictions = test(net, valloader)
#     precision = precision_score(labels, predictions)
#     recall = recall_score(labels, predictions)
#     f1 = f1_score(labels, predictions)
#     auc_roc = roc_auc_score(labels, predictions)
#     print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}, precision {precision}, recall {recall}, F1 {f1}, AUC-ROC {auc_roc}")

print("Creating clients...")
def gen_client(cid: str) -> FlowerClient:
    # Create a client representing a single institution
    net = Net().to(DEVICE)
    # each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(net, trainloader, valloader).to_client()
print("Clients created.")

def gen_client(cid: str) -> FlowerClient:
    print('GEN CLIENT')
    # Create a client representing a single institution
    net = Net().to(DEVICE)
    # each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(net, trainloader, valloader).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print("AVERAGE")
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

params = get_parameters(Net())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=gen_client,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

