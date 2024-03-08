from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flwr as fl

from constants import DEVICE
from torchvision.models import mobilenet_v2

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Load a pre-trained MobileNetV2 model
        self.mobilenet = mobilenet_v2(pretrained=True)
        # Replace the last layer
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_ftrs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mobilenet(x)
        x = torch.sigmoid(x)  # Use a sigmoid activation function
        return x
def train(net, trainloader, epochs: int, verbose=False):
    pos_weight = torch.tensor([0.9]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        i = 0
        for batch in trainloader:
            print(i)
            i+=1
            images = batch["image"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float().view(-1, 1)  # Convert labels to Float
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    pos_weight = torch.tensor([0.9]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float().view(-1, 1)  # Convert labels to Float
            # Reshape labels to match the output of the model
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Print out the predictions for this batch
            print("Predictions:", predicted)
            print("True labels:", labels)

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
