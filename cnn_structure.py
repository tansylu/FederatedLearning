from collections import OrderedDict
import math
from typing import List
from sklearn.utils.class_weight import compute_class_weight
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
    criterion = torch.nn.BCEWithLogitsLoss()
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
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

import matplotlib.pyplot as plt

def test(net, testloader):
    criterion = torch.nn.BCEWithLogitsLoss()
    """Evaluate the network on the entire test set."""
    net.eval()
    correct = 0
    total = 0
    loss_total = 0
    labels_total = []
    predictions_total = []
    flag = False
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images= data["image"]
            labels = data["labels"]
            labels = labels.view(-1, 1).float()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            # Convert tensors to lists and add to total
            labels_total += labels.tolist()
            predictions_total += predicted.tolist()

            # Print outputs
            print(f"Outputs: {outputs}")
            print(f"Predicted: {predicted}")

            # Plot and save image, label, and prediction
            grid_size = math.isqrt(len(images))
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
            if not flag:
                for j, ax in enumerate(axs.flatten()):
                    if j < len(images) and isinstance(ax, plt.Axes):
                        img = images[j].permute((1, 2, 0))  # Move channels dimension to the end
                        ax.imshow(img.cpu().numpy())
                        ax.set_title(f"L: {labels[j].item()}, G: {predicted[j].item()}")
                plt.savefig(f"output_{i}.png")
                flag = True

    accuracy = correct / total
    loss_avg = loss_total / total
    return loss_avg, accuracy, labels_total, predictions_total


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    print('Getting parameters...')
    parameters = []
    for name, val in net.state_dict().items():
        param_numpy = val.cpu().numpy()
        parameters.append(param_numpy)
        print(f"Parameter {name}: shape {param_numpy.shape}, size {param_numpy.size}")
    print(f"Total number of parameters: {sum(param.size for param in parameters)}")
    return parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print("Getting parameters...")
        params = get_parameters(self.net)
        print("Parameters:")
        return params

    def fit(self, parameters, config):
        print("Setting parameters...")
        set_parameters(self.net, parameters)
        print("Starting training...")
        train(self.net, self.trainloader, epochs=1)
        params = get_parameters(self.net)
        print("Parameters after training:")
        return params, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print("Setting parameters for evaluation...")
        set_parameters(self.net, parameters)
        print("Starting evaluation...")
        loss, accuracy, labels, predictions = test(self.net, self.valloader)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}