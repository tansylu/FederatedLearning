import torch

import flwr as fl


DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
# disable_progress_bar()

from sys import argv

assert len(argv) == 5

NUM_CLIENTS = int(argv[1])  # 25
BATCH_SIZE = int(argv[2])  # 32
NUM_ROUNDS = int(argv[3])  # 2
NUM_EPOCHS = int(argv[4])  # 2
