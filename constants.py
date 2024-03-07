import torch

import flwr as fl


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
# disable_progress_bar()

NUM_CLIENTS = 5
BATCH_SIZE = 50
