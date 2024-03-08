from matplotlib import pyplot as plt
import torch
from cnn_structure import Net
from constants import DEVICE

def show_result(batch, net: Net):
    images, labels = batch["image"], batch["labels"]
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    net.eval()
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].cpu().numpy()[0], cmap='gray')
        ax.set_title(
            "Guess: "
            + str(predicted[i].item())
            + " - Label: "
            + str(labels[i].item())
        )
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()