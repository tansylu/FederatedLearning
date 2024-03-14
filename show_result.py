from matplotlib import pyplot as plt
from cnn_structure import Net


# trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, BATCH_SIZE)

# batch = next(iter(trainloaders[0]))
# images, labels = batch["image"], batch["label"]
# # Reshape and convert images to a NumPy array
# # matplotlib requires images with the shape (height, width, 3)
# images = images.permute(0, 2, 3, 1).numpy()
# # Denormalize
# images = images / 2 + 0.5

# # Create a figure and a grid of subplots
# fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# # Loop over the images and plot them
# for i, ax in enumerate(axs.flat):
#     ax.imshow(images[i])
#     ax.set_title(trainloaders[0].dataset.features["label"].int2str([labels[i]])[0])
#     ax.axis("off")

# # Show the plot
# fig.tight_layout()
# plt.show()


def save_result(batch, net: Net):
    images, labels = batch["image"], batch["label"]

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    net.eval()
    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i][0])
        ax.set_title(
            "Guess: "
            + str(most_likely(net, images[i]))
            + " - Label: "
            + str(labels[i].item())
        )
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()


def most_likely(net: Net, img) -> int:
    r = net(img)
    m = r[0][0].item()
    mi = 0
    for n, i in zip(range(len(r[0])), list(r[0])):
        if i > m:
            m = i
            mi = n
    return mi
