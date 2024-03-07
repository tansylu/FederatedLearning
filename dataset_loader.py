import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset

def load_datasets(num_cli_model: int, BATCH_SIZE: int):

    fds = FederatedDataset(dataset="trpakov/chest-xray-classification", subset="full",  partitioners={"train": num_cli_model})

    def apply_transforms(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert the images to PyTorch tensors
                # transforms.Normalize(
                #     (0.1307,), (0.3081,)
                # ),  # Normalize the images using the mean and standard deviation of the MNIST dataset
            ]
        )
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    trainloaders = []
    valloaders = []
    for partition_id in range(num_cli_model):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))

    # Load the test set and apply the transformations
    testset = fds.load_full("test").with_transform(apply_transforms)
    # Create a data loader for the test set
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    # Return the lists of data loaders and the test data loader
    return trainloaders, valloaders, testloader