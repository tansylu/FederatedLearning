import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset

def load_datasets(num_cli_model: int, BATCH_SIZE: int):
    print("Creating OversampledFederatedDataset...")
    fds = FederatedDataset(dataset="trpakov/chest-xray-classification", subset="full",  partitioners={"train": num_cli_model})
    
    def apply_transforms(batch):                                                                                                                                                                                                         
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the images to 224x224
                transforms.ToTensor(),  # Convert the images to PyTorch tensors
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize the images using the mean and standard deviation of the ImageNet dataset
            ]
        )
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    trainloaders = []
    valloaders = []
    for partition_id in range(num_cli_model):
        print(f"Loading partition {partition_id}...")
        partition = fds.load_partition(partition_id, "train")
        print("Applying transforms...")
        partition = partition.with_transform(apply_transforms)
        print("Splitting train and test sets...")
        partition = partition.train_test_split(train_size=0.8)
        print("Creating data loaders...")
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))

    # Load the test set and apply the transformations
    print("Loading test set...")
    testset = fds.load_full("test").with_transform(apply_transforms)
    # Create a data loader for the test set
    print("Creating test data loader...")
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    # Return the lists of data loaders and the test data loader
    print("Finished loading datasets.")
    return trainloaders, valloaders, testloader