from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST, PathMNIST
import os

DATASETS = [
    "cifar10", 
    "pneumonia",
    "path",
    "tiny_imagenet",
]

def get_dataloaders(dataset_name, batch_size, image_size, DATA_DIR, normalize=False):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if dataset_name in ["cifar10", "tiny_imagenet"]:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

    elif dataset_name in ["pneumonia", "path"]:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    transform = transforms.Compose(transform_list)

    if dataset_name == "cifar10":
        train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == "cifar100":
        train = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
        num_classes = 100

    elif dataset_name == "pneumonia":
        train = PneumoniaMNIST(split="train", download=True, transform=transform, root=DATA_DIR)
        test = PneumoniaMNIST(split="test", download=True, transform=transform, root=DATA_DIR)
        num_classes = 2

    elif dataset_name == "path":
        train = PathMNIST(split="train", download=True, transform=transform, root=DATA_DIR)
        test = PathMNIST(split="test", download=True, transform=transform, root=DATA_DIR)
        num_classes = 9

    elif dataset_name == "tiny_imagenet":

        train = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, "tiny-imagenet-200", "train"),
            transform=transform
        )

        test = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, "tiny-imagenet-200", "val"),
            transform=transform
        )

        num_classes = 200

    else:
        raise ValueError()

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2),
        num_classes
    )