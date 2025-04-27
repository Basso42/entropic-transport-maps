from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import numpy as np

def load_mnist(dataset_name='fashion-mnist', classes=(0,1), n_samples_per_class=1000, data_dir='./data'):
    if dataset_name == 'mnist':
        dataset = datasets.MNIST
    elif dataset_name == 'fashion-mnist':
        dataset = datasets.FashionMNIST
    else:
        raise ValueError("Dataset not supported: choose 'mnist' or 'fashion-mnist'")

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = dataset(root=data_dir, train=True, download=True, transform=transform)

    class_0_idx = (train_data.targets == classes[0]).nonzero().squeeze()
    class_1_idx = (train_data.targets == classes[1]).nonzero().squeeze()

    idx_0 = class_0_idx[torch.randperm(len(class_0_idx))[:n_samples_per_class]]
    idx_1 = class_1_idx[torch.randperm(len(class_1_idx))[:n_samples_per_class]]

    images_0 = train_data.data[idx_0]
    images_1 = train_data.data[idx_1]

    return images_0.float()/255., images_1.float()/255.

def load_cifar(classes=(0,1), n_samples_per_class=1000, data_dir='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    class_0_idx = [i for i, label in enumerate(train_data.targets) if label == classes[0]]
    class_1_idx = [i for i, label in enumerate(train_data.targets) if label == classes[1]]

    idx_0 = torch.tensor(class_0_idx)[torch.randperm(len(class_0_idx))[:n_samples_per_class]]
    idx_1 = torch.tensor(class_1_idx)[torch.randperm(len(class_1_idx))[:n_samples_per_class]]

    images_0 = torch.stack([train_data[i][0] for i in idx_0])
    images_1 = torch.stack([train_data[i][0] for i in idx_1])

    return images_0, images_1

def plot_2d(X, y=None, title="Dataset"):
    plt.figure(figsize=(6, 6))
    if y is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=5)
    else:
        plt.scatter(X[:, 0], X[:, 1], s=5)
    plt.title(title)
    plt.show()