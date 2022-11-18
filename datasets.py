import wget
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

data_sets_urls = {
    'cifar10':
        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100':
        'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
}

data_sets_path = 'datasets/'


def data_download():
    try:
        os.mkdir(data_sets_path)
    except OSError as error:
        print("/datasets folder already exist")

    for data_set, url in data_sets_urls.items():
        path = f'{data_sets_path}{data_set}.tar.gz'
        if not os.path.exists(path):
            wget.download(url, path)


if __name__ == "__main__":
    data_download()
    training_data = datasets.CIFAR10(root="datasets", train=True, download=True, transform=ToTensor())
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title('')
        plt.axis("off")
        plt.imshow(img.permute(1,2,0))
    plt.show()


    batch_size = 64
    '''
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    '''