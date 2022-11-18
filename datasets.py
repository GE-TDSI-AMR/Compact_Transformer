import torch
import wget
import os

from collections import Counter

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
        path = f'{data_sets_path}{url.split("/")[-1]}'
        if not os.path.exists(path):
            wget.download(url, path)


def print10samples():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    data_download()
    train_dataset = datasets.CIFAR100(
        root="datasets",
        train=True,
        transform=ToTensor(),
        download=True
    )

    img, label = train_dataset[0]
    plt.imshow(img.permute(1, 2, 0))
    print(img.shape)
    print(train_dataset.__len__())
    print(train_dataset.classes)
    print(Counter(train_dataset.targets))
