import gzip
import os
import shutil
from collections import Counter

import torch
import wget
from matplotlib import pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import src as models

data_sets_urls = {
    'cifar10':
        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100':
        'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
}

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}

data_sets_path = 'datasets/'


def gz_extract(directory):
    extension = ".gz"
    for item in os.listdir(directory):  # loop through items in dir
        if item.endswith(extension):  # check for ".gz" extension
            print(item)
            splitted = item.split(".")[0]  # get file name for file within
            file_name = f"{splitted[0]}"
            with gzip.open(f"{directory}/{item}", "rb") as f_in, open(f"{directory}/{file_name}", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            # os.remove(gz_name) # delete zipped file


def data_download():
    try:
        os.mkdir(data_sets_path)
    except OSError as error:
        print("/datasets folder already exist")

    for data_set, url in data_sets_urls.items():
        path = f'{data_sets_path}{url.split("/")[-1]}'
        if not os.path.exists(path):
            print(path)
            wget.download(url, path)
            print(f"downloaded {url}")


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


def dataset_exploration(dataset):
    img, label = dataset[0]
    plt.imshow(img.permute(1, 2, 0))
    print(img.shape)
    print(dataset.__len__())
    print(dataset.classes)
    print(Counter(dataset.targets))


if __name__ == "__main__":
    data_download()

    # dataset getting
    args_dataset = "cifar100"

    img_size = DATASETS[args_dataset]['img_size']
    num_classes = DATASETS[args_dataset]['num_classes']
    img_mean, img_std = DATASETS[args_dataset]['mean'], DATASETS[args_dataset]['std']

    train_dataset = datasets.__dict__[args_dataset.upper()](
        root=data_sets_path,
        train=True,
        transform=ToTensor(),
        download=True
    )

    val_dataset = datasets.__dict__[args_dataset.upper()](
        root=data_sets_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )
    )

    # model getting
    args_model = 'cct_7_3x1_32_c100'

    model = models.__dict__[args_model]()
