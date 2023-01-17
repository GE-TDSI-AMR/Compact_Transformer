import gzip
import os
import shutil
from collections import Counter
from time import time
import math

import torch
import torch.nn as nn
import torch.utils.data
import wget
from matplotlib import pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms

from utils.losses import LabelSmoothingCrossEntropy

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
checkpoint_path = 'saved/'


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


def print10samples(train_dataset):
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


def adjust_learning_rate(optimizer, epoch, args_lr, args_epochs, args_warmup, args_disable_cos):
    lr = args_lr
    if args_warmup is not None and epoch < args_warmup:
        lr = lr / (args_warmup - epoch)
    elif not args_disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args_warmup) / (args_epochs - args_warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_train(train_loader, model, criterion, optimizer, epoch, args_no_cuda, args_gpu_id, args_clip_grad_norm, print_freq=10):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if (not args_no_cuda) and torch.cuda.is_available():
            images = images.cuda(args_gpu_id, non_blocking=True)
            target = target.cuda(args_gpu_id, non_blocking=True)
        output = model(images)

        loss = criterion(output, target)

        acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args_clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args_clip_grad_norm, norm_type=2)

        optimizer.step()

        if print_freq >= 0 and i % print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            print(f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def cls_validate(val_loader, model, criterion, args_no_cuda, args_gpu_id, epoch=None, time_begin=None, print_freq=10):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args_no_cuda) and torch.cuda.is_available():
                images = images.cuda(args_gpu_id, non_blocking=True)
                target = target.cuda(args_gpu_id, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if print_freq >= 0 and i % print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                print(f'[Epoch {epoch + 1}][Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc1


def training(model, train_loader, val_loader, print_freq=10):
    #           gpu parameters
    args_no_cuda = True
    args_gpu_id = 0

    #           training parameters
    args_lr = 5e-4
    args_weight_decay = 1e-4
    args_epochs = 200

    args_warmup = False
    args_disable_cos = True
    args_clip_grad_norm = 0

    #           training components
    criterion = LabelSmoothingCrossEntropy()

    if (not args_no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args_gpu_id)
        model.cuda(args_gpu_id)
        criterion = criterion.cuda(args_gpu_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args_lr,
        weight_decay=args_weight_decay
    )

    #           training execution
    best_acc1 = 0
    acc1 = 0
    print("Beginning training")
    time_begin = time()
    for epoch in range(args_epochs):
        adjust_learning_rate(optimizer, epoch, args_lr, args_epochs, args_warmup, args_disable_cos)
        cls_train(train_loader, model, criterion, optimizer, epoch, args_no_cuda, args_gpu_id, args_clip_grad_norm, print_freq)
        acc1 = cls_validate(
            val_loader, model, criterion, args_no_cuda, args_gpu_id,
            epoch=epoch, time_begin=time_begin, print_freq=10
        )
        best_acc1 = max(acc1, best_acc1)

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'final top-1: {acc1:.2f}')
    torch.save(model.state_dict(), checkpoint_path)

    return best_acc1


def train():
    #           dataset parameters
    data_download()

    args_dataset = "cifar100"

    img_size = DATASETS[args_dataset]['img_size']
    num_classes = DATASETS[args_dataset]['num_classes']
    img_mean, img_std = DATASETS[args_dataset]['mean'], DATASETS[args_dataset]['std']

    #           augmentation
    args_disable_aug = False

    augmentations = []
    if not args_disable_aug:
        from utils.autoaug import CIFAR10Policy

        augmentations += [
            CIFAR10Policy()
        ]
    augmentations += [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ]
    augmentations = transforms.Compose(augmentations)

    #           datasets creation
    train_dataset = datasets.__dict__[args_dataset.upper()](
        root=data_sets_path,
        train=True,
        transform=augmentations,
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

    #           batch parameters
    args_batch_size = 128
    args_workers = 4

    #           data_loaders creation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args_batch_size,
        shuffle=True,
        num_workers=args_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args_batch_size,
        shuffle=False,
        num_workers=args_workers
    )

    #           model definition
    args_model = 'cct_7_3x1_32_c100'

    model = models.__dict__[args_model](pretrained=False)

    #           training
    print_freq = 1
    best_acc1 = training(model, train_loader, val_loader, print_freq)


if __name__ == "__main__":
    train()
