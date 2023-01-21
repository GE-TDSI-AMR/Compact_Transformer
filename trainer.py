import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

import src as models


@dataclass
class TrainingContext:
    # Datasets
    dataset_name: str
    train_dataset: Dataset
    val_dataset: Dataset

    # Data Loaders
    train_loader: DataLoader = field(init=False)
    val_loader: DataLoader = field(init=False)

    # Model
    model_name: str.lower
    model: nn.Module = field(init=False)

    # Batch Params
    batch_size: int = 128
    workers: int = 4

    # Model saving and reloading
    pretrained: bool = False
    load_model: bool = False
    checkpoint_path: str = "saved/"
    epoch: int = field(default=0, init=False)
    save_freq: int = 10
    saving: bool = True

    # Printing
    print_freq: int = 10
    printing: bool = True

    def __post_init__(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )

        self.model = models.__dict__[self.model_name](pretrained=self.pretrained)
        if self.load_model:
            self.model_loading()

    def model_loading(self):
        checkpoint_epochs = [
            int(file.split(".")[0].split("epoch")[1]) for file in os.listdir(self.checkpoint_path)
            if file.startswith(f"dataset-{self.dataset_name}_model-{self.model_name}_epoch")
        ]
        self.epoch = max(checkpoint_epochs) if len(checkpoint_epochs) != 0 else 0
        if self.epoch != 0:
            model_path = f"{self.checkpoint_path}dataset-{self.dataset_name}_model-{self.model_name}_epoch{self.epoch}.pt"
            print(f"Loading {model_path} to model")
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No model found to load")

    def model_saving(self):
        file_name = f"/dataset-{self.dataset_name}_" \
                    f"model-{self.model_name}_" \
                    f"epoch{self.epoch + 1}.pt"
        torch.save(
            self.model.state_dict(),
            self.checkpoint_path + file_name
        )
        print(
            file_name + " was saved in " + self.checkpoint_path
        )


@dataclass
class Trainer:
    context: TrainingContext

    # gpu parameters
    no_cuda: bool = False
    gpu_id: int = 0
    cuda_usage: bool = field(init=False)

    # training parameters
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200

    # specific parameters
    warmup: int = 5
    disable_cos: bool = True
    clip_grad_norm: int = 0

    # training components
    criterion: nn.Module = field(init=False)
    optimizer: Optimizer = field(init=False)

    def __post_init__(self):
        self.cuda_usage = (not self.no_cuda) and torch.cuda.is_available()
        if self.cuda_usage:
            self.cuda_activation()

    def cuda_activation(self):
        torch.cuda.set_device(self.gpu_id)
        self.context.model.cuda(self.gpu_id)
        self.criterion = self.criterion.cuda(self.gpu_id)

    def data_to_cuda(self, images, target):
        images = images.cuda(self.gpu_id, non_blocking=True)
        target = target.cuda(self.gpu_id, non_blocking=True)
        return images, target

    def adjust_learning_rate(self):
        lr = self.lr
        if self.warmup is not None and self.context.epoch < self.warmup:
            lr = lr / (self.warmup - self.context.epoch)
        elif not self.disable_cos:
            lr *= 0.5 * (1. + math.cos(math.pi * (self.context.epoch - self.warmup) / (self.epochs - self.warmup)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch(self, train: bool):
        if train:
            self.context.model.train()
        else:
            self.context.model.eval()

        loss_val, acc1_val = 0, 0
        n = 0

        with torch.no_grad() if not train else nullcontext():
            for i, (images, target) in enumerate(self.context.train_loader):
                if self.cuda_usage:
                    images, target = self.data_to_cuda(images, target)

                output = self.context.model(images)
                loss = self.criterion(output, target)
                acc1 = self.accuracy(output, target)

                n += images.size(0)
                loss_val += float(loss.item() * images.size(0))
                acc1_val += float(acc1[0] * images.size(0))

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.context.model.parameters(), max_norm=self.clip_grad_norm, norm_type=2)

                    self.optimizer.step()

                if self.conditional_freq(self.context.printing, i, self.context.print_freq):
                    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                    print(
                        f'[Epoch {self.context.epoch + 1}][{"Train" if train else "Eval"}][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}'
                    )

            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            return avg_loss, avg_acc1

    def training(self):
        #           training execution
        best_acc1 = 0
        print("Beginning training")
        time_begin = time()
        while self.context.epoch-1 <= self.epochs:
            self.context.epoch += 1
            self.adjust_learning_rate()
            avg_train_loss, avg_train_acc1 = self.epoch(train=True)
            avg_val_loss, avg_val_acc1 = self.epoch(train=False)

            total_time = (time() - time_begin) / 60
            print(f'[Epoch {self.context.epoch + 1}] \t \t Top-1 {avg_val_acc1:6.2f} \t \t Time: {total_time:.2f}')
            best_acc1 = max(avg_val_acc1, best_acc1)

            if self.conditional_freq(self.context.saving, self.context.epoch+1, self.context.save_freq):
                self.context.model_saving()

        total_time = (time() - time_begin) / 60
        print(f'Script finished in {total_time:.2f} minutes, '
              f'best top-1: {best_acc1:.2f}, '
              f'final top-1: {avg_val_acc1:.2f}')
        self.context.model_saving()

        return best_acc1

    @staticmethod
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

    @staticmethod
    def conditional_freq(activation, param, freq):
        return activation and freq >= 0 and param % freq == 0
