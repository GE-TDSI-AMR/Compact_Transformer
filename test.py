import torch
from torch import tensor
from torchvision.transforms import transforms

import src.image as models
import os

import vizualisation
from src.datasets.datasets import available_datasets
from trainer import TrainingContext, Trainer


def reload():
    checkpoint_path = 'saved/'

    args_dataset = "cifar100"

    args_model = 'cct_7_3x10_32_c100'
    model = models.__dict__[args_model](pretrained=False)
    print(model)

    max_epoch = max([
        int(file.split(".")[0].split("epoch")[1]) for file in os.listdir(checkpoint_path)
        if file.startswith(f"dataset-{args_dataset}_model-{args_model}_epoch")
    ])
    model_path = f"{checkpoint_path}dataset-{args_dataset}_model-{args_model}_epoch{max_epoch}.pt"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    return model


def print_model(model_name='cct_7_4x10_32_c100'):
    model = models.__dict__[model_name](pretrained=False)
    print(model)


def cct_naming(
    layers,
    kernel_size,
    conv_layers,
    img_size,
    positional_embedding,
    classes
):
    pos_emb = f"_{positional_embedding}_" if positional_embedding != "learnable" else ""
    name = f"cct_{layers}_{kernel_size}x{conv_layers}_{img_size}{pos_emb}_{classes}"
    return name


def model_naming(model_type: str, model_params: dict):
    return globals()[model_type + "_naming"](**model_params)


def tensor_manip():
    a = tensor([10])
    tr = lambda x: tensor(*x)
    target_transform = transforms.Compose([tr])
    print(target_transform(a))


def training_mnist():
    available_datasets()
    model_name = 'cct_7_3x1_28_9'
    dataset_name = 'pathmnist'

    t_context = TrainingContext(
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=256,
        workers=4,
        pretrained=False,
        load_model=False,
        checkpoint_path="saved/",
        save_freq=5,
        saving=True,
        print_freq=10,
        printing=True
    )

    print(t_context.train_dataset)
    for img, label in t_context.train_loader:
        print(type(label))
        break

    cct = Trainer(
        context=t_context,
        no_cuda=False,
        gpu_id=0,
        lr=55e-5,
        weight_decay=6e-2,
        epochs=100,
        warmup=2,
        disable_cos=True,
        clip_grad_norm=0,
    )

    cct.training()


def viz(file_name):
    # training_mnist()
    h = torch.load(file_name)
    print(h)
    data = [list(ele) for ele in list(zip(*h["history"]))]
    print(data)
    vizualisation.vis_accuracy(
        param=range(len(h["history"])), train_acc=data[1], test_acc=data[3], param_name="Accuracy"
    )
    vizualisation.vis_accuracy(
        param=range(len(h["history"])), train_acc=data[0], test_acc=data[2], param_name="Loss"
    )


if __name__ == "__main__":
    training_mnist()
