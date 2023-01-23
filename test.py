import torch

import src.image as models
import os

from trainer import TrainingContext, Trainer


def reload():
    checkpoint_path = 'saved/'

    args_dataset = "cifar100"

    args_model = 'cct_7_3x1_32_c100'
    model = models.__dict__[args_model](pretrained=False)

    max_epoch = max([
        int(file.split(".")[0].split("epoch")[1]) for file in os.listdir(checkpoint_path)
        if file.startswith(f"dataset-{args_dataset}_model-{args_model}_epoch")
    ])
    model_path = f"{checkpoint_path}dataset-{args_dataset}_model-{args_model}_epoch{max_epoch}.pt"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    return model


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


if __name__ == "__main__":
    reload()
    print("reoaeldd")
    model_name = 'cct_7_3x1_32_c100'
    dataset_name = 'CIFAR100'

    t_context = TrainingContext(
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=256,
        workers=4,
        pretrained=False,
        load_model=True,
        checkpoint_path="saved/",
        save_freq=10,
        saving=True,
        print_freq=10,
        printing=True
    )

    cct = Trainer(
        context=t_context,
        no_cuda=True,
        gpu_id=0,
        lr=5e-4,
        weight_decay=1e-4,
        epochs=200,
        warmup=5,
        disable_cos=True,
        clip_grad_norm=0,

    )

    cct.training()
