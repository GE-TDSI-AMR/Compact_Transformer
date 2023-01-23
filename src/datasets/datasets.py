import medmnist

from torchvision.transforms import transforms
import torchvision.datasets as datasets

datasets_names = sorted(
    name for name in [*datasets.__all__, *medmnist.INFO]
    if not name.startswith("_")
)

datasets_path = "datasets/"

INFOS = {
    'CIFAR10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'CIFAR100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}


def tv_ds(data_class):
    img_size = None
    if data_class in INFOS:
        img_size = INFOS[data_class]['img_size']
        img_mean = INFOS[data_class]['mean']
        img_std = INFOS[data_class]['std']
    else:
        print(f"{data_class} is not present in the INFOS dictionary")
        img_mean = 0.5
        img_std = 0.5

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    cropping = [] if img_size is None else [transforms.RandomCrop(img_size, padding=4)]

    train_dataset = data_class(
        root=datasets_path, train=True, download=True, transform=transforms.Compose(
            [
                *augmentation(data_class),
                *[
                    *cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    *normalize,
                ]
            ]
        )
    )

    val_dataset = data_class(
        root=datasets_path, train=False, download=True, transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                *normalize,
            ]
        )
    )

    return train_dataset, val_dataset


def medmnist_ds(data_class):
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = data_class(split='train', transform=data_transform, download=True)
    val_dataset = data_class(split='test', transform=data_transform, download=True)

    return train_dataset, val_dataset


def ds(dataset_name):
    if dataset_name in datasets.__all__:
        data_class = datasets.__dict__[dataset_name]
        return tv_ds(data_class)
    elif dataset_name in medmnist.INFO:
        data_class = getattr(medmnist, medmnist.INFO[dataset_name]['python_class'])
        return medmnist_ds(data_class)
    else:
        print(f"{dataset_name} is not available")
        raise NotImplementedError


def available_datasets():
    print(*datasets_names)


def augmentation(data_class):

    augmentations = []
    if data_class in ["CIFAR10", "CIFAR100"]:
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]

    return augmentations


if __name__ == "__main__":
    available_datasets()
