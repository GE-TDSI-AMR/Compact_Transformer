import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random

from torchvision.io import read_image


class LIDC_CROPS(Dataset):
    images = []
    labels = []

    def __init__(self, dataset_location="data/test/", transform=None, masks_count=4):
        print("Start loading")
        self.transform = transform
        self.masks_count = masks_count

        images_path = dataset_location + "images/"
        gt_path = dataset_location + "gt/"

        for patient in os.listdir(images_path):
            patient_images_path = images_path + f"{patient}/"
            patient_gt_path = gt_path + f"{patient}/"
            for image in os.listdir(patient_images_path):
                entry_name = image[:-4]
                self.images.append(read_image(patient_images_path + image).type(torch.float) / 255)
                masks = []
                for gt in os.listdir(patient_gt_path):
                    if gt.startswith(entry_name):
                        masks.append(read_image(patient_gt_path + gt).type(torch.float) / 255)
                self.labels.append(torch.stack(masks))

        assert (len(self.images) == len(self.labels))
        assert torch.max(torch.stack(self.images)) <= 1 and torch.min(torch.stack(self.images)) >= 0
        assert torch.max(torch.stack(self.labels)) <= 1 and torch.min(torch.stack(self.labels)) >= 0

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        # Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0, self.masks_count)].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
