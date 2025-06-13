
# %%

import importlib
import os
import logging
import datetime
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, random_split, RandomSampler, ConcatDataset

from torchvision import transforms
from torchvision.datasets import FashionMNIST

from sklearn.metrics import classification_report, confusion_matrix

from model.custom_model import Net_bn, train_model

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s:%(lineno)d - %(funcName)s - %(message)s')

# %%


def transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
        ]
    )
    logging.info(f'return data in tensor and normalised')
    return transform


def transformer_flip():
    transform = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        ),
        transforms.RandomHorizontalFlip(p=1)
        ]
    )
    logging.info(f'return flip')
    return transform


def transformer_grey():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        ),
        transforms.RandomGrayscale(p=1)
        ]
    )
    logging.info(f'return flip')
    return transform




def denormalize(tensor, mean=(0.5,), std=(0.5,)):
    """
    Denormalizes a tensor image.
    Assumes the tensor is in [C, H, W] format and values are in [-1.0, 1.0]
    after having been normalized with the given mean and std.
    """
    mean_tensor = torch.tensor(mean).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std).view(len(std), 1, 1)

    denormalized_tensor = (tensor * std_tensor) + mean_tensor

    denormalized_tensor = denormalized_tensor * 255

    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 255)

    return denormalized_tensor




def imshow(image_tensor:torch.Tensor, ax=None):
    #image_tensor = denormalize(image_tensor)
    img_np = image_tensor.numpy()
    if not ax:

        plt.imshow(img_np[0],cmap='grey')
    plt.show()



# %%

if __name__ == '__main__':
    root = '/media/laurent/SSD2/dataset/fmnist'

    os.makedirs(root,exist_ok=True)

    train_dataset = FashionMNIST(root=root,train=True, download=True,transform= transformer())
    test_dataset = FashionMNIST(root=root,train=False, download=True, transform= transformer())

    train_dataset_trans1 = FashionMNIST(root=root,train=True, download=True,transform= transformer_flip())
    train_dataset_trans2 = FashionMNIST(root=root,train=True, download=True,transform= transformer_grey())

    train_dataset = ConcatDataset([train_dataset,
                                   RandomSampler(data_source=train_dataset_trans1, replacement=True,num_samples=6000),
                                   RandomSampler(data_source=train_dataset_trans2, replacement=True,num_samples=6000)
                                   ])


    print(len(train_dataset))
    logging.info(f'dl train data in {train_dataset}')
    logging.info(f'dl testt data in {test_dataset}')

    logging.info(f'x size: {train_dataset[0][0].size()}')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)




    for index, (x, y) in enumerate(train_dataset):
        logging.info(f'x size: {x.size()}\n')

        if index == 0:
            break



# %%
