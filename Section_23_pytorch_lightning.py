# %%

import os, logging
import imaplib
from typing import List, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms, datasets
import torch.optim as optim

import torchvision
from torchvision.transforms import transforms

import numpy as np

import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics 

from PIL import Image


logging.basicConfig(level=logging.INFO, format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

# %%

def denormalize(tensor: torch.Tensor, mean=(0.5,), std=(0.5,)):
    logging.debug(f'min: {tensor.min().item()} max: {tensor.max().item()}')
    mean_tensor = torch.tensor(mean, device=tensor.device).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(len(std), 1, 1)
    denormalized_tensor = (tensor * std_tensor) + mean_tensor
    denormalized_tensor = denormalized_tensor * 255
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 255)

    return denormalized_tensor.to(torch.uint8)

def transformer():
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        ),
    ])
    return transform

def pre_proc_data():
    path = '/media/laurent/SSD2/dataset/catsanddogs/PetImages'
    logging.info(f'{os.path.isdir(path)}')
    set_dataset = datasets.ImageFolder(path, transform=transformer())
    return set_dataset


def imshow(image: torch.Tensor, ax =None):
    image = image.detach().cpu()
    image_denorm = denormalize(image)
    image_np = np.array(transforms.ToPILImage()(image_denorm))

    if ax==None:
        plt.imshow(image_np)
        plt.axis('off')
    else:
        ax.imshow(image_np)
        ax.axis('off')


def show_sample( dataset, no_img:int = 1):
    if no_img==1:
        x,_ = dataset[0]

        print(type(x))
        imshow(x)
        plt.show()
    else:
        img_per_row = 3
        img_per_col = int(np.ceil(no_img/img_per_row))
        
        fig, ax =  plt.subplots(img_per_col,img_per_row)
        ax_flatten = ax.flatten()
        for i in range(no_img):
            x,_ = dataset[i]
            imshow(x, ax=ax_flatten[i])
        for j in range(no_img, len(ax_flatten)):
            ax_flatten[j].axis('off')

        plt.tight_layout()
        plt.show()


# %%


if __name__ == '__main__':
    train_dataset = pre_proc_data() 

    generator1 = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(train_dataset,
                                                            lengths=[20000,3000,2000],
                                                            generator=generator1)
    if True:
        show_sample(dataset=train_dataset,no_img=1)


# %%
