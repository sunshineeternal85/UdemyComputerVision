
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

from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset

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
    logging.info(f'return data with flip')
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
    logging.info(f'return data with grey')
    return transform




def denormalize(tensor: torch.Tensor, mean=(0.5,), std=(0.5,)):
    """
    Denormalizes a tensor image.
    Assumes the tensor is in [C, H, W] format and values are in [-1.0, 1.0]
    after having been normalized with the given mean and std.
    """
    logging.debug(f'min: {tensor.min().item()} max: {tensor.max().item()}')


    mean_tensor = torch.tensor(mean, device=tensor.device).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(len(std), 1, 1)

    logging.debug(f'mean_tesor shape {mean_tensor.shape}')
    logging.debug(f'std_tesor shape {std_tensor.shape}')

    denormalized_tensor = (tensor * std_tensor) + mean_tensor

    denormalized_tensor = denormalized_tensor * 255

    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 255)

    return denormalized_tensor




def imshow(image_tensor:torch.Tensor, ax=None):

    image_tensor_denorm = denormalize(image_tensor)
    img_np = image_tensor_denorm.cpu().squeeze().numpy()
    if ax is None:

        plt.imshow(img_np, cmap='gray')
        plt.show()

    else:
        ax.imshow(img_np, cmap='gray')
        



# %%

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    root = '/media/laurent/SSD2/dataset/fmnist'

    os.makedirs(root,exist_ok=True)

    train_dataset = FashionMNIST(root=root,train=True, download=True,transform= transformer())
    test_dataset = FashionMNIST(root=root,train=False, download=True, transform= transformer())

    np.random.seed(123)
    index_sample = np.random.choice(len(train_dataset), 3000, replace=True)

    train_dataset_trans1 = FashionMNIST(root=root,train=True, download=True,transform= transformer_flip())
    train_dataset_trans1 = Subset(train_dataset_trans1,index_sample)


    train_dataset_trans2 = FashionMNIST(root=root,train=True, download=True,transform= transformer_grey())
    train_dataset_trans2 = Subset(train_dataset_trans2,index_sample)

    train_dataset = ConcatDataset([train_dataset,
                                   train_dataset_trans1,
                                   train_dataset_trans2
                                   ])
    
    train_dataset, validation_dataset = random_split(train_dataset,lengths=[len(train_dataset)-4000,4000])

    logging.info(f'train_dataset has {len(train_dataset)} items')
    logging.info(f'validation_dataset has {len(validation_dataset)} items')
    logging.info(f'test_dataset has {len(test_dataset)} items')


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    if False:
        fig, ax = plt.subplots(3,3, figsize=(4,4))
    
        ax_flatten = ax.flatten()

        num_images_to_plot = min(len(train_dataset), len(ax_flatten))

    
        for i in range(num_images_to_plot):
            x, y = train_dataset[i]
            imshow(x, ax_flatten[i])
            ax_flatten[i].set_title(f'Label: {y}') # Add label as title
            ax_flatten[i].axis('off') 
        plt.tight_layout()
        plt.show()



# %%
    model = Net_bn().to(device)
    logging.info(f'loading model : {model}')

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = {
    'train_loss': [], 
    'train_accuracy': [],
    'validation_loss': [],
    'validation_accuracy': []
    }
    
    per_batch_metrics = {
        'train_loss': [],
        'train_accuracy': []
    }

    history, per_batch_metrics = train_model(model=model,
                                             train_loader=train_dataloader,
                                             val_loader=validation_dataloader,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             device=device,
                                             epochs=20)
# %%
