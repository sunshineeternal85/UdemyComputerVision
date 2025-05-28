# %%
#mnist

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import MNIST
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])
    return transform


def imshow(image: np.ndarray, label: str, image_shape:str):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f'class: {label} - {image_shape}')
    plt.show()



# %%


if __name__ == '__main__':
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    root = '/media/laurent/SSD2/dataset/mnist'
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

    os.makedirs(root, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path,exist_ok=True)

    train_dataset = MNIST(root= train_path, train= True,download= True, transform=transformer())
    test_dataset = MNIST(root= test_path, train= False,download= True, transform=transformer())

    print(train_dataset.classes)
    print(type(train_dataset))

    for i, data in enumerate(train_dataset):
        if i <2:
            label = train_dataset.classes[data[1]]
            image_object = data[0][0]
            image = image_object.numpy()
            image_shape = f'image size : {image.shape}'
            print(image_size)
            imshow(image, label, image_shape)
            
        else:
            break





# %%
