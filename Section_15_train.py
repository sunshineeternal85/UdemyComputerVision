# %%
import importlib

import logging, os
import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim 

from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST


import numpy as np
import matplotlib.pyplot as plt
import  custom_model
importlib.reload(custom_model)
from custom_model import Net, train_model




# %%

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(lineno)d - %(funcName)s - %(message)s'
)

def transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])
    logging.info(f'normalise the tensor image')
    return transform


def image_show(image: np.ndarray, ax=None):
    """Display an image on given axis or create new plot"""
    if ax is None:
        plt.imshow(image, cmap='gray')
    else:
        ax.imshow(image, cmap='gray')

# %%


if __name__ == '__main__':
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logging.info(f'set up the device as {device}')

    root = '/media/laurent/SSD2/dataset/mnist'
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

    os.makedirs(root, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path,exist_ok=True)

    train_dataset = MNIST(root= train_patsave_pathh, train= True,download= True, transform=transformer())
    test_dataset = MNIST(root= test_path, train= False,download= True, transform=transformer())

    total_no =  len(train_dataset) 
    train_no = int(round(len(train_dataset) * 0.90,0))
    validation_no =  total_no - train_no
    test_no = len(test_dataset)

    N_B_TRAIN = 256
    N_B_VAL = 128
    N_B_TEST = test_no    

    train_dataset , validation_dataset = random_split(train_dataset, [train_no, validation_no] )


    train_dataloader = DataLoader(train_dataset, batch_size= N_B_TRAIN, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=N_B_VAL, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=N_B_TEST, shuffle=False)

    logging.info(f'set up the data for train, validation and test')
    
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)


    print(images.shape)
    print(labels.shape)


    model = Net().to(device)

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
                                             epochs=3)

# %%
    if True:
        current_epoch_train_loss = history['train_loss'][-1]
        current_epoch_validation_loss = history['validation_loss'][-1]
        current_epoch_train_accuracy = history['train_accuracy'][-1]
        current_epoch_validation_accuracy = history['validation_accuracy'][-1]
        epoch = history['epoch_no'][-1]
    
        timestamp =  datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H-%M-%S')
        data = 'mnist'
        folder = 'mnist_model'
        save_path = os.path.join(os.path.dirname(__file__), folder)
        os.makedirs(save_path, exist_ok=True)

        # Create a meaningful filename
        filename = f"model_{data}_checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
        # The :03d ensures epoch number is zero-padded, e.g., 010 instead of 10

        CHECKPOINT_PATH = os.path.join(save_path, filename)


        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': current_epoch_train_loss,      # <-- Add train loss
            'validation_loss': current_epoch_validation_loss, # <-- Add validation loss
            'train_accuracy': current_epoch_train_accuracy,   # <-- Add train accuracy
            'validation_accuracy': current_epoch_validation_accuracy, # <-- Add validation accuracy
            'save_timestamp': timestamp,
        }, CHECKPOINT_PATH)

        logging.info(f'Model saved to: {CHECKPOINT_PATH}')


# %%
