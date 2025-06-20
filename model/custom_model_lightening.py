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

def conv_block(in_channels, out_channels):
    block = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                            nn.BatchNorm2d(num_features=out_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(2,2))
                            )
    return block




class LitModel1(L.LightningModule):
    def __init__(self, batch_size: int, num_class:int = 2, 
                 train_dataset:Dataset = None, val_dataset:Dataset = None, test_dataset:Dataset = None):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset', 'test_dataset'])

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.num_workers = max(1, os.cpu_count() //2)-2
        self.num_workers_val_test = max(1, os.cpu_count() // 4)-2


        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.fc1   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
            )
        self.fc2   = nn.Sequential(
            nn.Linear(128,num_class)
            )

        self.criterion = nn.CrossEntropyLoss() 


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits
        

    
    def train_dataloader(self):
        return DataLoader(
            dataset= self.train_dataset,
            batch_size= self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True 
            )
        
    def val_dataloader(self):
        return DataLoader(
            dataset= self.val_dataset,
            batch_size= self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers_val_test,
            pin_memory=True 
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset= self.test_dataset,
            batch_size= self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers_val_test,
            pin_memory=True 
            )
    
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        preds = torch.argmax(input=output, dim = 1)
        accuracy = (preds==label).float().mean()

        loss = self.criterion(input=output, target=label)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        preds = torch.argmax(input=output, dim = 1)
        accuracy = (preds==label).float().mean()

        loss = self.criterion(input=output, target=label)

        self.log('validation_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('validation_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        preds = torch.argmax(input=output, dim = 1)
        accuracy = (preds==label).float().mean()

        loss = self.criterion(input=output, target=label)

        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('test_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr =0.001)


if __name__ == '__main__':
    pass

# %%
