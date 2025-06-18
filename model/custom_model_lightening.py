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

class LitModel(L.LightningModule):
    def __init__(self, batch_size, train_dataset = None, val_dataset = None, test_dataset = None):
        super(LitModel, self).__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.fc1=nn.Sequential(nn.Flatten(), nn.Linear(64*28*28,256), nn.ReLU(), nn.Linear(256,128))
        self.fc2= nn.Sequential(nn.Linear(128,2))

    def forward(self, x):
        # This is the forward pass where data flows through the layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x is now (batch_size, 64, 28, 28)
        x = self.fc1(x)
        # x is now (batch_size, 128)
        logits = self.fc2(x)
        # logits is now (batch_size, 2)
        return logits
    
    # --- Essential PyTorch Lightning Methods ---

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # Call the forward method
        loss = F.cross_entropy(logits, y) # Use CrossEntropyLoss for classification
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # You might also want to log accuracy
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True) # Log epoch-wise validation
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss # Return loss is good for early stopping callbacks

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    # These methods should RETURN DataLoader instances

    def train_dataloader(self):
        # Ensure self.train_dataset is not None before creating DataLoader
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided to LitModel for training.")
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

    def val_dataloader(self):
        if self.val_dataset is None:
            # You might choose to skip validation or raise an error
            print("Warning: val_dataset not provided. Skipping validation dataloader.")
            return None # Or raise ValueError("val_dataset must be provided...")
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=os.cpu_count() // 2)

    def test_dataloader(self):
        if self.test_dataset is None:
            print("Warning: test_dataset not provided. Skipping test dataloader.")
            return None # Or raise ValueError("test_dataset must be provided...")
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=os.cpu_count() // 2)



if __name__ == '__main__':
    L.LightningModule.test_step
