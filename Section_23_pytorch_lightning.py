# %%

import os, logging
import imaplib
from typing import List, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.optim as optim

import torchvision
from torchvision.transforms import transforms

import numpy as np

import matplotlib.pyplot as plt
import lightning as L
import torchmetrics 

from PIL import Image


logging.basicConfig(level=logging.INFO, format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')


# %%

class SimpleCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.save_hyperparameters() # Good practice to save hyperparameters

        # Convolutional Layers
        # Input: (Batch_size, 1, 28, 28) for grayscale images like MNIST
        # If using color images (e.g., CIFAR10), input_channels would be 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Calculate the size of the flattened features after conv and pooling layers
        # We'll run a dummy tensor through the conv layers to determine this
        self._feature_extractor = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )
        # Dummy input to calculate the flattened size. Assuming 28x28 input for now.
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            flattened_size = self._feature_extractor(dummy_input).view(1, -1).size(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 128) # First FC layer
        self.fc2 = nn.Linear(128, num_classes)    # Second FC layer (output layer)

    def forward(self, x):
        # Apply convolutional layers and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten (Batch_size, Channels * Height * Width)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output layer (no activation here, as CrossEntropyLoss expects logits)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # Forward pass
        loss = F.cross_entropy(logits, y) # Calculate loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # Forward pass
        loss = F.cross_entropy(logits, y) # Calculate loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # You could also log accuracy here if you use torchmetrics
        # preds = torch.argmax(logits, dim=1)
        # accuracy = (preds == y).float().mean()
        # self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# --- Demonstration of how to use the CNNModel ---

if __name__ == '__main__':
    # Define hyperparameters for the dummy data and model
    num_samples = 1000
    image_height = 28
    image_width = 28
    num_channels = 1 # Grayscale
    num_classes = 10
    batch_size = 32
    max_epochs = 5

    print("Setting up dummy dataset...")
    # Create dummy data: random images and labels
    # Images: (num_samples, num_channels, height, width)
    # Labels: (num_samples,)
    dummy_images = torch.randn(num_samples, num_channels, image_height, image_width)
    dummy_labels = torch.randint(0, num_classes, (num_samples,))

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(dummy_images, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Use the same for validation for simplicity

    print(f"Initializing SimpleCNN model with {num_classes} classes...")
    model = SimpleCNN(num_classes=num_classes)

    print("Initializing Lightning Trainer...")
    # Initialize the Lightning Trainer
    # You can add arguments like 'accelerator', 'devices' for GPU/TPU training
    # For CPU-only, you can omit accelerator/devices or set accelerator="cpu"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto", # Automatically choose CPU, GPU, or TPU
        devices="auto",     # Automatically choose available devices
        log_every_n_steps=10 # Log training loss every 10 steps
    )

    print(f"Starting training for {max_epochs} epochs...")
    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\nTraining complete!")
    print("You can now access the trained model:", model)

    # Example of making a prediction after training
    print("\nMaking a dummy prediction:")
    model.eval() # Set model to evaluation mode
    dummy_input_for_prediction = torch.randn(1, num_channels, image_height, image_width)
    # Move input to the same device as the model
    dummy_input_for_prediction = dummy_input_for_prediction.to(model.device)

    with torch.no_grad():
        output_logits = model(dummy_input_for_prediction)
        predicted_class = torch.argmax(output_logits, dim=1).item()
    print(f"Predicted class for dummy input: {predicted_class}")
    
