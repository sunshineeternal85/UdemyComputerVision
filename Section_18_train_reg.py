
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
import  model.custom_model as custom_model
importlib.reload(custom_model)
from model.custom_model import Net_bn_1, train_model




logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s:%(lineno)d - %(funcName)s - %(message)s')



def transformer(is_train: bool = True):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomRotation(degrees=(-15,15)),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5,p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5,p=0.25),
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            ),
            ] 
        )
        logging.info(f'return train data with data augmentation, then to tensor and normalised')
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,))
        ])
        logging.info(f'return test or the validation data to tensor and normalised')
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

    return denormalized_tensor.to(torch.uint8)




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

    no_threads = os.cpu_count() // 2
    
    no_threads = max(1, no_threads-1)
    logging.info(f'Number of threads set to: {no_threads}')

    num_gpus = torch.cuda.device_count()
    logging.info(f'{num_gpus} gpu available')
    
    for i in range(num_gpus):
        logging.info(f"\n--- GPU {i} ---")
        # Get the device object
        device = torch.device(f"cuda:{i}")
        logging.info(f"Device Name: {torch.cuda.get_device_name(i)}")

    # Get total memory (in GB)
    total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
    total_memory_gb = total_memory_bytes / (1024**3)
    logging.info(f"Total Memory: {total_memory_gb:.2f} GB")


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logging.info(f'using {device}')



# %%

    root = '/media/laurent/SSD2/dataset/fmnist'

    os.makedirs(root,exist_ok=True)

    train_transformer = transformer(is_train= True)
    val_transformer = transformer(is_train= False)
    test_transformer = transformer(is_train= False)


    full_train_dataset = FashionMNIST(root=root,train=True, download=True,transform= None)

    len_full_train_dataset = len(full_train_dataset)
    len_val_dataset = 4000
    len_train_dataset = len_full_train_dataset - len_val_dataset

    my_generator = torch.Generator().manual_seed(3)
    index_train_dataset, index_val_dataset = random_split(
        range(len_full_train_dataset), 
        lengths= [len_train_dataset,len_val_dataset],
        generator= my_generator
        )

    train_dataset = FashionMNIST(root=root,train=True, download=False,transform= train_transformer)
    train_dataset = Subset(train_dataset, index_train_dataset)

    val_dataset = FashionMNIST(root=root,train=True, download=False,transform= val_transformer)
    val_dataset = Subset(val_dataset, index_val_dataset)

    test_dataset = FashionMNIST(root=root,train=False, download=True, transform= test_transformer) 
    

    logging.info(f'train_dataset has {len(train_dataset)} items')
    logging.info(f'validation_dataset has {len(val_dataset)} items')
    logging.info(f'test_dataset has {len(test_dataset)} items')


    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=512, 
        shuffle=True, 
        num_workers=no_threads, 
        pin_memory=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=512, 
        shuffle=True, 
        num_workers=no_threads, 
        pin_memory=True)
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=no_threads, 
        pin_memory=True)

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



    model = Net_bn_1().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    logging.info(f'Using device: {device}')
    logging.info(f'loading model : {model}')

    history = {
    'epoch_no':[],
    'train_loss': [], 
    'train_accuracy': [],
    'validation_loss': [],
    'validation_accuracy': []
    }
    
    per_batch_metrics = {
        'train_loss': [],
        'train_accuracy': []
    }


    saved_model_path = './fmnist_model_transformers/model_StdBn_fmnist_checkpoint_epoch_005_2025-06-14_20-01-08.pth'
    
    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f'Model loaded from: {saved_model_path}')
    
    model = model.to(device)




    history, per_batch_metrics = train_model(model=model,
                                             train_loader=train_dataloader,
                                             val_loader=val_dataloader,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             device=device,
                                             epochs=20)
# %%
    if True:


        
        train_x_axis = list(range(len(history['train_loss'])))

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Loss on the first y-axis (ax1)
        color_loss = 'tab:red' # Choose a color for loss
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(train_x_axis, history['train_loss'], label='Train Loss', color=color_loss)
        ax1.plot(train_x_axis, history['validation_loss'], label='Validation Loss', color=color_loss, linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.grid(True) # Grid for the first y-axis

        # Create a second y-axis (ax2) that shares the same x-axis
        ax2 = ax1.twinx()
        color_accuracy = 'tab:blue' # Choose a different color for accuracy
        ax2.set_ylabel('Accuracy', color=color_accuracy)
        ax2.plot(train_x_axis, history['train_accuracy'], label='Train Accuracy', color=color_accuracy)
        ax2.plot(train_x_axis, history['validation_accuracy'], label='Validation Accuracy', color=color_accuracy, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_accuracy)

        # Title for the entire plot
        plt.title('Training and Validation Metrics over Epochs')

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center')

        plt.show()


# %%
    if True:
        current_epoch_train_loss = history['train_loss'][-1]
        current_epoch_validation_loss = history['validation_loss'][-1]
        current_epoch_train_accuracy = history['train_accuracy'][-1]
        current_epoch_validation_accuracy = history['validation_accuracy'][-1]
        epoch = history['epoch_no'][-1]
    
        timestamp =  datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H-%M-%S')
        data = 'fmnist'
        folder = 'fmnist_model_transformers'
        save_path = os.path.join(os.path.dirname(__file__), folder)
        os.makedirs(save_path, exist_ok=True)

        # Create a meaningful filename
        filename = f"model_StdBn_{data}_checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
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
    if True:
        transforms.ToPILImage()