# %%

import os, logging
import importlib
from typing import List, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms, datasets
import torch.optim as optim

import torchvision
from torchvision.transforms import transforms

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import lightning as L


import model.custom_model_lightening 
importlib.reload(model.custom_model_lightening )
from model.custom_model_lightening import LitModel1



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

def transformer()->transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=[-15,15]),
        transforms.RandomInvert(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])
    return transform

def pre_proc_data(path: str = '/media/laurent/SSD2/dataset/catsanddogs/PetImages') -> datasets.ImageFolder:
    logging.info(f'Loading dataset from: {path}')
    logging.info(f'Path exists: {os.path.isdir(path)}')

    full_dataset = datasets.ImageFolder(path, transform=None) # No transform yet
    valid_samples = []
    skipped_count = 0
    logging.info("Starting image validation (this might take a moment)...")
    for i, (image_path, label) in enumerate(full_dataset.samples):
        try:
            # Try to open the image with PIL
            img = Image.open(image_path)
            img.verify() # Verify file integrity (closes the file)
            # Reopen and convert mode if needed, just to be sure it's readable as RGB
            Image.open(image_path).convert('RGB')
            valid_samples.append((image_path, label))
        except (IOError, UnidentifiedImageError, OSError) as e:
            skipped_count += 1
            logging.warning(f"Skipping corrupted or unreadable image: {image_path} - Error: {e}")
        # Optional: Add a progress indicator for very large datasets
        if (i + 1) % 1000 == 0:
            logging.info(f"Processed {i + 1} images, skipped {skipped_count}")


    def is_valid_image(path_to_check: str) -> bool:
        try:
            Image.open(path_to_check).verify()
            Image.open(path_to_check).convert('RGB') # Ensure it's readable as RGB
            return True
        except (IOError, UnidentifiedImageError, OSError):
            logging.warning(f"Skipping corrupted image: {path_to_check}")
            return False

    final_dataset = datasets.ImageFolder(root=path, transform=transformer(), is_valid_file=is_valid_image)
    logging.info(f"Final dataset loaded with {len(final_dataset)} valid images.")

    return final_dataset


def imshow(image: torch.Tensor,  label_idx: int, class_names: List[str], ax =None):
    image = image.detach().cpu()
    image_denorm = denormalize(image)
    image_np = np.array(transforms.ToPILImage()(image_denorm))
    title_text = f'{class_names[label_idx]} - label_index: {label_idx}' 

    if ax==None:
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(title_text)
    else:
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(title_text)


def show_sample( dataset, class_names: List[str], no_img:int = 1):
    if no_img==1:
        x,y = dataset[0]

        print(type(x))
        imshow(image=x, label_idx=y, class_names=class_names)
        plt.show()
    else:
        img_per_row = 3
        img_per_col = int(np.ceil(no_img/img_per_row))
        
        fig, ax =  plt.subplots(img_per_col,img_per_row)
        ax_flatten = ax.flatten()

        for i in range(no_img):
            x, y = dataset[i]
            imshow(image=x, label_idx=y, class_names=class_names, ax=ax_flatten[i])

        for j in range(no_img, len(ax_flatten)):
            ax_flatten[j].axis('off')

        plt.tight_layout()
        plt.show()


# %%


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_dataset = pre_proc_data()
    class_names = full_dataset.classes 

    len_t_dataset = len(full_dataset) - 2000


    generator1 = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(full_dataset,
                                                            lengths=[len_t_dataset,1500,500],
                                                            generator=generator1)
    if False:
        show_sample(dataset=train_dataset, class_names=class_names, no_img=3)
#%%

    logger_path = os.path.join(os.getcwd(), 'logs', 'dog_cat')

    logger = TensorBoardLogger(logger_path, name="model_LitModel1") # Instantiate logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path, "checkpoints"), # Directory where checkpoints will be saved
        filename="{epoch}-{validation_loss_epoch:.4f}-{validation_accuracy_epoch:.4f}", # Naming convention for saved files
        monitor="validation_loss_epoch",    # Metric to monitor for saving
        mode="min",            # 'min' means save when val_loss decreases
        save_top_k=1,          # Save only the best 1 model based on monitor
        verbose=True           # Log when a new best model is saved
    )

#%%
    model = LitModel1(batch_size=48,
                      num_class=len(class_names), # Explicitly pass num_class for clarity
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,    # Pass val_dataset
                      test_dataset=test_dataset) 

    



#%%

    trainer = L.Trainer(
        accelerator=device,       # Specify accelerator (cuda or cpu)
        devices=1,                # Specify number of devices (1 for single GPU/CPU)
        max_epochs=15,            # Specify how many epochs to train for
        logger=logger,            # Pass the logger to the trainer
        callbacks= [
            EarlyStopping(monitor="validation_loss_epoch", mode="min", patience=4),
            checkpoint_callback]
    )


#%%


    logging.info("Starting model training...")
    trainer.fit(model)
    logging.info("Model training finished.")

    # 7. (Optional) Test the model after training
    logging.info("Starting model testing...")
    trainer.test(model) # You can pass test_dataloaders here if not in model
    logging.info("Model testing finished.")

    final_model_path = os.path.join(logger_path, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")

    torch.save(model.state_dict(), os.path.join(logger_path, "final_model_state_dict.pth"))


