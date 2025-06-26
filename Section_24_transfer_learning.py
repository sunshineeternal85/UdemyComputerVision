#%%

import os, logging, sys
import importlib
from PIL import Image
from typing import List,Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset, Subset, ConcatDataset

from torch import optim

from torchvision import transforms, datasets 
from torchvision.datasets import mnist
from torchvision import models



import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt






# %%



class Custom_Dataset():
    def __init__(self, root_path: str, transform: Callable = None):
        self.root_path = root_path
        self.transform = transform
        self.filenames_list = self._list_path_file(self.root_path)


    def _list_path_file(self,root_path):
        filenames_list = []        
        for root,_,  filenames in os.walk(root_path):
            for filename in filenames:
                filename_path = os.path.join(root,filename)
                filenames_list.append(filename_path)

        filenames_list.sort()
        return filenames_list

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx:int):
        imgpath = self.filenames_list[idx]
        folder_class = os.path.basename(os.path.dirname(imgpath))
        img = Image.open(imgpath).convert('RGB')
        
        if 'dog' in folder_class:
            label = 1
        else:
            label = 0

        if self.transform:
            img = self.transform(img) 

        return (img, label)


class To_Tensor():
    def __call__(self, img_PIL: Image.Image):
        img_np = np.asarray(img_PIL)
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)

        img_np = img_np.transpose(2,0,1)
        img_tensor = torch.tensor(img_np, dtype= torch.float32)
        return img_tensor.contiguous()


def transformer_base():
    transformer = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer


def transformer_aug()-> Callable:
    transformer = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.RandomRotation(degrees=[-15,15]),
        transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.5),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transformer

def denormalize(img: torch.Tensor)-> torch.Tensor:
    mean= (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1,1,1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(-1,1,1)
    img = img.detach().cpu()

    img_denormalize = torch.clamp((img * std_tensor + mean_tensor)* 255,min=0,max= 255)

    return img_denormalize.to(torch.uint8)

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def tensor_imshow(image: torch.Tensor,  label_idx: int, class_names: List[str],is_normalize:bool=False ,ax =None):
    image = image.detach().cpu()
    if is_normalize:
        image_denorm = denormalize(image)
    else:
        image_denorm = image
    
    image_np = image_denorm.to(torch.uint8).numpy().transpose(1,2,0)
    title_text = f'{class_names[label_idx]} - label_index: {label_idx}' 

    if ax==None:
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(title_text)
    else:
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(title_text)


class LitModel_Fine_Tuning(L.LightningModule):
    def __init__(self, batch_size: int, num_class:int = 2, 
                 train_dataset:Dataset = None, val_dataset:Dataset = None, test_dataset:Dataset = None):
        
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset', 'test_dataset'])

        backbone = models.resnet50(pretrained= True)
        fc_in_num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        self.num_class = num_class
        
        self.fc2 = nn.Linear(fc_in_num_filters,num_class)

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.num_workers = max(1, os.cpu_count() //2)
        self.num_workers_val_test = max(1, os.cpu_count() // 4)


        self.criterion = nn.CrossEntropyLoss() 


    def forward(self,x):
        self.feature_extractor.eval()
        with torch.no_grad(): 
            x = self.feature_extractor(x)
        x = x.flatten(1)

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






# %%

if __name__ == '__main__': # FIX: Removed leading space
    train_dir = '../../../../media/laurent/SSD2/dataset/catsanddogs/PetImages/'

    if False:
        full_dataset = datasets.ImageFolder(root=train_dir, transform=transformer_aug())
        
        tensor_imshow(full_dataset[3][0],full_dataset[3][1],['Cat', 'Dog'], is_normalize=True)

    if True:
        full_dataset = Custom_Dataset(train_dir, transform=transformer_base())

        np.random.seed(42)
        full_dataset_aug = Custom_Dataset(train_dir, transform=transformer_aug())
        train_dataset_aug_ind = np.random.choice(range(len(full_dataset_aug)), size=2000, replace=False)
        train_dataset_aug = Subset(full_dataset_aug,indices= train_dataset_aug_ind)

        if False:
            tensor_imshow(full_dataset[0][0],full_dataset[0][1],['Cat', 'Dog'],is_normalize=True)
            tensor_imshow(train_dataset_aug[0][0],train_dataset_aug[0][1],['Cat', 'Dog'],is_normalize=True)

    if False:
        full_dataset = datasets.DatasetFolder(
            root=train_dir, 
            loader= pil_loader,
            extensions=('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'),
            transform=transformer_aug())
        
        tensor_imshow(full_dataset[3][0],full_dataset[3][1],['Cat', 'Dog'], is_normalize=True)

    len_val, len_test = 1500, 500
    len_train = len(full_dataset)- len_val - len_test 

    generator = torch.manual_seed(234)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,lengths = [len_train,len_val,len_test],
        generator=generator)
    
    train_dataset = ConcatDataset(datasets=[train_dataset, train_dataset_aug])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    


# %%
    logger_path = os.path.join(os.getcwd(), 'logs', 'dog_cat')

    logger = TensorBoardLogger(logger_path, name="LitModel_Fine_Tuning") # Instantiate logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path, "checkpoints"), # Directory where checkpoints will be saved
        filename="{epoch}-{validation_loss_epoch:.4f}-{validation_accuracy_epoch:.4f}", # Naming convention for saved files
        monitor="validation_loss_epoch",    # Metric to monitor for saving
        mode="min",            # 'min' means save when val_loss decreases
        save_top_k=1,          # Save only the best 1 model based on monitor
        verbose=True           # Log when a new best model is saved
    )



#%%
    model = LitModel_Fine_Tuning(batch_size=48,
                      num_class=2, # Explicitly pass num_class for clarity
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,    # Pass val_dataset
                      test_dataset=test_dataset) 
    
    trainer = L.Trainer(
        accelerator=device,       # Specify accelerator (cuda or cpu)
        devices=1,                # Specify number of devices (1 for single GPU/CPU)
        max_epochs=2,            # Specify how many epochs to train for
        logger=logger,            # Pass the logger to the trainer
        callbacks= [
            EarlyStopping(monitor="validation_loss_epoch", mode="min", patience=4),
            checkpoint_callback]
    )
# %%
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
# %%
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs 