#%% 
import os, logging, sys
from typing import List, Callable, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset


from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import datetime

import torchsummary


# %%
def init_logs():
    log_dir = os.path.join(os.path.abspath(os.getcwd()),'logs','GAN')

    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir,'log_main.log')

    logging.basicConfig(level=logging.INFO, filename = log_filename ,  format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

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
        transforms.Resize(size=(28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
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
    mean= 0.5
    std = 0.5
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1,1,1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(-1,1,1)
    img = img.detach().cpu()
    img_denormalize = torch.clamp((img * std_tensor + mean_tensor)* 255, min=0, max= 255)

    return img_denormalize.to(torch.uint8)

def tensor_imshow(image: torch.Tensor,  label_idx: int, class_names: List[str],is_normalize:bool=False ,ax =None):
    image = image.detach().cpu()
    if is_normalize:
        image_denorm = denormalize(image)
    else:
        image_denorm = image
    
    image_np = image_denorm.to(torch.uint8).numpy().transpose(1,2,0)
    title_text = f'{class_names[label_idx]} - lab_ind: {label_idx}' 

    if ax==None:
        plt.imshow(image_np, cmap='gray')
        plt.axis('off')
        plt.title(title_text)
    else:
        ax.imshow(image_np, cmap= 'gray')
        ax.axis('off')
        ax.set_title(title_text)
    plt.tight_layout()

class Discriminator(nn.Module):
    def __init__(self ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=784,out_features=2048), #28
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=2048,out_features=1024), #28
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(p=0.25),
            
            nn.Linear(in_features=1024,out_features=512), #28
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(p=0.25),
            
            nn.Linear(in_features=512,out_features=256), #28
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=256,out_features=1), #28
            nn.Sigmoid()
        )

    def forward(self,x: torch.Tensor):
        # convert the image in 1d vector (28,28) = > 784 pix 
        x = x.view(x.size(0), 784)
        
        logits = self.model(x)
        return logits

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=100,out_features=256), #28
            nn.ReLU(),

            nn.Linear(in_features=256,out_features=512), #28
            nn.ReLU(),

            nn.Linear(in_features=512,out_features=1024), #28
            nn.ReLU(),

            nn.Linear(in_features=1024,out_features=784), # to get a 28 , 28 img size
            nn.Tanh(),
        )

    def forward(self, x): 
        logits = self.model(x)
        img = logits.view(x.size(0), 1, 28, 28) #  Reshapes 784 pixels to 1x28x28 image
        return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize Conv layers weights with a normal distribution mean 0, std 0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm weights with a normal distribution mean 1.0, std 0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # Initialize BatchNorm bias with zeros
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1: # Correct: Added initialization for Linear layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


#%%
if __name__ == '__main__':
    init_logs()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logging.info(f'using {device}')

    num_device = torch.cuda.device_count()

    logging.info(f'{device}-{num_device}')

    path = './SSD2/dataset/mnist'
    full_dataset = MNIST(root= path,train=True, transform=transformer_base(),download=True)

    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    if False: # option for displaying sample of image
        fig, ax = plt.subplots(3,3)
        ax_flatten = ax.flatten()


        i_dataset = np.random.choice(a=np.random.randint(0,59999,9),size=9,replace=False)
        
        for i in range(len(ax_flatten)):
            print(i_dataset[i])
            image, label_idx = full_dataset[i_dataset[i]]

            class_names = [str(i).split('-')[-1] for i in full_dataset.classes]


            logging.info(f'{label_idx}')

            tensor_imshow(image= image,  
                          label_idx= label_idx, 
                          class_names= class_names,
                          is_normalize=True ,
                          ax =ax_flatten[i])
    

    logging.info(f'init the models g and d')

    model_d = Discriminator().to(device=device)
    model_g = Generator().to(device=device)

    model_d.apply(weights_init)
    model_g.apply(weights_init)

    epochs = 20        # Number of training epochs
    batch_size = 32    # Batch size for training
    lr_g = 0.0002      # Learning rate for the Generator
    lr_d = 0.0002      # Learning rate for the Discriminator
    criterion = nn.BCELoss()  
    
    optim_d = optim.Adam(model_d.parameters(),lr=lr_d)
    optim_g = optim.Adam(model_g.parameters(),lr=lr_g)

    model_d_summary = torchsummary.summary(model_d, input_size=(1, 28, 28) , device='cuda')
    model_g_summary = torchsummary.summary(model_g, input_size=(100,), device='cuda')

    logging.info(f'model d:') # No newline at the end of the string here
    print(f'{model_d_summary}') # No newline at the end of the string here
    logging.info('') # This will create an empty log record, effectively a blank line

    logging.info(f'model g:') # No newline
    print(f'{model_g_summary}') # No newline
    logging.info('') # Another blank line
    

#%%
    if True: 
        for epoch in range(epochs):
            total_g_count_success = 0
            total_g_count = 0
            logging.info(f'Epoch: {epoch+1}/{epochs}')
            for batch_idx, (real_samples, _) in enumerate(train_loader):
                # generator process for initial data
                # not using the labels of the dataset as we overwrite by 1 as real
                #logging.info(f'Epoch: {epoch+1}/{epochs} - Batch: {n+1}/{len(train_loader)}')

                # get real data to device
                real_samples = real_samples.to(device=device)
                real_d_labels = torch.ones(batch_size,1).to(device=device)
                
                # Generate fake images, and labels to device
                latent_dim = 100 # Or whatever your chosen latent space dimension is
                latent_space_samples_d = torch.randn(batch_size, latent_dim).to(device=device)

                # Generate fake images using the generator for discriminator process
                fake_samples_d = model_g(latent_space_samples_d)
                fake_labels_d = torch.zeros(batch_size,1).to(device=device)
                
                # combine real and fake data
                all_samples = torch.cat((real_samples, fake_samples_d.detach()), dim=0)
                all_labels = torch.cat((real_d_labels, fake_labels_d), dim=0)
                
                ## init_discriminator ##
                # train discriminator
                model_d.train()
                optim_d.zero_grad()
                logits_d = model_d(all_samples)
                # calculate loss
                loss_d = criterion(logits_d, all_labels)
                loss_d_batch = loss_d.item()
                
                #logging.info(f'Loss Discriminator batch - {batch_idx}: {loss_d_batch:.4f}')
                # backpropagation
                loss_d.backward()
                optim_d.step()
                
                # data for generator
                latent_space_samples_g = torch.randn(batch_size, latent_dim).to(device=device)

                ## init_generator ##
                # train generator
                model_g.train()
                optim_g.zero_grad()
                fake_samples_g = model_g(latent_space_samples_g)
                fake_labels_g = torch.ones(batch_size, 1).to(device=device)

                logits_g = model_d(fake_samples_g)
                # calculate loss
                
                loss_g = criterion_d(logits_g, fake_labels_g) 

                loss_g_batch = loss_g.item()


                binary_pred_g_samples = (loss_g.detach() >= 0.5).float() # 1
                
                # Count how many of these were predicted as 'real' (which is the generator's goal)
                total_g_count_success += (binary_pred_g_samples == 1).sum().item()
                total_g_count += batch_size # Each batch adds `batch_size` samples to the total count

                #logging.info(f'Loss Generator batch - {batch_idx}: {loss_g_batch:.4f}')
                # backpropagation
                loss_g.backward()
                optim_g.step()

            if total_g_count > 0: # Avoid division by zero
                gen_success_accuracy = total_g_count_success / total_g_count
            else:
                gen_success_accuracy = 0.0

            loss_d_epoch = loss_d.item()
            loss_g_epoch = loss_g.item()
            logging.info(f'epoch {epoch}, Loss Discriminator: {loss_d_epoch:.4f} - Loss Generator: {loss_g_epoch:.4f}')
            logging.info(f'epoch {epoch}, Accuracy Success Generator: {gen_success_accuracy.item():.4f}')


    saved_model_path = './mnist_gan/'
    timestamp = datetime.datetime.now().isoformat()

    os.makedirs(saved_model_path, exist_ok=True)
    
    torch.save(
        {
            'dataset': 'mnist',
            'epoch': epoch,
            'model_d_state_dict': model_d.state_dict(),
            'optimizer_d_state_dict': optim_d.state_dict(),
            'model_g_state_dict': model_g.state_dict(),
            'optimizer_g_state_dict': optim_g.state_dict(),
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'accuracy_g': gen_success_accuracy.item(),
            'save_timestamp': timestamp, # Ensure 'timestamp' is defined (e.g., datetime.now().isoformat())
        },
        f"{saved_model_path}gan_checkpoint_epoch_{epoch+1}.pt" # <-- Missing filename here!
    )


 
