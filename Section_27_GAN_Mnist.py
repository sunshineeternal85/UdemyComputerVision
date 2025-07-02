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
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import requests
from PIL import Image, ImageFilter,ImageChops
from io import BytesIO


# %%
def init_logs():
    log_dir = os.path.join(os.path.abspath(os.getcwd()),'logs','GAN')

    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir,'log_main.log')

    logging.basicConfig(level=logging.INFO, filename = log_filename ,  format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')



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
        logging.info(folder_class)
        if 'dog' in str(folder_class).lower():
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
        img = logits.view(x.size(0), 1, 28, 28) # Example
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




#%%
if __name__ == '__main__':
    init_logs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_device = torch.cuda.device_count()

    logging.info(f'{device}-{num_device}')

    path = './SSD2/dataset/mnist'
    full_dataset = MNIST(root= path,train=True, transform=transformer_base(),download=True)

    np.random.seed(42)
    #full_dataset_aug = Custom_Dataset(train_dir, transform=transformer_aug())
    #train_dataset_aug_ind = np.random.choice(range(len(full_dataset_aug)), size=2000, replace=False)
    #train_dataset_aug = Subset(full_dataset_aug, indices= train_dataset_aug_ind)

    #full_dataset = ConcatDataset(datasets=[full_dataset,train_dataset_aug])

    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# %%

    if False:
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
    
# %%
    init_discriminator = Discriminator().to(device=device)
    init_generator = Generator().to(device=device)

#%%

    epochs = 20        # Number of training epochs
    batch_size = 32    # Batch size for training
    lr_g = 0.0002      # Learning rate for the Generator
    lr_d = 0.0002      # Learning rate for the Discriminator
    criterion = nn.BCELoss()  
    optim_gen = torch.optim.Adam(init_generator.parameters(),lr=lr_g)
    optim_dis = torch.optim.Adam(init_discriminator.parameters(),lr=lr_d)


#%%
    if True: 
        for epoch in range(epochs):
            for i, (samples, labels) in enumerate(train_loader):
                samples = samples.to(device=device)
                labels = labels.to(device=device)
                latent_space_samples = torch.rand(size=(batch_size,100)).to(device=device)
 


# %%
