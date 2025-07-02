#%% 
import os, logging, sys
from typing import List, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset


from torchvision import transforms
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

class Discriminator(nn.Module):
    def __init__(self ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1,padding=1), #224
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # 112
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1), #112
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # 56
            nn.Dropout(p=0.25),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1), #224
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # 28
            nn.Dropout(p=0.25),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1), #224
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # 14
            nn.Dropout(p=0.25)
        )
        self.fc1 = nn.Linear(256 * 14 * 14,1)


    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        return logits



class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super().__init__()
        self.img_size = img_size 
        # 1. Project the latent vector to a small spatial feature map
        # Example: latent_dim -> 256 * 7 * 7 (if img_size / 32 is roughly 7)
        self.initial_linear = nn.Linear(latent_dim, 256 * (img_size // 16) * (img_size // 16)) # Example calculation

        self.model = nn.Sequential(
            # Start with a small feature map, then upscale
            # Example: 256 x 14 x 14 (assuming initial_linear leads to this after reshape)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), # -> 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), # -> 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), # -> 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=img_channels, kernel_size=4, stride=2, padding=1), # -> img_channels x 224 x 224
            nn.Tanh() # Output pixel values in [-1, 1] range, common for GANs
        )

    def forward(self, z): # z is the latent noise vector
        x = self.initial_linear(z)
        # Reshape to a 3D tensor suitable for ConvTranspose2d
        # The specific dimensions depend on your img_size and how many upsampling steps you want
        x = x.view(x.size(0), 256, (self.img_size // 16), (self.img_size // 16)) # Example
        img = self.model(x)
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

    train_dir = '../../../../media/laurent/SSD2/dataset/catsanddogs/PetImages/'

    full_dataset = Custom_Dataset(train_dir, transform=transformer_base())

    np.random.seed(42)
    full_dataset_aug = Custom_Dataset(train_dir, transform=transformer_aug())
    train_dataset_aug_ind = np.random.choice(range(len(full_dataset_aug)), size=2000, replace=False)
    train_dataset_aug = Subset(full_dataset_aug, indices= train_dataset_aug_ind)

    full_dataset = ConcatDataset(datasets=[full_dataset,train_dataset_aug])

    train_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    if False:
        fig, ax = plt.subplots(3,3)
        ax_flatten = ax.flatten()


        i_dataset = np.random.choice(a=np.random.randint(0,26999,9),size=9,replace=False)
        
        for i in range(len(ax_flatten)):
            print(i_dataset[i])
            image, label_idx = full_dataset[i_dataset[i]]

            class_names = ['cat', 'dog']


            logging.info(f'{label_idx}')

            tensor_imshow(image= image,  
                          label_idx= label_idx, 
                          class_names= class_names,
                          is_normalize=True ,
                          ax =ax_flatten[i])
    

    epochs = 50        # Number of training epochs
    batch_size = 64    # Batch size for training
    latent_dim = 100   # Dimensionality of the Generator's input noise vector
    img_size = 224     # Desired image size (height and width)
    img_channels = 3   # Number of image channels (3 for RGB, 1 for grayscale)
    lr_g = 0.0002      # Learning rate for the Generator
    lr_d = 0.0002      # Learning rate for the Discriminator
    beta1 = 0.5    

    netG = Generator(latent_dim, img_channels, img_size).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    logging.info("Generator and Discriminator models initialized and weights applied.")
    logging.info(f"Generator Architecture:\n{netG}")
    logging.info(f"Discriminator Architecture:\n{netD}")

    criterion = nn.BCEWithLogitsLoss() 
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, device=device) # 64 samples for a 8x8 grid

    # Convention for real and fake labels in BCE loss (soft labels or label smoothing can be used)
    real_label = 1.0 # Target label for real images
    fake_label = 0.0 # Target label for fake images

    img_list = [] # Stores generated image grids for visualization over epochs
    G_losses = [] # Stores Generator's loss per iteration
    D_losses = [] # Stores Discriminator's loss per iteration
    iters = 0     # Counter for total training iterations


    if True: 
        for epoch in range(epochs):
            for i, (real_images, _) in enumerate(train_dataloader):
                real_images = real_images.to(device)
                b_size = real_images.size(0) # Get current batch size

                netD.zero_grad() # Zero the gradients for the Discriminator
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_images).view(-1) # .view(-1) flattens the output to a 1D tensor
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, latent_dim, device=device)

                fake = netG(noise)

                label.fill_(fake_label)

                output = netD(fake.detach()).view(-1)

                errD_fake = criterion(output, label)

                errD_fake.backward()

                D_G_z1 = output.mean().item()


                errD = errD_real + errD_fake

                optimizerD.step()


                netG.zero_grad() # Zero the gradients for the Generator

                noise = torch.randn(b_size, latent_dim, device=device)
                fake = netG(noise)

                label.fill_(real_label)

                output = netD(fake).view(-1)

                errG = criterion(output, label)
                # Compute gradients for G
                errG.backward()
                # Get the average output of D for the Generator's samples (for monitoring)
                D_G_z2 = output.mean().item()
                # Update Generator's weights
                optimizerG.step()

                # --- Logging and Visualization ---
                # Print training stats every 50 batches
                if i % 50 == 0:
                    logging.info(f'[{epoch}/{epochs}][{i}/{len(train_dataloader)}] '
                                f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                                f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                # Save Generator's output using the fixed noise at specified intervals
                # This helps visualize how the generator improves over time
                if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(train_dataloader)-1)):
                    with torch.no_grad(): # Disable gradient calculations for inference
                        # Generate images using the fixed noise
                        fake_images_fixed = netG(fixed_noise).detach().cpu()
                    
                    # Create a directory for saving generated images if it doesn't exist
                    output_dir = os.path.join(os.path.abspath(os.getcwd()), 'generated_images')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save the generated images as a grid
                    vutils.save_image(fake_images_fixed,
                                    f"{output_dir}/fake_samples_epoch_{epoch:03d}_iter_{iters:05d}.png",
                                    normalize=True, nrow=8) # normalize to rescale pixel values for display
                    logging.info(f"Saved generated images to {output_dir}")
                    
                    # Store a grid of generated images for later plotting
                    img_list.append(vutils.make_grid(fake_images_fixed, padding=2, normalize=True))

                iters += 1 # Increment total iteration counter
                G_losses.append(errG.item()) # Record Generator loss
                D_losses.append(errD.item()) # Record Discriminator loss

    logging.info("GAN training complete!")

    # --- Plotting Training Losses ---
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # Save the loss plot
    loss_plot_path = os.path.join(os.path.abspath(os.getcwd()), 'gan_losses.png')
    plt.savefig(loss_plot_path)
    logging.info(f"Saved loss plot to {loss_plot_path}")
    plt.show()

    # --- Display Final Generated Images ---
    # Display the last set of generated images from the fixed noise
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images After Training (Last Batch from Fixed Noise)")
    # `np.transpose` is needed to change PyTorch's (C, H, W) to Matplotlib's (H, W, C)
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # Save the final generated image grid
    final_img_path = os.path.join(os.path.abspath(os.getcwd()), 'final_generated_images.png')
    plt.savefig(final_img_path)
    logging.info(f"Saved final generated images to {final_img_path}")
    plt.show() 
# %%
