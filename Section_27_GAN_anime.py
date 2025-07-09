#%% 
import os, logging
from typing import List, Callable


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset


from torchvision import transforms
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import datetime




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
        valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
                
        for root, _,  filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith(valid_image_extensions):
                    filename_path = os.path.join(root,filename)
                    try:
                        img = Image.open(filename_path)
                        img.verify()  # PIL throws if not valid
                        filenames_list.append(filename_path)
                    except Exception:
                        logging.warning(f'Invalid image file: {filename_path}') 
                        continue
        filenames_list.sort()
        return filenames_list

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx:int):
        imgpath = self.filenames_list[idx]
        img = Image.open(imgpath).convert('RGB')
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
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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
    title_text = f'{class_names[label_idx]}- : {label_idx}' 

    if ax==None:
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(title_text)
    else:
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(title_text)
    plt.tight_layout()

class DiscriminatorDCGAN(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super().__init__()
        # Assuming 64x64 input
        self.main = nn.Sequential(
            # Input: (batch, img_channels, 64, 64) -> (64, 32, 32)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (256, 8, 8) -> (512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (512, 4, 4) -> (1, 1, 1) - Final convolution to get a single output logit
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
            
            # No sigmoid here, use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

class GeneratorRGB(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super().__init__()
        
        self.init_res = img_size // (2**4) 
        self.initial_channels = 512 

        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, 
                      out_features=self.initial_channels * self.init_res * self.init_res),
            nn.ReLU(),

            nn.ConvTranspose2d(self.initial_channels, self.initial_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.initial_channels // 2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.initial_channels // 2, self.initial_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.initial_channels // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(self.initial_channels // 4, self.initial_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.initial_channels // 8),
            nn.ReLU(),

            nn.ConvTranspose2d(self.initial_channels // 8, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model[0](x) 
        x = x.view(x.size(0), self.initial_channels, self.init_res, self.init_res)
        x = self.model[1:](x) 
        return x

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

def save_generated_images(model_g, epoch, device, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    model_g.eval()
    with torch.no_grad():
        z = torch.randn(32, 100).to(device)
        samples = model_g(z)
    grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(8,4))
    plt.axis('off')
    plt.title(f'Generated Images at Epoch {epoch}')
    plt.imshow(np.transpose(grid.cpu().numpy(), (1,2,0)))
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png")
    plt.close()


#%%
if __name__ == '__main__':
    init_logs()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logging.info(f'using {device}')

    num_device = torch.cuda.device_count()

    logging.info(f'{device}-{num_device}')

    path = '../../../../media/laurent/SSD2/dataset/anime/archive/images'
    batch_size = 100

    full_dataset = Custom_Dataset(root_path=path, transform=transformer_base())
    np.random.seed(42)  # For reproducibility

    len_subset = 15000
    img_subset = np.random.choice(a=np.arange(0, len(full_dataset)), size=len_subset, replace=False) # Randomly select 15000 samples

    full_dataset = Subset(full_dataset, img_subset) # Limit to 15000 samples for faster training
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    if True: # option for displaying sample of image
        fig, ax = plt.subplots(3,3)
        ax_flatten = ax.flatten()

        i_dataset = np.random.choice(a=np.random.randint(0,len_subset-1,9),size=9,replace=False)
        
        for i in range(len(ax_flatten)):
            print(i_dataset[i])
            image, label_idx = full_dataset[i_dataset[i]]

            class_names = ['anime']


            logging.info(f'{label_idx}')

            tensor_imshow(image= image,  
                          label_idx= label_idx, 
                          class_names= class_names,
                          is_normalize=True ,
                          ax =ax_flatten[i])    


#%%    

    logging.info(f'init the models g and d')

    model_d = DiscriminatorDCGAN().to(device=device)
    model_g = GeneratorRGB().to(device=device)

    model_d.apply(weights_init)
    model_g.apply(weights_init)


    epochs = 150        # Number of training epochs
    lr_g = 0.0002      # Learning rate for the Generator
    lr_d = 0.0002      # Learning rate for the Discriminator
    criterion = nn.BCEWithLogitsLoss() 
    
    optim_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optim_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(0.5, 0.999))


    if True:
        checkpoint_path = './anime_gan'

        if os.path.isdir(checkpoint_path):
            list_saved = os.listdir(checkpoint_path)
            list_saved.sort()
            backup_file_name = list_saved[-1] 
            full_checkpoint_path = os.path.join(checkpoint_path, backup_file_name)

            model_static = torch.load(full_checkpoint_path, map_location=device)
            model_static_d = model_static['model_d_state_dict']
            model_static_g = model_static['model_g_state_dict']
            model_static_g = model_static['model_g_state_dict']
            optim_d.load_state_dict(model_static['optimizer_d_state_dict'])
            optim_g.load_state_dict(model_static['optimizer_g_state_dict'])

            model_d.load_state_dict(model_static_d)
            model_g.load_state_dict(model_static_g)

            model_d.to(device=device)
            model_g.to(device=device)
            logging.info(f'loading {backup_file_name}')



    logging.info(f'model d:') # No newline at the end of the string here
    print(f'{model_d}') # No newline at the end of the string here
    logging.info('') # This will create an empty log record, effectively a blank line

    logging.info(f'model g:') # No newline
    print(f'{model_g}') # No newline
    logging.info('') # Another blank line

#%%   
    if True: 
        for epoch in range(epochs):
            total_g_count_success = 0
            total_g_count = 0
            logging.info(f'Epoch: {epoch}/{epochs}')
            for batch_idx, (real_samples, _) in enumerate(train_loader):

                current_batch_size = real_samples.size(0) 

                # generator process for initial data
                # not using the labels of the dataset as we overwrite by 1 as real
                #logging.info(f'Epoch: {epoch+1}/{epochs} - Batch: {n+1}/{len(train_loader)}')


                # get real data to device
                real_samples = real_samples.to(device=device)
                #real_d_labels = torch.ones(batch_size,1).to(device=device)
                real_d_labels = torch.full((current_batch_size, 1), 0.8, device=device)

                # Generate fake images, and labels to device
                latent_dim = 100 # Or whatever your chosen latent space dimension is
                latent_space_samples_d = torch.randn(current_batch_size, latent_dim).to(device=device)
                
                # Generate fake images using the generator for discriminator process
                fake_samples_d = model_g(latent_space_samples_d)
                #fake_labels_d = torch.zeros(batch_size,1).to(device=device)
                fake_labels_d = torch.full((current_batch_size, 1), 0.2, device=device)
                
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
                
                k_generator_steps = 3 # Or 2, 3, etc. Experiment with this!
                for _ in range(k_generator_steps):
                # data for generator
                    latent_space_samples_g = torch.randn(current_batch_size, latent_dim).to(device=device)

                    ## init_generator ##
                    # train generator
                    model_g.train()
                    optim_g.zero_grad()
                    fake_samples_g = model_g(latent_space_samples_g)
                    #fake_labels_g = torch.ones(batch_size, 1).to(device=device)
                    fake_labels_g = torch.ones(current_batch_size, 1).to(device=device) 

                    logits_g = model_d(fake_samples_g)
                    logging.debug(f'logits_g_avg: {logits_g.mean().item():.4f}')  # Debugging info
                    # calculate loss
                    
                    loss_g = criterion(logits_g, fake_labels_g) 

                    loss_g_batch = loss_g.item()
                    #logging.info(f'Loss Generator batch - {batch_idx}: {loss_g_batch:.4f}')
                    # backpropagation
                    loss_g.backward()
                    optim_g.step()


                binary_pred_g_samples = (F.sigmoid(logits_g.detach()) >= 0.5).float() # 1

                # Count how many of these were predicted as 'real' (which is the generator's goal)
                total_g_count_success += (binary_pred_g_samples == 1).sum().item()
                #total_g_count += current_batch_size # Each batch adds `current_batch_size` samples to the total count
                total_g_count += binary_pred_g_samples.size(0)



            if total_g_count > 0: # Avoid division by zero
                gen_success_accuracy = total_g_count_success / total_g_count
                logging.info(f'{total_g_count_success} - {total_g_count}')
            else:
                gen_success_accuracy = 0.0

            loss_d_epoch = loss_d.item()
            loss_g_epoch = loss_g.item()

            logging.info(f'epoch {epoch}, Loss Discriminator: {loss_d_epoch:.4f} - Loss Generator: {loss_g_epoch:.4f}')
            logging.info(f'epoch {epoch}, Accuracy Success Generator: {gen_success_accuracy:.4f}')



    saved_model_path = './anime_gan/'
    timestamp = datetime.datetime.now().isoformat()

    os.makedirs(saved_model_path, exist_ok=True)
    
    torch.save(
        {
            'dataset': 'anime',
            'epoch': epoch,
            'model_d_state_dict': model_d.state_dict(),
            'optimizer_d_state_dict': optim_d.state_dict(),
            'model_g_state_dict': model_g.state_dict(),
            'optimizer_g_state_dict': optim_g.state_dict(),
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'accuracy_g': gen_success_accuracy,
            'save_timestamp': timestamp, # Ensure 'timestamp' is defined (e.g., datetime.now().isoformat())
        },
        f"{saved_model_path}gan_checkpoint_epoch_{timestamp}_epoch-{epoch}.pt" # <-- Missing filename here!
    )


 

# %%
    if True:
        # Generate latent space samples for these images
        display_num_samples = 4 
        latent_dim = 100 # Your latent space dimension
        display_latent_samples = torch.randn(display_num_samples, latent_dim).to(device)

        # Generate the images
        model_g.eval() # Set generator to evaluation mode
        with torch.no_grad(): # No need for gradients when generating for display
            generated_images = model_g(display_latent_samples) # Shape: (display_num_samples, 1, 28, 28)

        # Get the discriminator's classification for these generated images
        model_d.eval() # Set discriminator to evaluation mode
        with torch.no_grad(): # No need for gradients here either
            # Pass generated images to discriminator; detach them from G's graph for D's inference
            discriminator_outputs = model_d(generated_images.detach()) # Shape: (display_num_samples, 1)

        # Convert discriminator outputs (probabilities) to binary predictions (0 or 1)
        # 0 = Fail (discriminator thinks it's fake), 1 = Success (discriminator thinks it's real)
        binary_predictions = (discriminator_outputs >= 0.5).int().flatten().tolist()
        # Note: I'm calling it 'binary_predictions' here, not 'numpy_array_2d' as that variable was used for something else.

        # Set up the plot grid
        fig, ax = plt.subplots(2, 2) # Added figsize for better readability
        ax_flatten = ax.flatten()

        # Loop through and display each generated image with its predicted label
        for i in range(display_num_samples): # Iterate based on the number of samples you generated
            tensor_imshow(
                image=generated_images[i],          # Pass the single image tensor
                label_idx=binary_predictions[i],    # Pass the single integer prediction (0 or 1)
                class_names=['F', 'S'],    # Your class names for 0 and 1
                is_normalize=True,                  # Keep true if generated images are normalized
                ax=ax_flatten[i]                    # Pass the specific subplot axis
            )

        plt.suptitle('Generated Images and Discriminator Predictions') # Optional title for the whole figure
        plt.tight_layout()
        plt.show() # Display the plot




# %%
