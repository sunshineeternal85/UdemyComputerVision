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

# --- Configuration Constants ---
# You can easily change these parameters here
IMG_SIZE = 64
IMG_CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 128 # Common batch size for GANs, adjust based on GPU memory
NUM_EPOCHS = 100 # Or more, as discussed for meaningful results
LR_G = 0.0002
LR_D = 0.0002
BETA1 = 0.5 # Adam optimizer beta1
K_GEN_STEPS = 1 # Number of generator steps per discriminator step (experiment with 1, 2, 3)
LOG_DIR = os.path.join(os.path.abspath(os.getcwd()), 'logs', 'anime_gan_gemini')
CHECKPOINT_DIR = os.path.join(os.path.abspath(os.getcwd()), 'anime_gan_gemini_checkpoints')
SAMPLE_DIR = os.path.join(os.path.abspath(os.getcwd()), 'generated_samples')
NUM_WORKERS = 4 # Number of data loading workers (adjust based on CPU cores)



# --- 1. Logging Setup ---
def init_logs():
    log_dir = os.path.join(os.path.abspath(os.getcwd()),'logs','GAN')

    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir,'log_main.log')

    logging.basicConfig(level=logging.INFO, filename = log_filename ,  format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

# --- 2. Custom Dataset Class ---
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
        label = 1 # For GANs, real images are typically given a "real" label (e.g., 1 or 0.8)

        if self.transform:
            img = self.transform(img) 

        return (img, label)


# --- 3. Image Transformers and Denormalization ---
class To_Tensor():
    def __call__(self, img_PIL: Image.Image):
        img_np = np.asarray(img_PIL)
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)

        img_np = img_np.transpose(2,0,1)
        img_tensor = torch.tensor(img_np, dtype= torch.float32)
        return img_tensor.contiguous()

def transformer_base(img_size=IMG_SIZE):
    transformer = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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

# --- 4. DCGAN Models ---
class DiscriminatorDCGAN(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, img_size=IMG_SIZE):
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
    def __init__(self, latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE):
        super().__init__()
        assert img_size == 64, "Generator is hardcoded for 64x64 output"

        # Calculate initial dimensions for ConvTranspose2d input
        # We need to map latent_dim vector to (initial_channels, init_res, init_res)
        self.init_res = img_size // (2**4) # 64 / 16 = 4
        self.initial_channels_factor = 512 # Number of channels at the start of transposed convolutions

        self.projection = nn.Sequential(
            nn.Linear(in_features=latent_dim,
                      out_features=self.initial_channels_factor * self.init_res * self.init_res,
                      bias=False), # Bias typically excluded if BatchNorm follows
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            # (initial_channels_factor, 4, 4) -> (initial_channels_factor/2, 8, 8)
            nn.ConvTranspose2d(self.initial_channels_factor, self.initial_channels_factor // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_channels_factor // 2),
            nn.ReLU(True),

            # (initial_channels_factor/2, 8, 8) -> (initial_channels_factor/4, 16, 16)
            nn.ConvTranspose2d(self.initial_channels_factor // 2, self.initial_channels_factor // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_channels_factor // 4),
            nn.ReLU(True),

            # (initial_channels_factor/4, 16, 16) -> (initial_channels_factor/8, 32, 32)
            nn.ConvTranspose2d(self.initial_channels_factor // 4, self.initial_channels_factor // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_channels_factor // 8),
            nn.ReLU(True),

            # (initial_channels_factor/8, 32, 32) -> (img_channels, 64, 64)
            nn.ConvTranspose2d(self.initial_channels_factor // 8, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Output pixel values in [-1, 1]
        )

    def forward(self, x):
        # Project latent vector to spatial feature map
        x = self.projection(x)
        # Reshape to (batch_size, initial_channels, init_res, init_res)
        x = x.view(x.size(0), self.initial_channels_factor, self.init_res, self.init_res)
        # Pass through transpose convolutions
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1: 
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# --- 4. orchestration functions ---

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

def save_checkpoint(epoch, model_d, optim_d, model_g, optim_g,
                    avg_loss_d, avg_loss_g, gen_success_accuracy,
                    out_dir=CHECKPOINT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filename = os.path.join(out_dir, f"gan_checkpoint_epoch_{epoch:04d}_{timestamp}.pt")
    
    torch.save(
        {
            'epoch': epoch,
            'model_d_state_dict': model_d.state_dict(),
            'optimizer_d_state_dict': optim_d.state_dict(),
            'model_g_state_dict': model_g.state_dict(),
            'optimizer_g_state_dict': optim_g.state_dict(),
            'avg_loss_d': avg_loss_d,
            'avg_loss_g': avg_loss_g,
            'accuracy_g': gen_success_accuracy,
            'save_timestamp': timestamp,
        },
        checkpoint_filename
    )
    logging.info(f"Checkpoint saved: {checkpoint_filename}")

def load_checkpoint(model_d, optim_d, model_g, optim_g, out_dir=CHECKPOINT_DIR, device='cpu'):
    if not os.path.isdir(out_dir):
        logging.info("No checkpoint directory found.")
        return 0 # Return epoch 0 if no checkpoint

    list_saved = [f for f in os.listdir(out_dir) if f.endswith('.pt')]
    if not list_saved:
        logging.info("No checkpoints found in the directory.")
        return 0

    list_saved.sort() # Sort to get the latest by filename (assuming timestamp in name)
    backup_file_name = list_saved[-1]
    full_checkpoint_path = os.path.join(out_dir, backup_file_name)

    try:
        checkpoint = torch.load(full_checkpoint_path, map_location=device)
        model_d.load_state_dict(checkpoint['model_d_state_dict'])
        optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        model_g.load_state_dict(checkpoint['model_g_state_dict'])
        optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        logging.info(f"Loaded checkpoint from: {full_checkpoint_path}, resuming from epoch {start_epoch}")
        return start_epoch
    except Exception as e:
        logging.error(f"Error loading checkpoint {full_checkpoint_path}: {e}")
        return 0 # Start from epoch 0 if loading fails



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

    # --- Display Sample Images (Optional) ---
    if True:
        logging.info("Displaying sample real images from the dataset...")
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        axes_flatten = axes.flatten()
        
        # Get a batch to sample from
        sample_batch, _ = next(iter(train_loader))
        
        for i in range(min(9, sample_batch.size(0))): # Display up to 9 images
            tensor_imshow(image=sample_batch[i], ax=axes_flatten[i], title=f'Real Image {i}', is_normalize=True)
        plt.tight_layout()
        plt.show()
        logging.info("Sample real images displayed.")  

    # --- Initialize Models, Optimizers, and Loss Function ---

    logging.info(f'init the models g and d')
    logging.info(f'Initializing Generator and Discriminator models...')

    model_d = DiscriminatorDCGAN().to(device=device)
    model_g = GeneratorRGB().to(device=device)
 

    model_d.apply(weights_init)
    model_g.apply(weights_init)


    epochs = 15        # Number of training epochs
    lr_g = 0.0002      # Learning rate for the Generator
    lr_d = 0.0002      # Learning rate for the Discriminator
    criterion = nn.BCEWithLogitsLoss() 
    
    optim_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optim_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(0.5, 0.999))


    if False:
        checkpoint_path = './anime_gan'

        if os.path.isdir(checkpoint_path):
            list_saved = os.listdir(checkpoint_path)
            list_saved.sort()
            backup_file_name = list_saved[-1] 
            full_checkpoint_path = os.path.join(checkpoint_path, backup_file_name)

            model_static = torch.load(full_checkpoint_path, map_location=device)
            model_static_d = model_static['model_d_state_dict']
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
                
                total_d_loss += loss_d.item()

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
                    total_g_loss += loss_g.item()


                binary_pred_g_samples = (F.sigmoid(logits_g.detach()) >= 0.5).float() # 1

                # Count how many of these were predicted as 'real' (which is the generator's goal)
                total_g_success_count += (binary_pred_g_samples == 1).sum().item()
                #total_g_count += current_batch_size # Each batch adds `current_batch_size` samples to the total count
                total_g_eval_count += binary_pred_g_samples.size(0)

            # End of epoch processing
            avg_d_loss_epoch = total_d_loss / len(train_loader)
            avg_g_loss_epoch = total_g_loss / (len(train_loader) * k_generator_steps) # Divide by total G steps
            
            gen_success_accuracy = total_g_success_count / total_g_eval_count if total_g_eval_count > 0 else 0.0

            logging.info(f'epoch {epoch}, Loss Discriminator: {avg_d_loss_epoch:.4f} - Loss Generator: {avg_g_loss_epoch:.4f}')
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
