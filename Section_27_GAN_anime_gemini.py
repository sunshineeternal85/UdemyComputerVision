# %%
import os
import logging
import datetime
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import transforms
import torchvision.utils as vutils # Renamed for clarity in usage

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# For FID calculation (Optional, requires torch-fidelity)
# from torch_fidelity.metric_main import calculate_metrics

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
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, 'gan_gemini_training.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=log_filename,
        format='%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s'
    )
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

# --- 2. Custom Dataset Class ---
class Custom_Dataset(Dataset): # Inherit from Dataset, not just object
    def __init__(self, root_path: str, transform: Callable = None):
        self.root_path = root_path
        self.transform = transform
        self.filenames_list = self._list_path_file(self.root_path)
        logging.info(f"Found {len(self.filenames_list)} images in {root_path}")

    def _list_path_file(self, root_path):
        filenames_list = []
        valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

        for root, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith(valid_image_extensions):
                    filename_path = os.path.join(root, filename)
                    try:
                        # Attempt to open and verify the image
                        img = Image.open(filename_path)
                        img.verify()  # PIL throws if not valid
                        filenames_list.append(filename_path)
                    except Exception:
                        # Log a warning if it's an invalid image file
                        logging.warning(f'Invalid or corrupted image file: {filename_path}')
                        continue
        filenames_list.sort()
        return filenames_list

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx: int):
        imgpath = self.filenames_list[idx]
        img = Image.open(imgpath).convert('RGB')
        label = 0 # For GANs, real images are typically given a "real" label (e.g., 1 or 0.8)

        if self.transform:
            img = self.transform(img)

        return (img, label)

# --- 3. Image Transformers and Denormalization ---
def transformer_base(img_size=IMG_SIZE):
    # Standard DCGAN transform: Resize, ToTensor (0-1), Normalize (-1 to 1)
    transformer = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(), # Converts PIL Image to Tensor (0-1)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Converts 0-1 to -1 to 1
    ])
    return transformer

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    # Denormalize image from [-1, 1] to [0, 255]
    
    # First, move the image tensor to CPU
    img_denormalized = img_tensor.detach().cpu()
    
    # THEN create mean and std on CPU to match the image tensor
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=img_denormalized.device).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=img_denormalized.device).view(-1, 1, 1)
    
    # Apply denormalization formula: (img * std + mean) * 255
    img_denormalized = (img_denormalized * std + mean) # Scale to [0, 1]
    img_denormalized = torch.clamp(img_denormalized * 255, 0, 255) # Scale to [0, 255] and clamp
    
    return img_denormalized.to(torch.uint8)

def tensor_imshow(image: torch.Tensor, ax=None, title: str = "", is_normalize: bool = True):
    if is_normalize:
        image_display = denormalize(image)
    else:
        image_display = image.detach().cpu().to(torch.uint8) # Ensure uint8 and CPU

    image_np = image_display.numpy().transpose(1, 2, 0) # Convert (C, H, W) to (H, W, C)

    if ax is None:
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(title)
    else:
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(title)

# --- 4. DCGAN Models ---
class DiscriminatorDCGAN(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, img_size=IMG_SIZE):
        super().__init__()
        assert img_size == 64, "Discriminator is hardcoded for 64x64 input"

        self.main = nn.Sequential(
            # Input: (batch, img_channels, 64, 64) -> (64, 32, 32)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), # Added dropout for stability

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
            # No sigmoid here, BCEWithLogitsLoss handles it
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

            # (initial_channels_factor/8, 32,     logging.info(f'Initializing Generator and Discriminator models...')32) -> (img_channels, 64, 64)
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

def save_generated_images(model_g, epoch, device, fixed_noise, out_dir=SAMPLE_DIR):
    os.makedirs(out_dir, exist_ok=True)
    model_g.eval() # Set generator to evaluation mode
    with torch.no_grad():
        samples = model_g(fixed_noise)
    
    # Make a grid and save it
    grid = vutils.make_grid(samples, nrow=8, padding=2, normalize=True, value_range=(-1, 1))
    
    # Denormalize for matplotlib display if needed (vutils.make_grid normalizes already for saving)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generated Samples - Epoch {epoch}")
    # Convert from (C, H, W) to (H, W, C) for matplotlib
    plt.imshow(np.transpose(denormalize(grid).cpu().numpy(), (1, 2, 0)))
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"))
    plt.close()
    model_g.train() # Set generator back to train mode

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
# %%
# --- Main Training Logic ---
if __name__ == '__main__':
    init_logs()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    logging.info(f'Number of CUDA devices available: {torch.cuda.device_count()}')

    # --- Dataset and DataLoader ---
    # !! IMPORTANT !! Update this path to your actual anime dataset image directory
    anime_dataset_path = '../../../../media/laurent/SSD2/dataset/anime/archive/images'

    full_dataset = Custom_Dataset(root_path=anime_dataset_path, transform=transformer_base(IMG_SIZE))

    # Optional: Limit dataset size for faster experimentation
    np.random.seed(42)
    len_subset = 15000
    if len(full_dataset) > len_subset:
        img_subset_indices = np.random.choice(a=np.arange(0, len(full_dataset)), size=len_subset, replace=False)
        full_dataset = Subset(full_dataset, img_subset_indices)
        logging.info(f"Using a subset of {len(full_dataset)} images.")
    else:
        logging.info(f"Using full dataset of {len(full_dataset)} images.")

    train_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True # Speeds up data transfer to GPU
    )

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
    logging.info(f'Initializing Generator and Discriminator models...')
    model_d = DiscriminatorDCGAN(img_channels=IMG_CHANNELS, img_size=IMG_SIZE).to(device=device)
    model_g = GeneratorRGB(latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE).to(device=device)

    model_d.apply(weights_init)
    model_g.apply(weights_init)

    logging.info(f'--- Discriminator Architecture --- \n{model_d}')
    logging.info(f'--- Generator Architecture --- \n{model_g}')

    criterion = nn.BCEWithLogitsLoss() # Uses raw logits, applies sigmoid internally

    optim_d = optim.Adam(model_d.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    optim_g = optim.Adam(model_g.parameters(), lr=LR_G, betas=(BETA1, 0.999))

    # --- Load Checkpoint if Exists ---
    start_epoch = load_checkpoint(model_d, optim_d, model_g, optim_g, CHECKPOINT_DIR, device)

    # --- Fixed Noise for Visualization ---
    # Use a fixed noise vector to see progress of generation over epochs
    fixed_noise = torch.randn(64, LATENT_DIM, device=device) # Generate 64 samples for viz

    # --- Training Loop ---
    logging.info("Starting GAN training...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model_d.train()
        model_g.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        total_g_success_count = 0
        total_g_eval_count = 0

        for batch_idx, (real_samples, _) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            real_samples = real_samples.to(device=device)

            # --- Discriminator Training ---
            optim_d.zero_grad()

            # Train with all-real batch
            label_real = torch.full((current_batch_size, 1), 0.8, device=device) # Label smoothing
            output_real = model_d(real_samples)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()

            # Train with all-fake batch
            noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
            fake_samples = model_g(noise).detach() # Detach here is CRITICAL
            label_fake = torch.full((current_batch_size, 1), 0.2, device=device) # Label smoothing
            output_fake = model_d(fake_samples)
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optim_d.step()
            total_d_loss += errD.item()

            # --- Generator Training ---
            for _ in range(K_GEN_STEPS): # Train G multiple times if K_GEN_STEPS > 1
                optim_g.zero_grad()
                noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
                fake_samples = model_g(noise) # DO NOT DETACH for G training
                label_real_for_g = torch.full((current_batch_size, 1), 1.0, device=device) # G wants D to think fakes are real
                output_g = model_d(fake_samples)
                errG = criterion(output_g, label_real_for_g)
                errG.backward()
                optim_g.step()
                total_g_loss += errG.item() # Accumulate loss over generator steps

            # --- Metrics for logging ---
            with torch.no_grad():
                # Evaluate how well the *last* generated batch fooled D (for logging accuracy)
                last_fake_samples = model_g(torch.randn(current_batch_size, LATENT_DIM, device=device)) # Fresh batch for evaluation
                output_g_eval = model_d(last_fake_samples)
                binary_pred_g_samples = (torch.sigmoid(output_g_eval) >= 0.5).float()
                total_g_success_count += (binary_pred_g_samples == 1).sum().item()
                total_g_eval_count += current_batch_size

            if batch_idx % 50 == 0: # Log every 50 batches
                logging.info(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} "
                             f"| D_Loss: {errD.item():.4f} | G_Loss: {errG.item():.4f}")

        # --- End of Epoch Summary ---
        avg_d_loss_epoch = total_d_loss / len(train_loader)
        avg_g_loss_epoch = total_g_loss / (len(train_loader) * K_GEN_STEPS) # Divide by total G steps
        
        gen_success_accuracy = total_g_success_count / total_g_eval_count if total_g_eval_count > 0 else 0.0

        logging.info(f"--- Epoch {epoch} Complete ---")
        logging.info(f"Avg D Loss: {avg_d_loss_epoch:.4f} | Avg G Loss: {avg_g_loss_epoch:.4f}")
        logging.info(f"Generator Success Accuracy (D thinks real): {gen_success_accuracy:.4f}")

        # Save generated samples and checkpoint
        save_generated_images(model_g, epoch, device, fixed_noise)
        save_checkpoint(epoch, model_d, optim_d, model_g, optim_g,
                        avg_d_loss_epoch, avg_g_loss_epoch, gen_success_accuracy)

        # --- Optional: Calculate FID Score ---
        # This part is complex and requires FID setup/dataset preprocessing
        # It's usually done less frequently, e.g., every 10-25 epochs.
        # Ensure you have 'torch-fidelity' installed and an InceptionV3 model available.
        # if epoch % 10 == 0 and epoch > 0: # Calculate FID every 10 epochs
        #     logging.info(f"Calculating FID score for epoch {epoch}...")
        #     # You would need to generate a large number of images (e.g., 10k-50k)
        #     # and save them to a temporary directory.
        #     # Then, provide paths to real and generated images to calculate_metrics.
        #     # This part is highly dependent on your FID setup.
        #     # metrics = calculate_metrics(
        #     #     input1=r'path/to/real_images_folder',
        #     #     input2=r'path/to/generated_images_folder',
        #     #     cuda=True,
        #     #     isc=False, # Inception Score, usually not used with FID
        #     #     fid=True,
        #     #     verbose=False,
        #     # )
        #     # logging.info(f"FID Score at Epoch {epoch}: {metrics['fid']:.2f}")

    logging.info("GAN training complete.")

    # --- Final Visualization of Generated Samples ---
    logging.info("Displaying final generated sample grid...")
    fig, axes = plt.subplots(8, 8, figsize=(8, 8)) # 64 images in an 8x8 grid
    axes_flatten = axes.flatten()

    model_g.eval()
    with torch.no_grad():
        final_samples = model_g(fixed_noise).cpu() # Use the fixed noise to see consistent progress

    for i in range(fixed_noise.size(0)):
        tensor_imshow(image=final_samples[i], ax=axes_flatten[i], title='', is_normalize=True)
    plt.suptitle('Final Generated Samples', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()
    logging.info("Final samples displayed.")
# %%
