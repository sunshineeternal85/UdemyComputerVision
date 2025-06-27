# %%
import os, logging, sys
from typing import List, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset 


from torchvision import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from IPython.display import Image as Img


logging.basicConfig(level=logging.INFO,  format=  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

import requests
from PIL import Image, ImageFilter,ImageChops
from io import BytesIO

# %%
from PIL import Image
# Register a hook on the target layer (used to get the output channels of the layer)
class Hook():
    def __init__(self, module: nn.Module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()




# Make gradients calculations from the output channels of the target layer  
def get_gradients(net_in: torch.Tensor, net: nn.Module, layer: nn.Module):     
    net_in = net_in.unsqueeze(0).to(device)
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    return net_in.grad.data.squeeze()





# Denormalization image transform
denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
                              transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                              ])

# Run the Google Deep Dream.
def dream(image:np.array, net:nn.Module, layer:nn.Module, iterations: int, lr: float):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).to(device)
    for i in range(iterations):
        gradients = get_gradients(image_tensor, net, layer)
        image_tensor.data = image_tensor.data + lr * gradients.data

    img_out = image_tensor.detach().cpu()
    img_out = denorm(img_out)
    img_out_np = img_out.numpy().transpose(1,2,0)
    img_out_np = np.clip(img_out_np, 0, 1)
    img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
    return img_out_pil




# %%
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights= weights)
    model.to(device)

    model.eval()

    logging.info(f'model loaded')

except Exception as err:
    logging.info(f'error loading the model: {err}')

url = './image/download.jpeg'
dream_iterations = 20
dream_lr = 1

try:
    #response = requests.get(url=url)
    #img = Image.open(BytesIO(response.content))
    img = Image.open(url)
    logging.info(f"Image loaded successfully from local path: {url}")
    
    orig_size = np.array(img.size)
    new_size = np.array(img.size)*0.5
    img = img.resize(new_size.astype(int))
    layer = list( model.features.modules())[27]

    logging.info(f"DeepDream layer selected: {type(layer).__name__}")

    # Execute our Deep Dream Function 
    logging.info(f"Starting Deep Dream for {dream_iterations} iterations with LR {dream_lr}...")
    img = dream(img, model, layer, 20, 1)
    logging.info("Deep Dream process complete.")


    img = img.resize(orig_size, Image.LANCZOS) # Use a good resampling filter for resizing
    fig = plt.figure(figsize = (10 , 10))
    plt.imshow(img)
    plt.axis('off') # Turn off axes for cleaner image display
    plt.title("Deep Dream Output")
    plt.show() # Explicitly show the plot



except requests.exceptions.RequestException as e:
    logging.error(f"Error fetching image: {e}", exc_info=True)
except Image.UnidentifiedImageError as e:
    logging.error(f"Error: The downloaded content is not a valid image. Check the URL. {e}", exc_info=True)
except AttributeError as e:
    logging.error(f"AttributeError: Check module method names or object types. E.g., register_module_forward_hook is incorrect. {e}", exc_info=True)
except Exception as e:
    logging.error(f"An unexpected error occurred during Deep Dream process: {e}", exc_info=True)

# %%

# %%
