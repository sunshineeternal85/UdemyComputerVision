# %%
import os
import logging
from typing import List, Tuple, Dict
import importlib
from pytorch_grad_cam import GradCAM


from torchvision.models import vgg16
import torch
import torchinfo

import numpy as np
import matplotlib.pyplot as plt


import model.custom_model as custom_model
importlib.reload(custom_model)
from model.custom_model import Net_bn_1, train_model

logging.basicConfig(level= logging.INFO, format = '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

# %%

def model_modifer(current_model: torch.nn.Module)-> torch.nn.Module:
    """
    Modify the model to use a different activation function.
    This is a placeholder function and should be implemented as needed.
    """
    # Example: Replace ReLU with LeakyReLU
    target_layer = current_model.features[0]  # Assuming the first layer is a Conv2d layer
    return current_model



# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vgg16(pretrained=True)
    model = model.to(device)
    model.eval()
    logging.info(f'VGG16 model loaded with pretrained weights')
    logging.info(f'{model.state_dict()}')

# %%

    path_custom_model = os.path.join(os.path.dirname(__file__), 'fmnist_model_transformers', 'model_StdBn_fmnist_checkpoint_epoch_005_2025-06-14_20-01-08.pth')
    model1 = Net_bn_1()
    model1 = model1.to(device)
    model1_static = torch.load(path_custom_model, map_location=device)
    model1.load_state_dict(model1_static['model_state_dict'])

    logging.info(f'Custom model loaded from {path_custom_model}')

# %%
    #print(torchinfo.summary(model=model))
    #torchinfo.summary(model=model)
    #print(model)
    #print("--- Model Architecture ---")
    #print(model1)
    #print("\n" + "="*40 + "\n")

    no_layers = 0

    


    for name, param in model1.named_parameters():
        name = str(name).split('.')[0]
        #print(name)
        if name in ['conv1','bn1','fc1','fc2']:
            #print(name)
            param.requires_grad = False
            #logging.info(f'{name} unfrozen')
        else:
            param.requires_grad = False
            #logging.info(f'{name} frozen')
    
    total_parameters = len(list(model1.parameters()))-1
    
    for i , param in enumerate(model1.parameters()):
        if i in [0,1,18, total_parameters-1, total_parameters ]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for i in model1.named_parameters():
        print(i)

# %%
    