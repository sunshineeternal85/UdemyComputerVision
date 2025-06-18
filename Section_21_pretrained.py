# %%

import os, logging, importlib, sys, shutil

import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import MobileNet_V3_Small_Weights
import torch
from torch.utils.data import DataLoader
from torchsummary import  summary
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

import json
import os
import requests 

logging.basicConfig(level=logging.INFO, format =  '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

logging.info(f'initiate librairies')


def transformer():
    transformer = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transformer

def denormalize(tensor: torch.Tensor, mean=(0.5,), std=(0.5,)):
    """
    Denormalizes a tensor image.
    Assumes the tensor is in [C, H, W] format and values are in [-1.0, 1.0]
    after having been normalized with the given mean and std.
    """
    logging.debug(f'min: {tensor.min().item()} max: {tensor.max().item()}')


    mean_tensor = torch.tensor(mean, device=tensor.device).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(len(std), 1, 1)

    logging.debug(f'mean_tesor shape {mean_tensor.shape}')
    logging.debug(f'std_tesor shape {std_tensor.shape}')

    denormalized_tensor = (tensor * std_tensor) + mean_tensor

    denormalized_tensor = denormalized_tensor * 255

    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 255)

    return denormalized_tensor.to(torch.uint8)

def imshow(image: torch.Tensor, ax= None ):
    image_tensor_cpu = image.detach().cpu()
    image_tensor_denorm = denormalize(image_tensor_cpu)

    pil_image = transforms.ToPILImage()(image_tensor_denorm)
    image_np = np.array(pil_image)

    if ax==None:

        plt.imshow(image_np)
    else:
        ax.imshow(image_np)
        ax.axis('off')


def imagenet_classes():
    #1. define dict container of the classes and index
    classes_dict = {}
    # 2. Define the URL for the ImageNet class labels file
    imagenet_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_filename = "imagenet_classes.txt"

    # 3. Download the labels file if it doesn't exist
    if not os.path.exists(labels_filename):
        print(f"Downloading ImageNet class labels to {labels_filename}...")
        try:
            response = requests.get(imagenet_labels_url)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            with open(labels_filename, "w") as f:
                f.write(response.text)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading labels: {e}")
            # Handle the error, perhaps exit or use a fallback
            exit() # Or raise an exception

    # 4. Read the class labels into a list
    with open(labels_filename, "r") as f:
        classes_dict = {index:line.strip().lower() for index, line in enumerate(f)}
    
    logging.info(f'{classes_dict}')
    return classes_dict





# %%
if __name__ == '__main__':
    if False:
        # prep the folder structure of the file 
        path = './images'

        if os.path.isdir(path):
            logging.debug(f'path found for {path}')
        else:
            logging.debug(f'path not found for {path}')


        for root, dirs, names in os.walk(path):
            for name in names:
                words_list = name.split('_')[1:]
                folder_name = '_'.join(words_list)[:-5].lower()
                try:
                    path_new_folder = os.path.join('./images',folder_name)
                    os.makedirs(path_new_folder,exist_ok=True)
                    logging.debug(f'{path_new_folder} folder created')
                except:
                    logging.debug(f'{path_new_folder} folder not created')
                
                try:
                    source_path= os.path.join('./images', name)
                    destination_path = os.path.join(path_new_folder,name)
                    shutil.move(src=source_path, dst=destination_path)
                    logging.info(f'{name} moved to {destination_path}')
                except:
                    logging.info(f'cannot move the file {name}')

    no_threads = max(1,os.cpu_count()-2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'{no_threads} cpu threads available, device set to {device}')
    
    model = models.mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.DEFAULT)

    model.to(device)

    logging.info(f'{model} loaded to {device}')

    dataset_path = './images'
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=transformer())
    
    logging.info(f'{len(test_dataset) } data in test dataset')

    if False:
        for index, (x,y) in enumerate(test_dataset):
            print(x,y)
            imshow(x)
            if index==1:
                break
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=transformer())

    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers= no_threads)

    
    logging.info(f'{test_dataset.classes} classes in dataloader test')

    classes_from_model = imagenet_classes()


    logging.info('start inference process from test dataset')
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(test_dataloader):
            if i == 15:
                x = x.to(device)
                y = y.to(device) 
                output = model(x)
                
                softmax_layer = torch.nn.Softmax(dim=1)
                output_prob = softmax_layer(output)

                y_pred = torch.max(output,1)[1].cpu().numpy()
                
                print(torch.topk(output_prob,k=3))
        
                
                y_true = y.cpu().numpy()
                
                images = x.detach()

                fig, axes = plt.subplots(2,2)
                ax_flatten = axes.flatten()


                for index, image in enumerate(images):
                    #print(image)
                    imshow(image, ax_flatten[index])


                    class_prediction = classes_from_model[y_pred[index]]
                    true_class = test_dataset.classes[y_true[index]].lower().replace('_', ' ')
                    logging.info(f'predicted : [{class_prediction}] vs truth : [{true_class}]')
                
                plt.tight_layout()
                plt.show()


            if i == 16:
                break



# %%
