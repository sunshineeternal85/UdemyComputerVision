# %%
import importlib

import logging, os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim 

from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import matplotlib.pyplot as plt

import model.custom_model as custom_model

importlib.reload(custom_model)
from model.custom_model import Net, train_model, NetAvg, NetAvg_Bn, Net_bn





# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(lineno)d - %(funcName)s - %(message)s'
)

def transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])
    logging.info(f'normalise the tensor image')
    return transform


def image_show(image: np.ndarray, ax=None):
    """Display an image on given axis or create new plot"""
    if ax is None:
        plt.imshow(image, cmap='gray')
    else:
        ax.imshow(image, cmap='gray')

# %%


if __name__ == '__main__':
    logging.info(f'Is cuda available? {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logging.info(f'set up the device as {device}')

    root = '/media/laurent/SSD2/dataset/mnist'
    
    test_path = os.path.join(root, 'test')

    os.makedirs(root, exist_ok=True)
    os.makedirs(test_path,exist_ok=True)

    
    test_dataset = MNIST(root= test_path, train= False,download= True, transform=transformer())

    N_B_TEST =  len(test_dataset) 
    if N_B_TEST > 128:
        N_B_TEST = 128

    test_dataloader = DataLoader(test_dataset, batch_size=N_B_TEST, shuffle=False)
    
    dataiter = iter(test_dataloader)
    images, labels = next(dataiter)

    CLASSES = test_dataset.classes

    logging.info(f'set up test data ! img batch shape : {images.shape} | label batch : { labels.shape}')
    logging.info(f'classes: {CLASSES}')


    PATH_TO_MODEL = './mnist_model_transformers/model_Std_mnist_checkpoint_epoch_016_2025-06-09_18-44-04.pth'
   
    model = Net().to(device)
    model_save = torch.load(PATH_TO_MODEL, map_location=torch.device('cpu'))
    train_accuracy = model_save['train_accuracy']

    model_param = model_save['model_state_dict']
    model.load_state_dict(model_param)
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total = 0
        total_incorrect = 0
        wrong_prediction =[]
        result = []
        for batch_i , (x_test, y_test) in enumerate(test_dataloader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            output = model(x_test)
            y_pred = torch.max(output,1)[1]

            total_correct += (y_pred == y_test).sum().item()
            total += len(y_test)
            total_incorrect += (y_pred != y_test).sum().item()
            accuracy = total_correct / total

            result.append([batch_i, y_test, y_pred])

            index_bool = (y_pred == y_test).tolist()
            for index, bool in enumerate(index_bool):
                position = batch_i * len(y_test) + index
                
                if not(bool):
                    wrong_prediction.append(
                        (position, 
                        y_pred[index].item(),
                        y_test[index].item(),
                        x_test[index].cpu().numpy())
                                            )
                    #print(wrong_prediction[-1][0:-1])
                    #image_show(x_test[index].numpy()[0])



            logging.debug(f'prediction accuracy {accuracy:.6f} for {total_correct} correct and {total_incorrect} wrong, out of  {total} tests')
            logging.debug(f'training accuracy : {train_accuracy:.6f}')

            #logging.info(f'batch no {batch_i} | \n output : {y_pred}')

            if batch_i == 20000:
                break
        logging.info(f'prediction accuracy {accuracy:.6f} for {total_correct} correct and {total_incorrect} wrong, out of  {total} tests')
        logging.info(f'training accuracy : {train_accuracy:.6f}')
    

        fig, ax = plt.subplots(3,4)
        ax_flatten = ax.flatten()
        fig.suptitle('Examples of Incorrect Predictions', fontsize=16, weight='bold')

        index_sample = np.random.randint(0, len(wrong_prediction), len(ax_flatten))
        for i in range(len(ax_flatten)):
            y_position = wrong_prediction[index_sample[i]][0]
            y_pred = wrong_prediction[index_sample[i]][1]
            y_true = wrong_prediction[index_sample[i]][2]
            image = wrong_prediction[index_sample[i]][3][0]
            logging.debug(f'{y_position}-{y_pred}-{y_true}')
            image_show(image, ax=ax_flatten[i])
            ax_flatten[i].set_title(f'True: {y_true}\nPred: {y_pred}', fontsize=10)
        ('Sample of wrong prediction')
        plt.tight_layout()
        plt.show() 


        y_pred_vals = []
        y_true_vals = []
        for index, i in enumerate(result):
            y_pred_i = i[2].cpu().numpy()  # i[2] contains the predictions
            y_true_i = i[1].cpu().numpy()  # i[1] contains the true labels
            y_pred_vals.extend(y_pred_i)   # use extend instead of append
            y_true_vals.extend(y_true_i)   # use extend instead of append

        y_pred = np.array(y_pred_vals)
        y_true = np.array(y_true_vals)
        print(confusion_matrix(y_true=y_true, y_pred=y_pred))

        print(classification_report(y_true=y_true, y_pred=y_pred))

# %%
