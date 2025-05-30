# %%
#mnist
import logging, os
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import MNIST

import numpy as np
import matplotlib.pyplot as plt

# %%

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__),'section_15.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s')



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


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.fc1(x)
        x = self.fc2(x)
        logging.info(f'load the NN model')
        return x

def train_model(model, train_loader, validation_loader, optimizer, criterion, epochs: int = 10)-> Tuple(Dict,Dict):
    result = {}
    log ={}

    for epoch in range(epochs):
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'validation_loss': [],
            'validation_accuracy': []
        }

        per_batch_metrics = {
            'train_loss_batches': [],
            'train_accuracy_batches': []
        }

        for train_b_i, (x_train,y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            model.train()
            optimizer.zero_grad()

            z = model(x_train)
            loss = criterion(z,y_train)
            loss.backward()
            optimizer.step()
            
            _, yhat = torch.max(z.detach(),1)
            correct_in_batch += (yhat==y_train).sum().item()
            total_in_batch = len(y_train)
            correct_train_samples += correct_in_batch
            total_train_samples += total_in_batch
            running_train_loss += loss.item() 

            # Calculate and store per-batch metrics (if desired)
            per_batch_acc = correct_in_batch / total_in_batch
            per_batch_loss = loss.item() # Use .item() for scalar loss

            current_epoch_batch_losses.append(per_batch_loss)
            current_epoch_batch_accuracies.append(per_batch_acc)


            logging.info(f'Epoch {epoch+1}/{epochs} - Batch {train_b_i+1}/{len(train_loader)} - '
                            f'Train Loss: {per_batch_loss:.4f} - Train Acc: {per_batch_acc:.4f}')

        











    return result, log


# %%


if __name__ == '__main__':
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logging.info(f'set up the device as {device}')

    root = '/media/laurent/SSD2/dataset/mnist'
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

    os.makedirs(root, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path,exist_ok=True)

    train_dataset = MNIST(root= train_path, train= True,download= True, transform=transformer())
    test_dataset = MNIST(root= test_path, train= False,download= True, transform=transformer())

    total_no =  len(train_dataset) 
    train_no = int(round(len(train_dataset) * 0.90,0))
    validation_no =  total_no - train_no
    test_no = len(test_dataset)

    N_B_TRAIN = 12
    N_B_VAL = 12
    N_B_TEST = test_no    

    train_dataset , validation_dataset = random_split(train_dataset, [train_no, validation_no] )


    train_dataloader = DataLoader(train_dataset, batch_size= N_B_TRAIN, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=N_B_VAL, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=N_B_TEST, shuffle=True)

    logging.info(f'set up the data for train, validation and test')
    


    # %%
    if True:
        print(train_dataset.classes)
        print(type(train_dataset))

        for i, data in enumerate(train_dataset):
            if i <2:
                label = train_dataset.classes[data[1]]
                image_object = data[0][0]
                image = image_object.numpy()
                image_shape = f'image size : {image.shape}'
                print(image_shape)
                image_show(image)
                
            else:
                break
    
    # %%
    if True:
        
        fig, ax = plt.subplots(3,4)
        ax_flat = ax.flatten()

        # Get first batch of images
        for index, (images, labels) in enumerate(validation_dataloader):
            if index <1:
                for i in range(12):
                    image = images[i][0].numpy()

                    label = str(labels[i].numpy())
                    image_show(image, ax_flat[i])
                plt.show()

                

            else:
                break

    # %%
    if True:
        if torch.cuda.is_available():
            device = torch.device('gpu')
        else:
            device = torch.device('cpu')

        net = Net()
        net.to(device)
        print(net)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lf = 0.001, momentum=0.9)

        import torch
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.cuda.is_available())
        






# %%
