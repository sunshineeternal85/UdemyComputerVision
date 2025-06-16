
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict


# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s :%(lineno)d - %(funcName)s - %(message)s'
)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        logging.debug(f'Input shape: {x.shape}') # 28*28

        x = F.relu(self.conv1(x))
        logging.debug(f'after conv1 shape: {x.shape}') # 28-3+1 , 26*26
        
        x = F.relu(self.conv2(x))
        logging.debug(f'after conv2 shape: {x.shape}') # 26-3+1 , 24*24        
        
        x = self.pool(x)
        logging.debug(f'after maxpool shape: {x.shape}') # 24/2 , 12*12
        
        x = x.view(-1, 64*12*12) 
        logging.debug(f'After flatten: {x.shape}') # 9212

        x = F.relu(self.fc1(x))
        logging.debug(f'After fc1: {x.shape}') # 128
        
        x = self.fc2(x)
        logging.debug(f'After fc2: {x.shape}') # 10

        return x

class Net_bn(nn.Module):
    def __init__(self):
        super(Net_bn,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        logging.debug(f'Input shape: {x.shape}') # 28*28

        x = F.relu(self.conv1(x))
        x = self.bn1(x) 
        logging.debug(f'after conv1 shape: {x.shape}') # 28-3+1 , 26*26
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        logging.debug(f'after conv2 shape: {x.shape}') # 26-3+1 , 24*24

        x = self.pool(x)
        logging.debug(f'after maxpool shape: {x.shape}') # 24/2 , 12*12
        
        x = x.view(-1, 64*12*12) 
        logging.debug(f'After flatten: {x.shape}') # 9212

        x = F.relu(self.fc1(x))
        logging.debug(f'After fc1: {x.shape}') # 128
        
        x = self.fc2(x)
        logging.debug(f'After fc2: {x.shape}') # 10

        return x



class Net_bn_1(nn.Module):
    def __init__(self):
        super(Net_bn_1,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3), padding=1)  # Padding added to keep dimensions
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.25)  # Adding dropout layer

        self.conv2 = nn.Conv2d(32,64,(3,3), padding=1)  # Padding added to keep dimensions
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.25)  # Adding dropout layer
        
        self.conv3 = nn.Conv2d(64,128,(3,3), padding=1)  # Padding added to keep dimensions
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)  # Adding dropout layer
        
        self.conv4 = nn.Conv2d(128,256,(3,3), padding=1)  # Padding added to keep dimensions
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(0.25)  # Adding dropout layer
        
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256*14*14, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        logging.debug(f'Input shape: {x.shape}') # 28*28 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)  # Apply dropout after first conv layer 
        logging.debug(f'after conv1 shape: {x.shape}') # (28+2*1-3)/1 + 1 , 28*28
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)  # Apply dropout after second conv layer
        logging.debug(f'after conv2 shape: {x.shape}') # (28+2*1-3)/1 + 1 , 28*28

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)
        logging.debug(f'after conv3 shape: {x.shape}') # (28+2*1-3)/1 + 1 , 28*28

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop4(x)
        logging.debug(f'after conv4 shape: {x.shape}') # (28+2*1-3)/1 + 1 , 28*28

        x = self.pool(x)
        logging.debug(f'after maxpool shape: {x.shape}') # 28/2 , 14*14

        x = x.view(-1, 256*14*14)
        logging.debug(f'After flatten: {x.shape}') #  50176

        x = F.relu(self.fc1(x))
        logging.debug(f'After fc1: {x.shape}') # 128
        
        x = self.fc2(x)
        logging.debug(f'After fc2: {x.shape}') # 10

        return x



class NetAvg(nn.Module):
    def __init__(self):
        super(NetAvg,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.max_pool = nn.MaxPool2d(2,2)
        self.avg_pool = nn.AdaptiveAvgPool2d((10,10))
        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        logging.debug(f'Input shape: {x.shape}') # 28*28

        x = F.relu(self.conv1(x))
        logging.debug(f'after conv1 shape: {x.shape}') # 28-3+1 , 26*26
        
        x = F.relu(self.conv2(x))
        logging.debug(f'after conv2 shape: {x.shape}') # 26-3+1 , 24*24        
        
        x = self.max_pool(x)
        logging.debug(f'after maxpool shape: {x.shape}') # 24/2 , 12*12

        x = self.avg_pool(x)
        logging.debug(f'after avgpool shape: {x.shape}') # 24/2 , 10*10

        
        x = x.view(-1, 64*10*10) 
        logging.debug(f'After flatten: {x.shape}') # 9212

        x = F.relu(self.fc1(x))
        logging.debug(f'After fc1: {x.shape}') # 128
        
        x = self.fc2(x)
        logging.debug(f'After fc2: {x.shape}') # 10

        return x


class NetAvg_Bn(nn.Module):
    def __init__(self):
        super(NetAvg_Bn,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.max_pool = nn.MaxPool2d(2,2)
        self.avg_pool = nn.AdaptiveAvgPool2d((10,10))
        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        logging.debug(f'Input shape: {x.shape}') # 28*28

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        logging.debug(f'after conv1 shape: {x.shape}') # 28-3+1 , 26*26
        
        x = F.relu(self.conv2(x))
        logging.debug(f'after conv2 shape: {x.shape}') # 26-3+1 , 24*24

        x = self.max_pool(x)
        logging.debug(f'after maxpool shape: {x.shape}') # 24/2 , 12*12

        x = self.avg_pool(x)
        logging.debug(f'after avgpool shape: {x.shape}') # 24/2 , 10*10

        
        x = x.view(-1, 64*10*10) 
        logging.debug(f'After flatten: {x.shape}') # 9212

        x = F.relu(self.fc1(x))
        logging.debug(f'After fc1: {x.shape}') # 128
        
        x = self.fc2(x)
        logging.debug(f'After fc2: {x.shape}') # 10

        return x




def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,  # Added validation loader
                optimizer: optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device,    # Added device parameter
                epochs: int = 10,
                patience: int = 5  # Add early stopping patience                
                )-> Tuple[Dict,Dict]:

    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'epoch_no':[],
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }

    per_batch_metrics = {
        'epoch_no':[],
        'batch_no': [],
        'train_loss_batches': [],
        'train_accuracy_batches': []
    }

    try:
        for epoch in range(epochs):
            model.train() # Set model to training mode
            correct_train_samples_epoch = 0 # Accumulator for correct predictions in current epoch
            total_train_samples_epoch = 0   # Accumulator for total samples in current epoch
            running_train_loss_epoch = 0.0  # Accumulator for total loss in current epoch

            for train_b_i, (x_train, y_train) in enumerate(train_loader):
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                optimizer.zero_grad() # Zero gradients for this batch

                z = model(x_train) # Forward pass
                loss = criterion(z, y_train) # Calculate loss
                loss.backward() # Backward pass 
                optimizer.step() # Update model parameters

                # Calculate metrics for the CURRENT BATCH
                _, yhat = torch.max(z.detach(), 1) # Use .detach() to avoid unnecessary graph computations
                correct_in_this_batch = (yhat == y_train).sum().item()
                total_in_this_batch = len(y_train) # Actual number of samples in this batch

                # Update cumulative counts for epoch-level metrics
                correct_train_samples_epoch += correct_in_this_batch
                total_train_samples_epoch += total_in_this_batch
                running_train_loss_epoch += loss.item() # Accumulate scalar loss

                # Store PER-BATCH metrics
                # This is the accuracy of the CURRENT BATCH
                per_batch_acc_current_batch = correct_in_this_batch / total_in_this_batch

                per_batch_metrics['epoch_no'].append(epoch)
                per_batch_metrics['batch_no'].append(train_b_i)
                per_batch_metrics['train_accuracy_batches'].append(per_batch_acc_current_batch)
                per_batch_metrics['train_loss_batches'].append(loss.item()) # Current batch's scalar loss

                logging.debug(f'e: {epoch} |'
                             f'b: {train_b_i} |'
                             f'loss {running_train_loss_epoch/(train_b_i+1)} |'
                             f'acc:{per_batch_acc_current_batch}')


            # Calculate EPOCH-LEVEL aggregated metrics
            epoch_avg_train_loss = running_train_loss_epoch / len(train_loader)
            epoch_train_accuracy = correct_train_samples_epoch / total_train_samples_epoch

            # Store EPOCH-LEVEL metrics in history
            history['epoch_no'].append(epoch) # Corrected syntax
            history['train_loss'].append(epoch_avg_train_loss)
            history['train_accuracy'].append(epoch_train_accuracy)

            model.eval()  # Set model to evaluation mode
            correct_val_samples_epoch = 0
            total_val_samples_epoch = 0
            running_val_loss_epoch = 0.0

            with torch.no_grad():  # Disable gradient computation
                for val_b_i, (x_val, y_val) in enumerate(val_loader):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    # Forward pass only
                    z_val = model(x_val)
                    val_loss = criterion(z_val, y_val)

                    # Calculate validation metrics
                    _, yhat_val = torch.max(z_val, 1)
                    correct_in_val_batch = (yhat_val == y_val).sum().item()
                    total_in_val_batch = len(y_val)

                    # Update validation accumulators
                    correct_val_samples_epoch += correct_in_val_batch
                    total_val_samples_epoch += total_in_val_batch
                    running_val_loss_epoch += val_loss.item()

            # Calculate epoch-level validation metrics
            epoch_avg_val_loss = running_val_loss_epoch / len(val_loader)
            epoch_val_accuracy = correct_val_samples_epoch / total_val_samples_epoch

            # Store validation metrics in history
            history['validation_loss'].append(epoch_avg_val_loss)
            history['validation_accuracy'].append(epoch_val_accuracy)

            # Log progress
            logging.info(f'Epoch [{epoch+1}/{epochs}] '
                    f'Train Loss: {epoch_avg_train_loss:.4f} '
                    f'Train Acc: {epoch_train_accuracy:.4f} '
                    f'Val Loss: {epoch_avg_val_loss:.4f} '
                    f'Val Acc: {epoch_val_accuracy:.4f}')
            
            # Early stopping check
            if epoch_avg_val_loss < best_val_loss:
                best_val_loss = epoch_avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    except Exception as e:
        logging.error(f'Training failed with error: {str(e)}')
        raise

    return history, per_batch_metrics





#%%

if __name__ == "__main__":

    import torch
    import torch.nn as nn
    logging.debug('Test message4')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('cuda available')
    else:
        device = torch.device('cpu')
        logging.info('cuda not available')

    x = torch.tensor(torch.randint(1,10,(1,1,4,4)),dtype=float)
    x_samp = x[0,0,:2,0:2]
    print(x) 
    print('\n')
    print( x_samp)
    print(x_samp.mean().item())
    if False:
        pool = nn.MaxPool2d(kernel_size=(3,3), stride=3)
        x = pool(x)
        print(x)
    if True:
        pool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        x = pool(x)
        print(x)        



                 




# %%
