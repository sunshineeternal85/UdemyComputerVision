
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
                             f'loss {running_train_loss_epoch} |'
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


def inference(model: nn.Module,
              test_loader: DataLoader,
              criterion: nn.Module, # Loss function, useful for calculating test loss
              device : torch.device
             ) -> Dict[str, Any]: # Return type changed to indicate Dict of string to Any
    
    # Initialize accumulators for metrics *before* the loop
    correct_test_samples = 0
    total_test_samples = 0
    running_test_loss = 0.0

    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
    
    with torch.no_grad(): # Disable gradient computation for efficiency
        # Corrected: Loop through the provided 'test_loader' not 'data_loader'
        for x_test, y_test in test_loader: 
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            
            z_test = model(x_test) # Forward pass: get predictions (logits)
            
            # Calculate test loss for the current batch
            test_loss_batch = criterion(z_test, y_test) 

            # Calculate accuracy metrics for the current batch
            _, y_pred = torch.max(z_test, 1) # Get predicted class labels
            correct_in_test_batch = (y_pred == y_test).sum().item()
            total_in_test_batch = len(y_test) # Number of samples in current batch

            # Accumulate metrics across all batches
            correct_test_samples += correct_in_test_batch
            total_test_samples += total_in_test_batch
            running_test_loss += test_loss_batch.item() # Add the scalar loss for the batch

    # Calculate overall test metrics after processing all batches
    # Check if total_test_samples is zero to prevent division by zero
    overall_test_loss = running_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    overall_test_accuracy = correct_test_samples / total_test_samples if total_test_samples > 0 else 0.0
    
    # Store results in a dictionary
    test_result = {
        'test_loss': overall_test_loss,
        'test_accuracy': overall_test_accuracy,
        'correct_predictions': correct_test_samples, # Optionally include raw counts
        'total_samples': total_test_samples # Optionally include raw counts
    }
            
    return test_result     




if __name__ == "__main__":

    logging.debug('Test message4')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('cuda available')
    else:
        device = torch.device('cpu')
        logging.info('cuda not available')
                 



