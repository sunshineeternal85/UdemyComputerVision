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
import io # For saving plots to BytesIO for Gradio

import model.custom_model as custom_model # Assuming this path is correct

importlib.reload(custom_model)
from model.custom_model import Net, train_model, NetAvg, NetAvg_Bn, Net_bn
import gradio as gr # Import Gradio

# Setup logging configuration (optional, as Gradio will output to console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(lineno)d - %(funcName)s - %(message)s'
)

# --- Global Setup (Model and Data Loading - runs once) ---
# This part runs when the script starts, before Gradio launches the interface

# Determine device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'set up the device as {DEVICE}')

# Define transform function
def get_transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])
    logging.info(f'Normalizing the tensor image')
    return transform

# Data paths
ROOT = '/media/laurent/SSD2/dataset/mnist'
TEST_PATH = os.path.join(ROOT, 'test')
os.makedirs(ROOT, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# Load test dataset and dataloader
TEST_DATASET = MNIST(root=TEST_PATH, train=False, download=True, transform=get_transformer())
N_B_TEST = len(TEST_DATASET)
if N_B_TEST > 128: # Limiting batch size to 128 as in your original code
    N_B_TEST = 128
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=N_B_TEST, shuffle=False)

# Get class names
CLASSES = TEST_DATASET.classes
logging.info(f'set up test data! classes: {CLASSES}')

# Model path
PATH_TO_MODEL = './mnist_model_transformers/model_StdBN_mnist_checkpoint_epoch_019_2025-06-09_19-10-05.pth'

# Load the model and its state dictionary once
try:
    MODEL = Net_bn()
    # map_location=DEVICE ensures it loads directly to the target device
    MODEL_SAVE_DATA = torch.load(PATH_TO_MODEL, map_location=DEVICE) 
    
    # Extract training accuracy from saved model data
    TRAIN_ACCURACY = MODEL_SAVE_DATA['train_accuracy']

    MODEL.load_state_dict(MODEL_SAVE_DATA['model_state_dict'])
    MODEL.to(DEVICE) # Ensure model is on the correct device
    MODEL.eval() # Set model to evaluation mode
    logging.info(f"Model loaded successfully from {PATH_TO_MODEL}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # Handle error, maybe exit or provide a dummy model
    MODEL = None 
    TRAIN_ACCURACY = 0.0

# --- Gradio Interface Function ---

def run_mnist_inference(dummy_input=None):
    if MODEL is None:
        return "Model not loaded due to an error.", f"Training Accuracy: {TRAIN_ACCURACY:.6f}", [], "Error", "Error"

    total_correct = 0
    total = 0
    total_incorrect = 0
    wrong_prediction_samples = [] # Store (image, true_label, pred_label) for Gradio Gallery
    
    y_pred_vals = []
    y_true_vals = []

    with torch.no_grad():
        for batch_i, (x_test, y_test) in enumerate(TEST_DATALOADER):
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            output = MODEL(x_test)
            y_pred = torch.max(output, 1)[1]

            total_correct += (y_pred == y_test).sum().item()
            total += len(y_test)
            total_incorrect += (y_pred != y_test).sum().item()
            
            y_pred_vals.extend(y_pred.cpu().numpy())
            y_true_vals.extend(y_test.cpu().numpy())

            # Capture wrong predictions for display
            incorrect_indices = (y_pred != y_test).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                img_np = x_test[idx].cpu().numpy()[0] # Get the image as numpy array (remove channel dim if 1)
                true_label = y_test[idx].item()
                pred_label = y_pred[idx].item()
                # Store as (image_data, caption) for Gradio Gallery
                wrong_prediction_samples.append((img_np, f"True: {true_label}, Pred: {pred_label}"))

            # Break early if enough wrong predictions collected for display
            if len(wrong_prediction_samples) >= 12: # Collect up to 12 wrong predictions
                break 

    current_accuracy = total_correct / total if total > 0 else 0.0

    # Prepare output messages
    prediction_accuracy_msg = f"Prediction Accuracy: {current_accuracy:.6f} for {total_correct} correct and {total_incorrect} wrong, out of {total} tests"
    training_accuracy_msg = f"Training Accuracy (from saved model): {TRAIN_ACCURACY:.6f}"

    # Generate Confusion Matrix and Classification Report
    conf_matrix_str = str(confusion_matrix(y_true=y_true_vals, y_pred=y_pred_vals))
    class_report_str = classification_report(y_true=y_true_vals, y_pred=y_pred_vals, target_names=CLASSES)

    # Return results for Gradio
    return (
        prediction_accuracy_msg,
        training_accuracy_msg,
        wrong_prediction_samples, # List of (image_np, caption)
        f"Confusion Matrix:\n{conf_matrix_str}",
        f"Classification Report:\n{class_report_str}"
    )

# --- Gradio Interface Definition ---

# Define the interface components
interface = gr.Interface(
    fn=run_mnist_inference, # The function to call
    inputs=gr.Button("Run MNIST Inference on Test Set"), # A button to trigger the inference
    outputs=[
        gr.Textbox(label="Overall Test Prediction Accuracy", lines=2),
        gr.Textbox(label="Training Accuracy (from Saved Model)", lines=1),
        gr.Gallery(label="Examples of Incorrect Predictions", columns=4, rows=3, object_fit="contain", height="auto"),
        gr.Textbox(label="Confusion Matrix", lines=12, max_lines=12), # Adjust lines as needed
        gr.Textbox(label="Classification Report", lines=15, max_lines=15) # Adjust lines as needed
    ],
    title="MNIST Model Inference with Net_bn",
    description="Click the button to run inference on the MNIST test set and view performance metrics and misclassified images."
)

# Launch the Gradio application
if __name__ == '__main__':
    interface.launch(share=False) # Set share=True to get a public link (temporarily)
# %%
