# %%
import gradio as gr


# %%

if __name__ == '__main__':
    # Initial setup for data path (used by run_inference)
    # The actual data loading will happen inside run_inference, but root dir can be defined once.
    logging.info(f'Is cuda available? {torch.cuda.is_available()}')
    logging.info(f'set up the device as {"cuda" if torch.cuda.is_available() else "cpu"}')
    
    # Define paths to your model checkpoints
    # IMPORTANT: Adjust these paths to where your .pth files are located!
    DEFAULT_MODEL_DIR = './mnist_model/' 
    # Example paths (make sure these files exist or remove them if they don't)
    MODEL_PATHS = {
        'model_mnist_checkpoint_epoch_004_2025-06-02_17-33-14.pth': os.path.join(DEFAULT_MODEL_DIR, 'model_mnist_checkpoint_epoch_004_2025-06-02_17-33-14.pth'),
        'model_Avg_mnist_checkpoint_epoch_004_2025-06-03_16-17-39.pth': os.path.join(DEFAULT_MODEL_DIR, 'model_Avg_mnist_checkpoint_epoch_004_2025-06-03_16-17-39.pth'),
        # Add more model paths here if you have them
    }

    # Populate dropdown choices for model paths
    model_path_choices = list(MODEL_PATHS.values())
    if not model_path_choices:
        logging.warning(f"No model paths found in '{DEFAULT_MODEL_DIR}'. Please ensure your .pth files are there.")
        # Add a dummy path if no files are found, or Gradio might error
        model_path_choices = ["./dummy_path_to_model.pth"]


    # Create Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("# MNIST Model Inference")
        gr.Markdown("Select a model type and load a saved model checkpoint to run inference on the MNIST test set.")

        with gr.Row():
            model_type_dropdown = gr.Dropdown(
                label="Select Model Type",
                choices=["Net", "NetAvg"],
                value="NetAvg", # Default selection
                interactive=True
            )
            model_path_dropdown = gr.Dropdown(
                label="Select Model Checkpoint (.pth)",
                choices=model_path_choices,
                value=model_path_choices[0] if model_path_choices else None, # Default selection
                interactive=True
            )
        
        run_button = gr.Button("Run Inference on Test Set")

        with gr.Column():
            accuracy_output = gr.Textbox(label="Inference Results", lines=3)
            error_image_output = gr.Plot(label="Incorrect Predictions")

        run_button.click(
            fn=run_inference,
            inputs=[model_type_dropdown, model_path_dropdown],
            outputs=[accuracy_output, error_image_output]
        )

    demo.launch()
