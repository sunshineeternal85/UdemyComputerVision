#%%
# check nvidia gpu

import torch

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"--- GPU {i} ---")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")
        total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        print(f"Total Memory: {total_memory_gb:.2f} GB")
else:
    print("CUDA is NOT available.")
# %%
