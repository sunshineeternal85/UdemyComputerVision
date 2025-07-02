# %%
import sys
import multiprocessing
import torch
import logging
import psutil

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s: %(lineno)d - %(funcName)s - %(message)s')

#%%


def get_recommended_num_workers(
    manual_num_workers: int = None,
    reserved_cpu_cores: int = 4,
    reserved_ram_gb: int = 6,
    estimated_ram_per_worker_gb: float = 1.0
) -> int:
    
    total_cpu_cores = multiprocessing.cpu_count()
    total_available_ram_gb = psutil.virtual_memory().available // (1024**3)

    logging.info(f'{total_cpu_cores} logical CPU cores available.')
    logging.info(f'{total_available_ram_gb} GB RAM available on system (including cache).')

    cpu_based_workers = max(0, total_cpu_cores - reserved_cpu_cores)

    if cpu_based_workers == 0:
        logging.warning("Very few CPU cores available. Consider num_workers=0 to avoid overhead.")
        cpu_based_workers = 0

    num_workers = manual_num_workers
    if num_workers is None:
        num_workers = cpu_based_workers
    else:
        num_workers = min(manual_num_workers, cpu_based_workers)

    if num_workers > 0 and estimated_ram_per_worker_gb > 0:
        ram_needed_for_workers = num_workers * estimated_ram_per_worker_gb
        remaining_ram_for_model_and_system = total_available_ram_gb - ram_needed_for_workers

        if remaining_ram_for_model_and_system < reserved_ram_gb:
            logging.warning(
                f"Calculated num_workers ({num_workers}) might lead to RAM starvation. "
                f"Only {remaining_ram_for_model_and_system:.2f} GB left, but {reserved_ram_gb} GB reserved."
            )
            max_workers_by_ram = max(0, int((total_available_ram_gb - reserved_ram_gb) / estimated_ram_per_worker_gb))
            if num_workers > max_workers_by_ram:
                num_workers = max_workers_by_ram
                logging.info(f"Adjusted num_workers down to {num_workers} based on RAM constraints.")

    if num_workers < 1:
        num_workers = 1 if total_cpu_cores > reserved_cpu_cores else 0
        if num_workers == 0:
             logging.info("Calculated num_workers is 0. Setting to 0 as very few cores are available.")
        else:
            logging.info("Calculated num_workers adjusted to 1 to ensure at least one worker if enough cores.")

    logging.info(f'Final num_workers set for AI training: {num_workers}')
    logging.info("For batch size, focus on your GPU/TPU VRAM first. If you hit OOM errors, reduce batch size (number of samples).")

    return num_workers 


def set_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'device set to {device}')
    return device
    
    
# %%
if __name__ == '__main__':
    num_workers = get_recommended_num_workers()
    device = set_device()



# %%
