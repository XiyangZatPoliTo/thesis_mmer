from timeit import default_timer as timer
import torch


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """print difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds.")
    return total_time


def print_checkpoint_message(info: str):
    print(info)
