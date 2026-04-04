import torch
import torch.nn as nn
import os
import typing as T


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike[str] | T.IO[bytes] | T.BinaryIO) -> None:
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        iteration (int): The current training iteration, used for naming the checkpoint file.
        out (str | os.PathLike | T.IO[bytes] | T.BinaryIO): The output path or file-like object where the checkpoint will be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike[str] | T.IO[bytes] | T.BinaryIO, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        src (str | os.PathLike | T.IO[bytes] | T.BinaryIO): The source path or file-like object from which the checkpoint will be loaded.
        model (nn.Module): The model to which the loaded state will be applied.
        optimizer (torch.optim.Optimizer): The optimizer to which the loaded state will be applied.

    Returns:
        int: The training iteration at which the checkpoint was saved, extracted from the checkpoint file name.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']