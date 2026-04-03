import math


def get_lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_T: int,
    cosine_annealing_T: int,
) -> float:
    """
    Returns the learning rate at the given iteration according to a cosine schedule with linear warmup.
    Args:
        t: The current iteration.
        lr_max: The maximum learning rate.
        lr_min: The minimum learning rate.
        warmup_steps: The number of iterations to linearly increase the learning rate from 0 to lr_max.
        cosine_annealing_steps: The number of iterations to decrease the learning rate from lr_max to lr_min.
     Returns:
        The learning rate at the given iteration.
    """
    if t < warmup_T:
        return lr_max * t / warmup_T

    if t <= cosine_annealing_T:
        cosine_progress = (t - warmup_T) / (cosine_annealing_T - warmup_T)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * cosine_progress))

    return lr_min
