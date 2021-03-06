from functools import singledispatch
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None):
    """Find cost-minimizing permutation

    Parameters
    ----------
    y1 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_speakers)
    y2 : np.ndarray or torch.Tensor
        (num_samples, num_speakers) or (batch_size, num_samples, num_speakers)
    cost_func : callable
        Takes two (num_samples, num_speakers) sequences and returns (num_speakers, ) pairwise cost.
        Defaults to computing mean squared error.

    Returns
    -------
    permutated_y2 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_speakers)
    permutations : list of tuple
        List of permutations
    """
    raise TypeError()


def mse_cost_func(Y, y):
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor, y2: torch.Tensor, cost_func: Optional[Callable] = None
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_speakers = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_speakers)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_speakers_ = y2.shape
    if (
        batch_size != batch_size_
        or num_samples != num_samples_
        or num_speakers != num_speakers_
    ):
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    for y1_, y2_ in zip(y1, y2):
        with torch.no_grad():
            cost = torch.stack(
                [
                    cost_func(y2_, y1_[:, i : i + 1].expand(-1, num_speakers))
                    for i in range(num_speakers)
                ],
            )
        permutation = tuple(linear_sum_assignment(cost.cpu())[1])
        permutations.append(permutation)

        permutated_y2.append(y2_[:, permutation])

    return torch.stack(permutated_y2), permutations


@permutate.register
def permutate_numpy(
    y1: np.ndarray, y2: np.ndarray, cost_func: Optional[Callable] = None
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    permutated_y2, permutations = permutate(
        torch.from_numpy(y1), torch.from_numpy(y2), cost_func=cost_func
    )

    return permutated_y2.numpy(), permutations
