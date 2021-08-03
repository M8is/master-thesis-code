from typing import Tuple

import numpy as np


def step_with_end(arr: np.array, step_size: int) -> Tuple[np.array, np.array]:
    total_size = arr.shape[-1]
    indices = np.append(np.arange(0, total_size, step_size), total_size - 1).astype(dtype='int')
    return indices, arr[..., indices]
