from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def plot(means: np.array, stds: np.array, label: str, meta: Dict[str, Any], xspace: np.array = None) -> None:
    if xspace is None:
        xspace = np.arange(len(means))
    assert len(xspace) == len(means), f"Length of xspace {len(xspace)} does not match number of values {len(means)}."
    plot_label = __parse_label(label, meta)
    plt.plot(xspace, means, label=plot_label, linewidth=.5)
    if stds is not None:
        plt.fill_between(xspace, means - stds, means + stds, alpha=.1)


def legend() -> None:
    legend_ = plt.legend()
    plt.setp(legend_.get_lines(), linewidth=2)


def __parse_label(label: str, config: Dict[str, Any]) -> str:
    for k, v in config.items():
        label = label.replace(f'${k}$', str(v))
    return label
