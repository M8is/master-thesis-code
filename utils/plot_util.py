from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def plot(means: np.array, stds: np.array, meta: Dict[str, Any]) -> None:
    plot_label = __parse_label(meta)
    plt.plot(means, label=plot_label, linewidth=.5)
    if stds is not None:
        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=.3)


def legend() -> None:
    legend_ = plt.legend()
    plt.setp(legend_.get_lines(), linewidth=2)


def __parse_label(config: Dict[str, Any]) -> str:
    label = config['plot_label']
    for k, v in config.items():
        label = label.replace(f'${k}$', str(v))
    return label
