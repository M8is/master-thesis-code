import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


class LossHolder:
    def __init__(self, output_dir: str, train: bool):
        prefix = 'train' if train else 'test'
        self.__plot_file = os.path.join(output_dir, f'{prefix}_plot.svg')
        self.__file_path = os.path.join(output_dir, f'{prefix}_loss.pkl')

        if os.path.exists(self.__file_path):
            with open(self.__file_path, 'rb') as f:
                self.losses = pickle.load(f)
        else:
            self.losses = []

    def add(self, loss):
        self.losses.append(loss.cpu())

    def save(self):
        with open(self.__file_path, 'wb') as f:
            pickle.dump(self.losses, f)

    def numpy(self) -> np.array:
        with torch.no_grad():
            return torch.cat(self.losses).numpy()

    def plot(self, logscale=True):
        print(f"Plotting '{self.__plot_file}'.")
        if logscale:
            plt.yscale('log')
        plt.plot(self.numpy())
        plt.savefig(self.__plot_file)
        plt.clf()
