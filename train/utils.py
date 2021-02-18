import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder:
    _datasets = {
        'mnist': datasets.MNIST
    }

    def __init__(self, dataset_tag, batch_size, shuffle=True):
        if dataset_tag.lower() not in self._datasets:
            raise ValueError(f'Invalid dataset `{dataset_tag}`. Allowed values: {list(self._datasets.keys())}.')
        dataset = self._datasets[dataset_tag.lower()]

        self.train_holder = DataLoader(
            dataset(root='./data', train=True, download=True, transform=transforms.ToTensor()), shuffle=shuffle,
            batch_size=batch_size)
        self.test_holder = DataLoader(
            dataset(root='./data', train=False, download=True, transform=transforms.ToTensor()), shuffle=shuffle,
            batch_size=batch_size)

        _, self.height, self.width = self.train_holder.dataset.data.shape


class LossHolder:
    def __init__(self, output_dir: str, train: bool):
        prefix = 'train' if train else 'test'
        self.__file_path = os.path.join(output_dir, f'{prefix}_loss.pkl')

        if os.path.exists(self.__file_path):
            with open(self.__file_path, 'rb') as f:
                self.losses = pickle.load(f)
        else:
            self.losses = []

    def add(self, loss):
        self.losses.append(loss)

    def save(self):
        with open(self.__file_path, 'wb') as f:
            pickle.dump(self.losses, f)

    def numpy(self) -> np.array:
        with torch.no_grad():
            return torch.stack(self.losses).cpu().numpy()
