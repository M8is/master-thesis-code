import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder:
    _datasets = {
        'mnist': datasets.MNIST
    }

    def __init__(self):
        self.train_holder = None
        self.test_holder = None
        self.height = None
        self.width = None

    def load_datasets(self, dataset_tag, batch_size, shuffle=True):
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
    def __init__(self):
        self.test_loss = []

    def add(self, test_loss):
        self.test_loss.append(test_loss)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.test_loss, f)

    def test(self) -> np.array:
        with torch.no_grad():
            return torch.stack(self.test_loss).numpy()

    @staticmethod
    def load(file_path: str) -> 'LossHolder':
        with open(file_path, 'rb') as f:
            test_loss = pickle.load(f)
        result = LossHolder()
        result.test_loss = test_loss
        return result