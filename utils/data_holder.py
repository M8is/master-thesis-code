from abc import ABC, abstractmethod

import sklearn.datasets
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder(ABC):
    _datasets = dict()

    def __init__(self, batch_size):
        self.__train_holder, self.__test_holder = self.load(batch_size)

    @property
    def train_holder(self):
        return self.__train_holder()

    @property
    def test_holder(self):
        return self.__test_holder()

    @staticmethod
    def get(dataset_tag: str, *args, **kwargs) -> 'DataHolder':
        if dataset_tag.lower() not in DataHolder._datasets:
            raise ValueError(f'Invalid dataset `{dataset_tag}`. Allowed values: {list(DataHolder._datasets.keys())}.')
        dataset = DataHolder._datasets[dataset_tag.lower()]
        return dataset(*args, **kwargs)

    @abstractmethod
    def load(self, batch_size):
        pass

    @staticmethod
    def register_dataset(key: str):
        def __reg(cls):
            DataHolder._datasets[key] = cls
        return __reg


@DataHolder.register_dataset('mnist')
class MNIST(DataHolder):
    height = 28
    width = 28

    def load(self, batch_size):
        train_holder = DataLoader(
            datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), shuffle=True,
            batch_size=batch_size)
        test_holder = DataLoader(
            datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()), shuffle=True,
            batch_size=batch_size)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('cancer')
class BreastCancer(DataHolder):
    def load(self, batch_size):
        x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        x_train, x_test, y_train, y_test = (torch.tensor(d, requires_grad=False) for d in
                                            sklearn.model_selection.train_test_split(x, y, shuffle=True))
        return lambda: batch_iterator(batch_size, x_train, y_train), lambda: batch_iterator(batch_size, x_test, y_test)


def batch_iterator(batch_size, x, y):
    assert x.shape[0] == y.shape[0]
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]
