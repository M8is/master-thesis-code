from abc import ABC, abstractmethod

import sklearn.datasets
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder(ABC):
    DATA_ROOT = 'data'

    _datasets = dict()

    def __init__(self, batch_size, *args, **kwargs):
        self.__train_holder, self.__test_holder = self.load(batch_size, *args, **kwargs)
        self.batch_size = batch_size

    @property
    def train(self):
        return self.__train_holder()

    @property
    def test(self):
        return self.__test_holder()

    @property
    @abstractmethod
    def dims(self):
        pass

    @staticmethod
    def get(dataset_tag: str, *args, **kwargs) -> 'DataHolder':
        if dataset_tag.lower() not in DataHolder._datasets:
            raise ValueError(f'Invalid dataset `{dataset_tag}`. Allowed values: {list(DataHolder._datasets.keys())}.')
        dataset = DataHolder._datasets[dataset_tag.lower()]
        return dataset(*args, **kwargs)

    @staticmethod
    def register_dataset(key: str):
        def __reg(cls):
            DataHolder._datasets[key] = cls
        return __reg

    @abstractmethod
    def load(self, batch_size, *args, **kwargs):
        pass


@DataHolder.register_dataset('mnist')
class MNIST(DataHolder):
    @property
    def dims(self):
        return 784

    def load(self, batch_size, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.MNIST(root=self.DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()),
            shuffle=True, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.MNIST(root=self.DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()),
            shuffle=True, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('omniglot')
class Omniglot(DataHolder):
    @property
    def dims(self):
        return 105*105

    def load(self, batch_size, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.Omniglot(root=self.DATA_ROOT, background=True, download=True, transform=transforms.ToTensor()),
            shuffle=True, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.Omniglot(root=self.DATA_ROOT, background=False, download=True, transform=transforms.ToTensor()),
            shuffle=True, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('cancer')
class BreastCancer(DataHolder):
    @property
    def dims(self):
        return 30

    def load(self, batch_size, *_, **__):
        x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        x_train, x_test, y_train, y_test = (torch.tensor(d, requires_grad=False) for d in
                                            sklearn.model_selection.train_test_split(x, y, shuffle=True))
        return lambda: batch_iterator(batch_size, x_train, y_train), lambda: batch_iterator(batch_size, x_test, y_test)


def batch_iterator(batch_size, x, y):
    assert x.shape[0] == y.shape[0]
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]
