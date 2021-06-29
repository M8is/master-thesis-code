from abc import ABC, abstractmethod
from typing import List

import sklearn.datasets
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder(ABC):
    DATA_ROOT = 'data'

    _datasets = dict()

    def __init__(self, batch_size, *args, **kwargs):
        self.__train_holder, self.__test_holder = self._load(batch_size, *args, **kwargs)
        self.batch_size = batch_size

    @property
    def train(self):
        return self.__train_holder()

    @property
    def test(self):
        return self.__test_holder()

    @property
    @abstractmethod
    def dims(self) -> List[int]:
        pass

    @staticmethod
    def get(dataset: str, *args, **kwargs) -> 'DataHolder':
        if dataset.lower() not in DataHolder._datasets:
            raise ValueError(f'Invalid dataset `{dataset}`. Allowed values: {list(DataHolder._datasets.keys())}.')
        return DataHolder._datasets[dataset.lower()](*args, **kwargs)

    @staticmethod
    def register_dataset(key: str):
        def __reg(cls):
            DataHolder._datasets[key] = cls

        return __reg

    @staticmethod
    @abstractmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        pass


@DataHolder.register_dataset('empty')
class EmptyDataset(DataHolder):
    @property
    def dims(self):
        return ()

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        empty = [(torch.empty(batch_size), torch.empty(batch_size))]
        return lambda: empty, lambda: empty


@DataHolder.register_dataset('mnist')
class MNIST(DataHolder):
    @property
    def dims(self):
        return 1, 28, 28

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.MNIST(root=DataHolder.DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.MNIST(root=DataHolder.DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('omniglot')
class Omniglot(DataHolder):
    @property
    def dims(self):
        return 1, 105, 105

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.Omniglot(root=DataHolder.DATA_ROOT, background=True, download=True,
                              transform=transforms.ToTensor()), shuffle=shuffle, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.Omniglot(root=DataHolder.DATA_ROOT, background=False, download=True,
                              transform=transforms.ToTensor()), shuffle=shuffle, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('svhn')
class SteetViewHouseNumbers(DataHolder):
    @property
    def dims(self):
        return 1, 32, 32

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.SVHN(root=DataHolder.DATA_ROOT, split='train', download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.SVHN(root=DataHolder.DATA_ROOT, split='test', download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('celeba')
class CelebA(DataHolder):
    @property
    def dims(self):
        return 3, 64, 64

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.CelebA(root=DataHolder.DATA_ROOT, split='train', download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.CelebA(root=DataHolder.DATA_ROOT, split='test', download=True, transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('fashionmnist')
class FashionMNIST(DataHolder):
    @property
    def dims(self):
        return 3, 1, 1  # TODO

    @staticmethod
    def _load(batch_size, shuffle=True, *args, **kwargs):
        loader_args = dict()
        if 'device' in kwargs:
            loader_args['pin_memory'] = 'cuda' in kwargs['device']
        if 'num_workers' in kwargs:
            loader_args['num_workers'] = kwargs['num_workers']

        train_holder = DataLoader(
            datasets.FashionMNIST(root=DataHolder.DATA_ROOT, train=True, download=True,
                                  transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        test_holder = DataLoader(
            datasets.FashionMNIST(root=DataHolder.DATA_ROOT, train=False, download=True,
                                  transform=transforms.ToTensor()),
            shuffle=shuffle, batch_size=batch_size, **loader_args)
        return lambda: train_holder, lambda: test_holder


@DataHolder.register_dataset('cancer')
class BreastCancer(DataHolder):
    @property
    def dims(self):
        return 30,

    @staticmethod
    def _load(batch_size, shuffle=True, *_, **__):
        x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        # TODO: save/load in self.DATA_ROOT
        x_train, x_test, y_train, y_test = (torch.tensor(d, requires_grad=False) for d in
                                            sklearn.model_selection.train_test_split(x, y, shuffle=shuffle))
        return lambda: batch_iterator(batch_size, x_train, y_train), lambda: batch_iterator(batch_size, x_test, y_test)


def batch_iterator(batch_size, x, y):
    assert x.shape[0] == y.shape[0]
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]
