import pickle
from pathlib import Path
from time import process_time
from typing import Optional, List

import torch


class TensorHolder:
    def __init__(self, dir_path: str, name: str, add_timestamps: bool = False):
        self.name = name

        self.__data_file_path = self.__get_data_file_path(dir_path, name)
        self.__data = []

        self.__with_timestamps = add_timestamps
        if self.__with_timestamps:
            self.__timestamps_file_path = self.__get_timestamps_file_path(dir_path, name)
            self.__timestamps = []

    @staticmethod
    def load(dir_path: str, name: str) -> 'TensorHolder':
        data_file = TensorHolder.__get_data_file_path(dir_path, name)
        if not data_file.exists():
            raise FileNotFoundError(f"File not found: {data_file}.")

        timestamps_file = TensorHolder.__get_timestamps_file_path(dir_path, name)
        with_timestamps = timestamps_file.exists()

        holder = TensorHolder(dir_path, name, with_timestamps)
        holder.__data = TensorHolder.__load_tensors(data_file)
        if with_timestamps:
            holder.__timestamps = TensorHolder.__load_tensors(timestamps_file)

        return holder

    @property
    def data(self) -> torch.tensor:
        if self.is_empty():
            raise ValueError("Attempting to access empty holder.")
        return torch.tensor(self.__data)

    @property
    def timestamps(self) -> Optional[torch.tensor]:
        return torch.tensor(self.__timestamps) if self.__with_timestamps and self.__timestamps else None

    def add(self, tensor: torch.Tensor) -> None:
        if self.__with_timestamps:
            self.__timestamps.append(process_time())
        self.__data.append(tensor.detach().cpu())

    def save(self) -> None:
        if not self.is_empty():
            self.__save_tensors(self.__data_file_path, self.__data)
            if self.__with_timestamps:
                self.__save_tensors(self.__timestamps_file_path, self.__timestamps)

    def is_empty(self) -> bool:
        return len(self.__data) == 0

    @staticmethod
    def __get_data_file_path(dir_path: str, name) -> Path:
        return Path(dir_path) / f'{name}.pkl'

    @staticmethod
    def __get_timestamps_file_path(dir_path: str, name) -> Path:
        return Path(dir_path) / f'{name}_timestamps.pkl'

    @staticmethod
    def __save_tensors(path: Path, tensors: List[torch.Tensor]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(tensors, f)

    @staticmethod
    def __load_tensors(path: Path) -> List[torch.Tensor]:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f'TensorHolder({self.__data_file_path})'

    def __str__(self) -> str:
        return self.name
