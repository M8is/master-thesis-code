import os
import pickle

import numpy as np
import torch


class TensorHolder:
    def __init__(self, output_dir: str, file_name: str):
        self.__file_path = os.path.join(output_dir, f'{file_name}.pkl')

        if os.path.exists(self.__file_path):
            with open(self.__file_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []

    def add(self, tensor):
        self.data.append(tensor.detach().cpu().unsqueeze(0))

    def save(self):
        if not self.is_empty():
            os.makedirs(os.path.dirname(self.__file_path), exist_ok=True)
            with open(self.__file_path, 'wb') as f:
                pickle.dump(self.data, f)

    def numpy(self) -> np.array:
        with torch.no_grad():
            return torch.cat(self.data).numpy().flatten()

    def is_empty(self) -> bool:
        return len(self.data) == 0


class LossHolder(TensorHolder):
    def __init__(self, output_dir: str, train: bool):
        super().__init__(output_dir, f"{'train' if train else 'test'}_loss")
