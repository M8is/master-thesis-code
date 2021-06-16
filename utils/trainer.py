from abc import ABC, abstractmethod
from os import path
from typing import Tuple, Optional

import torch
import torch.utils.data

from utils.data_holder import DataHolder
from utils.eval_util import eval_mode
from utils.tensor_holders import LossHolder, TensorHolder


class Trainer(ABC):
    def __init__(self, results_dir: str, dataset: str, device: str, epochs: int, batch_size: int, mc_estimator: str,
                 compute_variance: bool = False, **kwargs):
        self.results_dir = results_dir
        self.device = device
        self.epochs = epochs
        self.data_holder = DataHolder.get(dataset, batch_size)
        self.estimator = mc_estimator
        self.compute_variance = compute_variance

        self.train_losses = LossHolder(self.results_dir, train=True)
        self.test_losses = LossHolder(self.results_dir, train=False)
        self.estimator_stds = TensorHolder(self.results_dir, 'estimator_stds')

    def train(self) -> Tuple[LossHolder, LossHolder, TensorHolder]:
        print(f'Training with {self.estimator}.')
        for epoch in range(1, self.epochs + 1):
            train_loss, test_loss, est_std = self.__train_epoch()
            self.train_losses.add(train_loss)
            self.test_losses.add(test_loss)
            if self.compute_variance:
                self.estimator_stds.add(est_std)
                self.estimator_stds.save()
            file_name = path.join(self.results_dir, f'{self.estimator}_{epoch}.pt')
            self.train_losses.save()
            self.test_losses.save()
            torch.save(self.model, file_name)
            print(f"Epoch: {epoch}/{self.epochs}, Train loss: {self.train_losses.numpy()[-1].mean():.2f}, "
                  f"Test loss: {self.test_losses.numpy()[-1].mean():.2f}",
                  flush=True)
        self.post_training()
        return self.train_losses, self.test_losses, self.estimator_stds

    def __train_epoch(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        train_losses = []
        test_losses = []
        estimator_stds = []
        self.model.train()
        print(60 * "-")
        for batch_id, (x_batch, _) in enumerate(self.data_holder.train):
            x_batch = x_batch.to(self.device)
            distribution, x_recon = self.model(x_batch)
            loss = self.loss(x_batch, x_recon).mean()
            kld = distribution.kl().mean()
            self.optimizer.zero_grad()
            kld.backward(retain_graph=True)

            def loss_fn(samples):
                return self.loss(x_batch, self.model.decoder(samples))

            self.model.probabilistic.backward(distribution, loss_fn)
            loss.backward()
            self.optimizer.step()
            if batch_id % 100 == 0:
                print(f"\r| ELBO: {-(loss + kld):.2f} | BCE loss: {loss:.1f} | KL Divergence: {kld:.1f} |")
            if self.compute_variance and batch_id % self.variance_interval == 0:
                distribution, _ = self.model(x_batch)
                estimator_stds.append(self.model.probabilistic.get_std(distribution, self.optimizer.zero_grad, loss_fn))

            train_losses.append(loss + kld)
        test_losses.append(self.__test_epoch())
        return torch.stack(train_losses), torch.stack(test_losses), torch.stack(
            estimator_stds) if estimator_stds else None

    def __test_epoch(self) -> torch.Tensor:
        with eval_mode(self.model):
            test_losses = []
            for x_batch, _ in self.data_holder.test:
                x_batch = x_batch.to(self.device)
                distribution, x_recons = self.model(x_batch)
                loss = self.loss(x_batch, x_recons).mean()
                kld = distribution.kl().mean()
                test_losses.append(loss + kld)
            return torch.tensor(test_losses).mean()

    @property
    def variance_interval(self) -> int:
        raise ValueError("Computing variance not expected.")

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def post_training(self) -> None:
        pass
