import time
from abc import ABC, abstractmethod
from os import path
from typing import Tuple, Callable

import torch
import torch.utils.data

from mc_estimators.distributions.distribution_base import Distribution
from utils.data_holder import DataHolder
from utils.eval_util import eval_mode
from utils.tensor_holders import LossHolder, TensorHolder


class Trainer(ABC):
    def __init__(self, results_dir: str, device: str, epochs: int, mc_estimator: str, compute_variance: bool = False,
                 compute_perf: bool = False, print_interval: int = 100, **kwargs):
        self.results_dir = results_dir
        self.device = device
        self.epochs = epochs
        self.data_holder = DataHolder.get(**kwargs)
        self.estimator = mc_estimator
        self.compute_variance = compute_variance
        self.compute_perf = compute_perf
        self.print_interval = print_interval

    def train(self) -> Tuple[LossHolder, LossHolder, TensorHolder, TensorHolder]:
        train_losses = LossHolder(self.results_dir, train=True)
        test_losses = LossHolder(self.results_dir, train=False)
        estimator_stds = TensorHolder(self.results_dir, 'estimator_stds')
        estimator_times = TensorHolder(self.results_dir, 'estimator_times')

        print(f'Training with {self.estimator}.')
        for epoch in range(1, self.epochs + 1):
            train_loss, est_std = self.__train_epoch()
            test_loss = self.__test_epoch()
            train_losses.add(train_loss)
            test_losses.add(test_loss)
            if est_std.numel():
                estimator_stds.add(est_std)
            print(f"Epoch: {epoch}/{self.epochs}, Train loss: {train_losses.numpy()[-1].mean():.2f}, "
                  f"Test loss: {test_losses.numpy()[-1].mean():.2f}",
                  flush=True)
            print(60 * "-")
        if self.compute_perf:
            print(f'Estimating performance of {self.estimator} ...')
            estimator_times.add(self.__estimate_time(n_estimates=10000))

        train_losses.save()
        test_losses.save()
        estimator_times.save()
        estimator_stds.save()
        torch.save(self.model, path.join(self.results_dir, f'{self.estimator}_{self.epochs}.pt'))
        self.post_training()
        return train_losses, test_losses, estimator_stds, estimator_times

    def __train_epoch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        train_losses = []
        estimator_stds = []
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
            if batch_id % self.print_interval == 0:
                print(f"\r| ELBO: {-(loss + kld):.2f} | BCE loss: {loss:.1f} | KL Divergence: {kld:.1f} |")
            if self.compute_variance and batch_id % self.variance_interval == 0:
                distribution, _ = self.model(x_batch)
                estimator_stds.append(self.__estimate_std(distribution, loss_fn, n_estimates=500))
            train_losses.append(loss + kld)
        return torch.tensor(train_losses), torch.tensor(estimator_stds)

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

    def __estimate_std(self, distribution: Distribution, loss_fn: Callable[[torch.Tensor], torch.Tensor],
                       n_estimates: int) -> torch.Tensor:
        old_sample_size = self.model.probabilistic.sample_size
        self.model.probabilistic.sample_size = 1
        grads = []
        self.optimizer.zero_grad()
        for i in range(n_estimates):
            retain_graph = (i + 1) < n_estimates
            distribution.params.retain_grad()
            self.model.probabilistic.backward(distribution, loss_fn, retain_graph=retain_graph)
            grads.append(distribution.params.grad)
            self.optimizer.zero_grad()
        self.model.probabilistic.sample_size = old_sample_size
        return torch.stack(grads).std(dim=0).mean()

    def __estimate_time(self, n_estimates: int) -> torch.Tensor:
        self.model.train()
        x_batch = None
        for (b, _) in self.data_holder.train:
            x_batch = b.to(self.device)
        distribution, _ = self.model(x_batch)

        def loss_fn(samples):
            return self.loss(x_batch, self.model.decoder(samples))

        times = []
        self.optimizer.zero_grad()
        for i in range(n_estimates):
            retain_graph = (i + 1) < n_estimates
            before = time.process_time()
            self.model.probabilistic.backward(distribution, loss_fn, retain_graph=retain_graph)
            after = time.process_time()
            times.append(after - before)
            self.optimizer.zero_grad()
        times = torch.FloatTensor(times)
        return torch.stack((times.mean(), times.std()))

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
