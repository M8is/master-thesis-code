import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.utils.data

from distributions.distribution_base import Distribution
from models.stochastic_model import StochasticModel
from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.stopwatch import Stopwatch
from utils.tensor_holders import TensorHolder


class StochasticTrainer(ABC):
    def __init__(self, results_dir: str, epochs: int, mc_estimator: str, sample_size: int, device: str = 'cpu',
                 compute_variance: bool = False, compute_perf: bool = False, optimize_kld: bool = True,
                 iteration_print_interval: Optional[int] = 100, epoch_print_interval: Optional[int] = 1,
                 save_model_interval: Optional[int] = None, **kwargs):
        self.results_dir = results_dir
        self.device = device
        self.epochs = epochs
        self.sample_size = sample_size
        self.data_holder = DataHolder.get(**kwargs)
        self.gradient_estimator = get_estimator(mc_estimator, **kwargs)
        self.compute_variance = compute_variance
        self.compute_perf = compute_perf
        self.optimize_kld = optimize_kld
        self.epoch_print_interval = epoch_print_interval
        self.iteration_print_interval = iteration_print_interval
        self.save_model_interval = save_model_interval
        self.stopwatch = Stopwatch()

        self.train_losses = TensorHolder(self.results_dir, 'train_loss')
        self.test_losses = TensorHolder(self.results_dir, 'test_loss')
        if self.compute_variance:
            self.estimator_stds = TensorHolder(self.results_dir, 'estimator_stds')
        if self.compute_perf:
            self.estimator_times = TensorHolder(self.results_dir, 'gradient_calculation_times')
            self.iteration_times = TensorHolder(self.results_dir, 'iteration_times')

    def train(self) -> Iterable[str]:
        print(f'Training with {self.gradient_estimator}.')
        self.model.to(self.device)
        model_path = Path(self.results_dir) / f'{self.gradient_estimator.name()}_0.pt'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, model_path)
        for epoch in range(1, self.epochs + 1):
            self.__train_epoch()
            self.__test_epoch()
            if epoch == self.epochs or (self.save_model_interval and epoch % self.save_model_interval == 0):
                model_path = Path(self.results_dir) / f'{self.gradient_estimator.name()}_{epoch}.pt'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model, model_path)
            if self.epoch_print_interval and (epoch % self.epoch_print_interval == 0 or epoch == self.epochs):
                print(f"Epoch: {epoch}/{self.epochs}, Train loss: {self.train_losses.data.numpy()[-1].mean():.2f}, "
                      f"Test loss: {self.test_losses.data.numpy()[-1].mean():.2f}",
                      flush=True)
                print(60 * "-")

        self.train_losses.save()
        self.test_losses.save()
        saved_metrics = [self.train_losses.name, self.test_losses.name]
        if self.compute_perf:
            print(f'Estimating performance of {self.gradient_estimator} ...')
            self.estimator_times.add(self.__estimate_time(n_estimates=10000))
            self.estimator_times.save()
            saved_metrics.append(self.estimator_times.name)
            self.iteration_times.save()
            saved_metrics.append(self.iteration_times.name)
        if self.compute_variance:
            self.estimator_stds.save()
            saved_metrics.append(self.estimator_stds.name)
        return saved_metrics

    def __train_epoch(self) -> None:
        self.model.train()
        for batch_id, (x_batch, y_batch) in enumerate(self.data_holder.train):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            distribution = self.model.encode(x_batch)
            interpretation = self.model.interpret(distribution.sample(), x_batch)
            loss = self.loss(x_batch, y_batch, interpretation).mean()
            self.optimizer.zero_grad()
            if self.optimize_kld:
                kld = distribution.kl().mean()
                kld.backward(retain_graph=True)
            else:
                kld = 0
            loss_fn = self.__get_loss_fn(x_batch, y_batch)
            self.stopwatch.resume()
            distribution.backward(self.gradient_estimator, loss_fn, self.sample_size)
            iteration_time = self.stopwatch.pause()
            if loss.requires_grad:
                loss.backward()
            self.optimizer.step()
            if self.compute_perf:
                self.iteration_times.add(torch.tensor(iteration_time))
            if self.iteration_print_interval and batch_id % self.iteration_print_interval == 0:
                elbo_prefix = f"| ELBO: {-(loss + kld):.2f} " if self.optimize_kld else ''
                print(f"{elbo_prefix}| Loss: {loss:.1f} | KLD: {kld:.1f} |")
            if self.compute_variance and batch_id % self.variance_interval == 0:
                self.estimator_stds.add(self.__estimate_std(self.model.encode(x_batch), loss_fn, n_estimates=500))
            self.train_losses.add(loss + kld)

    def __test_epoch(self) -> None:
        with eval_mode(self.model):
            test_losses = []
            for x_batch, y_batch in self.data_holder.test:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                distribution = self.model.encode(x_batch)
                x_recons = self.model.interpret(distribution.sample(), x_batch)
                loss = self.loss(x_batch, y_batch, x_recons).mean()
                kld = distribution.kl().mean() if self.optimize_kld else 0
                test_losses.append(loss + kld)
            self.test_losses.add(torch.tensor(test_losses).mean())

    def __estimate_std(self, distribution: Distribution, loss_fn: Callable[[torch.Tensor], torch.Tensor],
                       n_estimates: int) -> torch.Tensor:
        self.gradient_estimator.freeze()
        sample_size = 1
        grads = []
        self.optimizer.zero_grad()
        for i in range(n_estimates):
            retain_graph = (i + 1) < n_estimates
            distribution.params.retain_grad()
            distribution.backward(self.gradient_estimator, loss_fn, sample_size, retain_graph=retain_graph)
            grads.append(distribution.params.grad)
            self.optimizer.zero_grad()
        self.gradient_estimator.unfreeze()
        return torch.stack(grads).std(dim=0).mean()

    def __estimate_time(self, n_estimates: int) -> torch.Tensor:
        self.model.train()

        for x, y in self.data_holder.train:
            x = x.to(self.device)
            y = y.to(self.device)
            loss_fn = self.__get_loss_fn(x, y)
            distribution = self.model.encode(x)
            times = []
            self.optimizer.zero_grad()
            for i in range(n_estimates):
                retain_graph = (i + 1) < n_estimates
                before = time.process_time()
                distribution.backward(self.gradient_estimator, loss_fn, self.sample_size, retain_graph=retain_graph)
                after = time.process_time()
                times.append(after - before)
                self.optimizer.zero_grad()
            times = torch.FloatTensor(times)
            return torch.stack((times.mean(), times.std()))

    def __get_loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda samples: self.loss(x, y, self.model.interpret(samples, x))

    @property
    def variance_interval(self) -> int:
        raise ValueError("Computing variance not expected.")

    @property
    @abstractmethod
    def model(self) -> StochasticModel:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        pass
