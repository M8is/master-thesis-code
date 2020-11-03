import torch

from probabilistic_objective_gradient import ProbabilisticObjective


class PathwiseNormal(ProbabilisticObjective):
    def __init__(self, objective_derivative, episode_size, init_mean, init_cov):
        super().__init__(init_mean, init_cov)
        self.objective_derivative = objective_derivative
        self.episode_size = episode_size
        self.distribution = torch.distributions.MultivariateNormal
        self.losses = []
    
    def get_samples(self, x):
        dist = self.distribution(torch.zeros_like(self.mean), torch.eye(self.mean.size()[0]))
        return dist.sample((self.episode_size,))
        
    def gradient_estimate(self, x):
        samples = self.get_samples(x)
        l = torch.cholesky(self.cov)
        scaled_loss_derivs = self.objective_derivative(torch.matmul(samples, l) + self.mean)
        mean_surrogate_loss = scaled_loss_derivs.mean(dim=0)
        self.losses.append(mean_surrogate_loss.detach().clone())
        return mean_surrogate_loss
