import torch


class ProbabilisticObjective(torch.nn.Module):
    def __init__(self, init_mean, init_cov):
        super().__init__()
        self.mean = torch.nn.Parameter(init_mean)
        self.cov = torch.nn.Parameter(init_cov)
        
    def forward_mc(self, x):
        return self.gradient_estimate(self.get_samples(x)).detach() * self.mean
    
    def forward(self, x):
        if self.train:
            return self.forward_mc(x)
        return self.distribution(self.mean, self.cov).sample()
        
    def gradient_estimate(self, sample):
        raise NotImplementedError
