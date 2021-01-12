import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, probabilistic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.probabilistic = probabilistic

    def forward(self, x):
        params = self.encoder(x)
        samples = self.probabilistic.grad_samples(params) if self.training else self.probabilistic.sample(params)
        replications = self.decoder(samples)
        return params, replications

    def backward(self, losses, params):
        # Set decoder gradients. Also sets encoder gradients, if samples where not detached.
        losses.mean().backward(retain_graph=True)
        # Set encoder gradients.
        self.probabilistic.kl(params).mean().backward()
        self.probabilistic.backward(params, losses.detach())


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = [torch.nn.Linear(hidden_dim, latent_dim) for latent_dim in latent_dims]

    def forward(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.input_dim)))
        return [fc(h1) for fc in self.fc2]


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))
