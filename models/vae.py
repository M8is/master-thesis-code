import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, probabilistic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.probabilistic = probabilistic

    def forward(self, x):
        raw_params = self.encoder(x)
        if self.training:
            samples = self.probabilistic(raw_params)
        else:
            samples = self.probabilistic.distribution.sample(raw_params, with_grad=False)
        return raw_params, self.decoder(samples)

    def backward(self, raw_params, losses):
        # Mean over batch
        losses = losses.mean(dim=-1)
        # Set encoder gradients.
        self.probabilistic.backward(raw_params, losses.detach(), retain_graph=True)
        # Set decoder gradients. Also sets encoder gradients, if samples where not detached.
        losses.mean().backward()


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, sum(latent_dims))

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(h1)
        return self.fc3(h2)


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
