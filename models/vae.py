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

    def backward(self, loss):
        return self.probabilistic.backward(loss)


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = [torch.nn.Linear(hidden_dim, latent_dim) for latent_dim in latent_dims]

    def forward(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.input_dim)))
        h2 = F.relu(self.fc2(h1))
        return [fc(h2) for fc in self.fc3]


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = torch.sigmoid(self.fc2(h1))
        return h2
