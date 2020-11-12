import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, probabilistic, param_preprocessor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.probabilistic = probabilistic
        self.param_preprocessor = param_preprocessor

    def forward(self, x, loss_fn):
        params = self.encoder(x)
        if self.training:
            return self.probabilistic(x, lambda samples: loss_fn(x, self.decoder(samples), *params).mean(), params)
        else:
            return self.decoder(self.probabilistic(x, None, *params))


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims, post_processor=None):
        super().__init__()
        self.input_dim = input_dim
        self.post_processor = post_processor
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = [torch.nn.Linear(hidden_dim, latent_dim) for latent_dim in latent_dims]

    def forward(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.input_dim)))
        out = (fc(h1) for fc in self.fc2)
        return out if self.post_processor is None else self.post_processor(*out)


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h1))
