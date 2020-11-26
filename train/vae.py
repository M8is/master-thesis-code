import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHolder:
    def __init__(self):
        self.train_holder = None
        self.test_holder = None
        self.height = None
        self.width = None

    def load_datasets(self):
        self.train_holder = DataLoader(
            datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), shuffle=True)
        self.test_holder = DataLoader(
            datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()), shuffle=True)
        _, self.height, self.width = self.train_holder.dataset.data.shape


class VAE:
    def __init__(self, vae_model, data_holder, optimizer, learning_rate=1e-2):
        self.vae_model = vae_model
        self.data_holder = data_holder
        self.optimizer = optimizer(vae_model.parameters(), lr=learning_rate)

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            train_loss = self.train_epoch().mean()
            VAE.print_loss(epoch+1, epochs, train_loss, 'Avg Train')
            test_loss = self.test_epoch().mean()
            VAE.print_loss(epoch+1, epochs, test_loss, 'Avg Test')
            losses.append((train_loss, test_loss))
        return losses

    def train_epoch(self):
        self.vae_model.train()
        train_losses = []
        for batch_id, (x_batch, _) in enumerate(self.data_holder.train_holder):
            self.optimizer.zero_grad()
            params, x_preds = self.vae_model(x_batch)
            losses = self.__loss_func(x_batch, x_preds, *params)
            self.vae_model.backward(losses)
            self.optimizer.step()
            train_losses.append(losses.detach().mean())
        return torch.tensor(train_losses, requires_grad=False)

    def test_epoch(self):
        self.vae_model.eval()
        test_losses = []
        for batch_id, (x_batch, _) in enumerate(self.data_holder.test_holder):
            params, x_preds = self.vae_model(x_batch)
            losses = self.__loss_func(x_batch, x_preds, *params)
            test_losses.append(losses.detach().mean())
        return torch.tensor(test_losses, requires_grad=False)

    @staticmethod
    def print_loss(epoch, epochs, loss, name):
        print(f"===> Epoch: {epoch}/{epochs}, {name} Loss: {loss:.3f}")

    def __loss_func(self, x, x_pred, mu_z, log_sigma_z):
        """
        VAE loss function = kl + reconstruct
            kl: KL between multivariate gaussian and standard multivariate gaussian.
            reconstruct: reconstruction the image generated by the decoder and the original one.
        """
        kl = 0.5 * torch.sum(torch.exp(log_sigma_z) + torch.pow(mu_z, 2) - 1 - log_sigma_z)
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        x_orig = x.view(-1, self.data_holder.height * self.data_holder.width).repeat(x_pred.size(0), 1)
        reconstruct = binary_cross_entropy(x_pred, x_orig).mean(dim=1)
        return reconstruct + kl
