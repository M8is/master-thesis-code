import torch


class LossHolder:
    def __init__(self):
        self.train_loss = []
        self.test_loss = []

    def add(self, train_loss, test_loss):
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)

    def as_numpy(self):
        with torch.no_grad():
            train_loss = torch.stack(self.train_loss)
            test_loss = torch.stack(self.test_loss)
            return train_loss.numpy(), test_loss.numpy()


class VAE:
    def __init__(self, vae_model, data_holder, optimizer, learning_rate):
        self.vae_model = vae_model
        self.data_holder = data_holder
        self.optimizer = optimizer(vae_model.parameters(), lr=learning_rate)

    def train_epoch(self):
        train_losses = []
        test_losses = []
        for batch_id, (x_batch, _) in enumerate(self.data_holder.train_holder):
            self.vae_model.train()
            params, x_preds = self.vae_model(x_batch)
            losses = self.__bce_loss(x_batch, x_preds)
            self.optimizer.zero_grad()
            self.vae_model.backward(losses, params)
            self.optimizer.step()
            train_losses.append(losses + self.vae_model.probabilistic.kl(params))
            test_losses.append(self.test_epoch())
        return torch.cat(train_losses), torch.cat(test_losses)

    def test_epoch(self):
        with torch.no_grad():
            set_previous_mode = self.vae_model.train if self.vae_model.training else self.vae_model.eval
            self.vae_model.eval()
            test_losses = []
            for x_batch, _ in self.data_holder.test_holder:
                params, x_preds = self.vae_model(x_batch)
                losses = self.__bce_loss(x_batch, x_preds) + self.vae_model.probabilistic.kl(params)
                test_losses.append(losses.detach().mean())
                break # TODO delete break (just for debugging)

            set_previous_mode()
            return torch.tensor(test_losses).mean()

    def __bce_loss(self, x, x_pred):
        # Use no reduction to get separate losses for each image
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        x_orig = x.view(-1, self.data_holder.height * self.data_holder.width).repeat(x_pred.size(0), 1, 1)
        return binary_cross_entropy(x_pred, x_orig).mean(dim=2)
