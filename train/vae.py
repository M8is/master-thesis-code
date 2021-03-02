import torch


class VAE:
    def __init__(self, vae_model, data_holder, device, optimizer, learning_rate):
        self.vae_model = vae_model.to(device)
        self.data_holder = data_holder
        self.optimizer = optimizer(vae_model.parameters(), lr=learning_rate)
        self.device = device

    def train_epoch(self):
        train_losses = []
        test_losses = []
        self.vae_model.train()
        for batch_id, (x_batch, _) in enumerate(self.data_holder.train_holder):
            x_batch = x_batch.to(self.device)
            params, x_preds = self.vae_model(x_batch)
            losses = self.__bce_loss(x_batch, x_preds)
            self.optimizer.zero_grad()
            self.vae_model.backward(params, losses)
            kl = self.vae_model.probabilistic.distribution.kl(params)
            self.optimizer.step()
            train_losses.append((losses.detach().mean() + kl.detach().mean()))
            test_losses.append(self.test_epoch())
        return torch.stack(train_losses), torch.stack(test_losses)

    def test_epoch(self):
        with torch.no_grad():
            set_previous_mode = self.vae_model.train if self.vae_model.training else self.vae_model.eval
            self.vae_model.eval()
            test_losses = []
            for x_batch, _ in self.data_holder.test_holder:
                x_batch = x_batch.to(self.device)
                params, x_preds = self.vae_model(x_batch)
                losses = self.__bce_loss(x_batch, x_preds) + self.vae_model.probabilistic.distribution.kl(params)
                test_losses.append(losses.detach().mean())

            set_previous_mode()
            return torch.tensor(test_losses).mean()

    def get_estimator_stds(self, sample_size):
        result = []
        batch_size = self.data_holder.train_holder.batch_size
        for x_batch, _ in self.data_holder.train_holder:
            if x_batch.size(0) != batch_size:
                continue
            x_batch = x_batch.to(self.device)
            grads = []
            for _ in range(sample_size):
                self.vae_model.train()
                params, x_preds = self.vae_model(x_batch)
                for p in params:
                    p.retain_grad()
                losses = self.__bce_loss(x_batch, x_preds)
                self.optimizer.zero_grad()
                self.vae_model.backward(params, losses)
                for i, p in enumerate(params):
                    grad = p.grad.mean().unsqueeze(0)
                    try:
                        grads[i] = torch.cat((grads[i], grad), dim=0)
                    except IndexError:
                        grads.append(grad)
            result.append(torch.stack(grads).std(dim=1))
        return torch.stack(result).mean(dim=0)

    def __bce_loss(self, x, x_pred):
        # Use no reduction to get separate losses for each image
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        x_orig = x.view(-1, self.data_holder.height * self.data_holder.width).expand_as(x_pred)
        return binary_cross_entropy(x_pred, x_orig).sum(dim=-1)
