import torch


class LogisticRegression:
    def __init__(self, model, data_holder, device, optimizer, learning_rate):
        self.model = model.to(device)
        self.data_holder = data_holder
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.device = device

    def train_epoch(self):
        train_losses = []
        test_losses = []
        self.model.train()
        for x_batch, y_batch in self.data_holder.train:
            params, y_preds = self.model(x_batch.to(self.device))
            losses = self.__bce_loss(y_batch.to(self.device), y_preds)
            self.optimizer.zero_grad()
            self.model.backward(params, losses)
            self.optimizer.step()
            train_losses.append(self.__mean_train_loss())
            test_losses.append(self.__mean_test_loss())
        return torch.stack(train_losses), torch.stack(test_losses)

    def __mean_train_loss(self):
        with torch.no_grad():
            set_previous_mode = self.model.train if self.model.training else self.model.eval
            self.model.eval()
            test_losses = []
            for x_batch, y_batch in self.data_holder.train:
                params, y_preds = self.model(x_batch.to(self.device))
                kl = self.model.probabilistic.distribution.kl(params)
                losses = self.__bce_loss(y_batch.to(self.device), y_preds) + kl
                test_losses.append(losses.detach().mean())
            set_previous_mode()
            return torch.stack(test_losses).mean()

    def __mean_test_loss(self):
        with torch.no_grad():
            set_previous_mode = self.model.train if self.model.training else self.model.eval
            self.model.eval()
            test_losses = []
            for x_batch, y_batch in self.data_holder.test:
                params, y_preds = self.model(x_batch.to(self.device))
                kl = self.model.probabilistic.distribution.kl(params)
                losses = self.__bce_loss(y_batch.to(self.device), y_preds) + kl
                test_losses.append(losses.detach().mean())
            set_previous_mode()
            return torch.stack(test_losses).mean()

    @staticmethod
    def __bce_loss(y, y_pred):
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        return binary_cross_entropy(y_pred, y.expand_as(y_pred).double())
