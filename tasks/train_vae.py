import torch
from os import path, makedirs

from torchvision.utils import save_image

from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.model_factory import get_vae
from utils.trainer import Trainer


class TrainVAE(Trainer):
    def __init__(self, vae_type, hidden_dims, sample_size, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        estimator_tag = kwargs['mc_estimator']
        distribution_tag = kwargs['distribution']
        estimator = get_estimator(estimator_tag, distribution_tag, sample_size, self.device, self.latent_dim, *args, **kwargs)
        self.__model = get_vae(vae_type, estimator, self.data_holder.dims, hidden_dims).to(self.device)
        self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @property
    def variance_interval(self):
        return 20

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer

    def loss(self, inputs, outputs):
        x, x_recon = inputs, outputs
        n_data_dims = len(x.size()) - 1
        # Use no reduction to get separate losses for each image
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        return binary_cross_entropy(x_recon, x.expand_as(x_recon)).flatten(start_dim=-n_data_dims).sum(dim=-1)

    def post_training(self):
        self.__generate_images()

    def __generate_images(self):
        if not path.exists(self.results_dir):
            makedirs(self.results_dir)
        else:
            print(f"Skipping: '{self.results_dir}' already exists.")
            return

        with eval_mode(self.model):
            print(f'Generating images for in `{self.results_dir}`...')
            n = min(self.data_holder.batch_size, 8)
            for batch_id, (x_batch, _) in enumerate(self.data_holder.test):
                x_batch = x_batch[:n].to(self.device)
                _, x_pred_batch = self.model(x_batch)
                comparison = torch.cat((x_batch, x_pred_batch.view(x_batch.shape)))
                save_image(comparison, path.join(self.results_dir, f'recon_{batch_id}.png'), nrow=n)
