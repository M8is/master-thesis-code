from os import path, makedirs

import torch
import torch.utils.data
from torchvision.utils import save_image

from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.model_factory import get_vae
from utils.tensor_holders import LossHolder, TensorHolder


def train_vae(results_dir, vae_type, dataset, device, hidden_dims, latent_dim, epochs, learning_rate,
              mc_estimator, distribution, batch_size, **kwargs):
    data_holder = DataHolder.get(dataset, batch_size)

    # Create model
    estimator = get_estimator(estimator_tag=mc_estimator, distribution_tag=distribution, device=device,
                              latent_dim=latent_dim, **kwargs)
    vae_network = get_vae(vae_type, estimator, data_holder.dims, hidden_dims).to(device)
    optimizer = torch.optim.Adam(vae_network.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    estimator_stds = TensorHolder(results_dir, 'estimator_stds')
    for epoch in range(1, epochs + 1):
        train_loss, test_loss, est_std = __train_epoch(vae_network, data_holder, device, optimizer)
        train_losses.add(train_loss)
        test_losses.add(test_loss)
        estimator_stds.add(est_std)
        file_name = path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        train_losses.save()
        test_losses.save()
        estimator_stds.save()
        torch.save(vae_network, file_name)
        print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy()[-1].mean():.2f}, "
              f"Test loss: {test_losses.numpy()[-1].mean():.2f}",
              flush=True)
    __generate_images(vae_network, path.join(results_dir, f'images_{epochs}'), data_holder, device)
    return train_losses, test_losses, estimator_stds


def __train_epoch(vae_model, data_holder, device, optimizer):
    train_losses = []
    test_losses = []
    estimator_stds = []
    vae_model.train()
    print(60 * "-")
    for batch_id, (x_batch, _) in enumerate(data_holder.train):
        x_batch = x_batch.to(device)
        raw_params, x_recon = vae_model(x_batch)
        loss = __bce_loss(x_batch, x_recon, len(data_holder.dims)).mean()
        kld = vae_model.probabilistic.distribution.kl(raw_params).mean()
        optimizer.zero_grad()
        kld.backward(retain_graph=True)

        def loss_fn(samples):
            return __bce_loss(x_batch, vae_model.decoder(samples), len(data_holder.dims))

        vae_model.probabilistic.backward(raw_params, loss_fn)
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            raw_params, _ = vae_model(x_batch)
            stds = vae_model.probabilistic.generate_stds(raw_params, loss_fn, optimizer.zero_grad)
            print(f"\r| ELBO: {-(loss + kld):.2f} | BCE loss: {loss:.1f} | KL Divergence: {kld:.1f} |")
            estimator_stds.append(stds)
        train_losses.append(loss + kld)
    test_losses.append(__test_epoch(vae_model, data_holder, device))
    return torch.stack(train_losses), torch.stack(test_losses), torch.stack(estimator_stds)


def __test_epoch(vae_model, data_holder, device):
    with eval_mode(vae_model):
        test_losses = []
        for x_batch, _ in data_holder.test:
            x_batch = x_batch.to(device)
            raw_params, x_preds = vae_model(x_batch)
            loss = __bce_loss(x_batch, x_preds, len(data_holder.dims)).mean()
            kld = vae_model.probabilistic.distribution.kl(raw_params).mean()
            test_losses.append(loss + kld)
        return torch.tensor(test_losses).mean()


def __generate_images(model, output_dir, data_holder, device):
    if not path.exists(output_dir):
        makedirs(output_dir)
    else:
        print(f"Skipping: '{output_dir}' already exists.")
        return

    with eval_mode(model):
        print(f'Generating images for in `{output_dir}`...')
        n = min(data_holder.batch_size, 8)
        for batch_id, (x_batch, _) in enumerate(data_holder.test):
            x_batch = x_batch[:n].to(device)
            _, x_pred_batch = model(x_batch)
            comparison = torch.cat((x_batch, x_pred_batch.view(x_batch.shape)))
            save_image(comparison, path.join(output_dir, f'recon_{batch_id}.png'), nrow=n)


def __bce_loss(x, x_recon, n_data_dims):
    # Use no reduction to get separate losses for each image
    binary_cross_entropy = torch.nn.BCELoss(reduction='none')
    return binary_cross_entropy(x_recon, x.expand_as(x_recon)).flatten(start_dim=-n_data_dims).sum(dim=-1)
