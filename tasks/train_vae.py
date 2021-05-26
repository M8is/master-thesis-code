from os import path, makedirs

import torch
import torch.utils.data
from torchvision.utils import save_image

import models.vae
from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.loss_holder import LossHolder
from utils.eval_util import eval_mode
from utils.seeds import fix_random_seed


def train_vae(seed, results_dir, dataset, device, hidden_dim, param_dims, latent_dim, epochs, sample_size,
              learning_rate, mc_estimator, distribution, batch_size, **kwargs):
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)

    if path.exists(results_dir):
        print(f"Skipping training: '{results_dir}' already exists")
    else:
        makedirs(results_dir)
        fix_random_seed(seed)
        data_holder = DataHolder.get(dataset, batch_size)

        # Create model
        estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
        encoder = models.vae.Encoder(data_holder.dims, hidden_dim, param_dims)
        decoder = models.vae.Decoder(data_holder.dims, hidden_dim, latent_dim)
        vae_network = models.vae.VAE(encoder, decoder, estimator).to(device)
        optimizer = torch.optim.Adam(vae_network.parameters(), lr=learning_rate)

        print(f'Training with {estimator}.')
        for epoch in range(1, epochs + 1):
            train_loss, test_loss = __train_epoch(vae_network, data_holder, device, optimizer)
            train_losses.add(train_loss)
            test_losses.add(test_loss)
            file_name = path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
            train_losses.save()
            test_losses.save()
            torch.save(vae_network, file_name)
            __generate_images(vae_network, path.join(results_dir, f'images_{epoch}'), data_holder, device)
            print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy()[-1].mean():.2f}, "
                  f"Test loss: {test_losses.numpy()[-1].mean():.2f}",
                  flush=True)

    return train_losses, test_losses


def __train_epoch(vae_model, data_holder, device, optimizer):
    train_losses = []
    test_losses = []
    vae_model.train()
    print(50 * "-")
    for batch_id, (x_batch, _) in enumerate(data_holder.train):
        x_batch = x_batch.view(-1, data_holder.dims).to(device)
        raw_params, x_recon = vae_model(x_batch)
        kld = vae_model.probabilistic.distribution.kl(raw_params).mean()
        loss = __bce_loss(x_batch, x_recon).mean()
        optimizer.zero_grad()
        kld.backward(retain_graph=True)
        vae_model.probabilistic.backward(raw_params, lambda samples: __bce_loss(x_batch, vae_model.decoder(samples)))
        loss.backward()
        optimizer.step()
        print(f"\r| ELBO: {loss + kld:.2f} | BCE loss: {loss:.1f} | KL Divergence: {kld:.1f} | ", end='', flush=True)
        train_losses.append(loss + kld)
    print(50 * " " + "\r", end='')
    test_losses.append(__test_epoch(vae_model, data_holder, device))
    return torch.stack(train_losses), torch.stack(test_losses)


def __test_epoch(vae_model, data_holder, device):
    with eval_mode(vae_model):
        test_losses = []
        for x_batch, _ in data_holder.test:
            x_batch = x_batch.view(-1, data_holder.dims).to(device)
            raw_params, x_preds = vae_model(x_batch)
            loss = __bce_loss(x_batch, x_preds).mean()
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
            x_batch_flat = x_batch.view(-1, data_holder.dims)
            _, x_pred_batch = model(x_batch_flat)
            comparison = torch.cat((x_batch, x_pred_batch.view(x_batch.shape)))
            save_image(comparison, path.join(output_dir, f'recon_{batch_id}.png'), nrow=n)


def __bce_loss(x, x_recon):
    # Use no reduction to get separate losses for each image
    binary_cross_entropy = torch.nn.BCELoss(reduction='none')
    return binary_cross_entropy(x_recon, x.expand_as(x_recon)).sum(dim=-1)


# TODO: add option to run this before/after each iteration/epoch
def __get_estimator_stds(model, optimizer, data_holder, device):
    n_estimates = 100

    result = []
    batch_size = data_holder.train.batch_size
    for x_batch, _ in data_holder.test:
        if x_batch.size(0) != batch_size:
            continue
        x_batch = x_batch.view(-1, data_holder.dims).to(device)
        grads = []
        for _ in range(n_estimates):
            model.train()
            params, (decoder_x_preds, encoder_x_preds) = model(x_batch)
            for p in params:
                p.retain_grad()
            decoder_losses = __bce_loss(x_batch, decoder_x_preds)
            encoder_losses = __bce_loss(x_batch, encoder_x_preds)
            optimizer.zero_grad()
            model.backward(params, decoder_losses, encoder_losses)
            for i, p in enumerate(params):
                grad = p.grad.mean().unsqueeze(0)
                try:
                    grads[i] = torch.cat((grads[i], grad), dim=0)
                except IndexError:
                    grads.append(grad)
        result.append(torch.stack(grads).std(dim=1))
    return torch.stack(result).mean(dim=0)
