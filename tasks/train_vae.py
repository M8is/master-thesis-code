from os import path, makedirs

import torch
import torch.utils.data

import models.vae
from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def train_vae(seed, results_dir, dataset, device, hidden_dim, param_dims, latent_dim, epochs, sample_size,
              learning_rate, mc_estimator, distribution, batch_size, **kwargs):
    if not path.exists(results_dir):
        makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    fix_random_seed(seed)
    data_holder = DataHolder.get(dataset, batch_size)

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
    encoder = models.vae.Encoder(data_holder.dims, hidden_dim, param_dims)
    decoder = models.vae.Decoder(data_holder.dims, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator).to(device)
    optimizer = torch.optim.Adam(vae_network.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')

    # Train
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = __train_epoch(vae_network, data_holder, device, optimizer)
        train_losses.add(train_loss)
        test_losses.add(test_loss)
        file_name = path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        train_losses.save()
        test_losses.save()
        torch.save(vae_network, file_name)
        print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy().mean():.2f}, Test loss: {test_losses.numpy().mean():.2f}",
              flush=True)
    train_losses.plot()
    test_losses.plot()


def __train_epoch(vae_model, data_holder, device, optimizer):
    train_losses = []
    test_losses = []
    vae_model.train()
    for batch_id, (x_batch, _) in enumerate(data_holder.train):
        x_batch = x_batch.view(-1, data_holder.dims).to(device)
        params, x_preds = vae_model(x_batch)
        losses = __bce_loss(x_batch, x_preds)
        optimizer.zero_grad()
        vae_model.backward(params, losses)
        kl = vae_model.probabilistic.distribution.kl(params)
        optimizer.step()
        train_losses.append((losses.detach().mean() + kl.detach().mean()))
    test_losses.append(__test_epoch(vae_model, data_holder, device))
    return torch.stack(train_losses), torch.stack(test_losses)


def __test_epoch(vae_model, data_holder, device):
    with torch.no_grad():
        set_previous_mode = vae_model.train if vae_model.training else vae_model.eval
        vae_model.eval()
        test_losses = []
        for x_batch, _ in data_holder.test:
            x_batch = x_batch.view(-1, data_holder.dims).to(device)
            params, x_preds = vae_model(x_batch)
            losses = __bce_loss(x_batch, x_preds) + vae_model.probabilistic.distribution.kl(params)
            test_losses.append(losses.detach().mean())
        set_previous_mode()
        return torch.tensor(test_losses).mean()


def get_estimator_stds(seed, dataset, device, latent_dim, sample_size, learning_rate, mc_estimator, distribution,
                       batch_size, hidden_dim, **_):
    print(f'Generating standard deviation for {mc_estimator}.')
    fix_random_seed(seed)
    data_holder = DataHolder.get(dataset, batch_size)
    estimator = get_estimator(mc_estimator, distribution, sample_size, device)
    encoder = models.vae.Encoder(data_holder.dims, hidden_dim, (latent_dim, latent_dim))
    decoder = models.vae.Decoder(data_holder.dims, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)
    optimizer = torch.optim.SGD(vae_network.parameters(), lr=learning_rate)

    result = []
    batch_size = data_holder.train.batch_size
    for x_batch, _ in data_holder.train:
        if x_batch.size(0) != batch_size:
            continue
        x_batch = x_batch.view(-1, data_holder.dims).to(device)
        grads = []
        for _ in range(sample_size):
            vae_network.train()
            params, x_preds = vae_network(x_batch)
            for p in params:
                p.retain_grad()
            losses = __bce_loss(x_batch, x_preds)
            optimizer.zero_grad()
            vae_network.backward(params, losses)
            for i, p in enumerate(params):
                grad = p.grad.mean().unsqueeze(0)
                try:
                    grads[i] = torch.cat((grads[i], grad), dim=0)
                except IndexError:
                    grads.append(grad)
        result.append(torch.stack(grads).std(dim=1))
    return torch.stack(result).mean(dim=0)


def __bce_loss(x, x_pred):
    # Use no reduction to get separate losses for each image
    binary_cross_entropy = torch.nn.BCELoss(reduction='none')
    return binary_cross_entropy(x_pred, x.expand_as(x_pred)).sum(dim=-1)
