import os

import torch
from torchvision.utils import save_image

from utils.data_holder import DataHolder


def generate_images_vae(results_dir, dataset, epochs, mc_estimator, batch_size, device, **_):
    for epoch in range(1, epochs + 1):
        model_file_path = os.path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        if not os.path.exists(model_file_path):
            continue  # Skip silently if model does not exist
        model = torch.load(model_file_path).eval()

        out_dir = os.path.join(results_dir, f'images_{epoch}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            print(f"Skipping: '{out_dir}' already exists.")
            continue

        data_holder = DataHolder.get(dataset, batch_size)

        print(f'Generating images for `{model_file_path}` in `{out_dir}`...')
        for batch_id, (x_batch, _) in enumerate(data_holder.test):
            n = min(x_batch.size(0), 8)
            x_batch = x_batch[:n].view(-1, data_holder.dims).to(device)
            _, x_pred_batch = model(x_batch)
            comparison = torch.cat((x_batch, x_pred_batch.view(x_batch.shape)))
            save_image(comparison, os.path.join(out_dir, f'recon_{batch_id}.png'), nrow=n)
