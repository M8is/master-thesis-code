import torch


# TODO: add option to run this before/after each iteration/epoch
def get_estimator_stds(model, optimizer, data_holder, device, loss_fn):
    n_estimates = 100

    result = []
    batch_size = data_holder.train.batch_size
    for x_batch, _ in data_holder.train:
        if x_batch.size(0) != batch_size:
            continue
        x_batch = x_batch.view(-1, data_holder.dims).to(device)
        grads = []
        for _ in range(n_estimates):
            model.train()
            params, x_preds = model(x_batch)
            for p in params:
                p.retain_grad()
            losses = loss_fn(x_batch, x_preds)
            optimizer.zero_grad()
            model.backward(params, losses)
            for i, p in enumerate(params):
                grad = p.grad.mean().unsqueeze(0)
                try:
                    grads[i] = torch.cat((grads[i], grad), dim=0)
                except IndexError:
                    grads.append(grad)
        result.append(torch.stack(grads).std(dim=1))
    return torch.stack(result).mean(dim=0)
