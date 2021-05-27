from matplotlib import pyplot as plt
from os import path, makedirs


def plot_losses(out_dir, train_losses, test_losses, logscale):
    print(f"Plotting losses in {out_dir} ...")
    for task, train_losses in train_losses.items():
        __plot(train_losses, path.join(out_dir, task, f'train.png'), logscale, linewidth=.25)
    for task, test_losses in test_losses.items():
        __plot(test_losses, path.join(out_dir, task,  f'test.png'), logscale, linewidth=1.)


def __plot(losses, file_name, logscale, **plot_kwargs):
    if logscale:
        plt.yscale('log')
    for config, loss in losses:
        plt.plot(loss.numpy(), label=config.get('plot_label', config['mc_estimator']), **plot_kwargs)

    plt.legend()
    if not path.exists(path.dirname(file_name)):
        makedirs(path.dirname(file_name))
    plt.savefig(file_name)
    plt.clf()
