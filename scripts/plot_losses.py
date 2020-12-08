import matplotlib.pyplot as plt
import argparse
from train.utils import LossHolder

import sys
import train.vae


def main(loss_file_paths):
    for loss_file_path in loss_file_paths:
        try:
            losses = LossHolder.load(loss_file_path)
        except FileNotFoundError:
            print(f"Could not find file '{loss_file_path}'.")
            continue

        train_losses, test_losses = losses.as_numpy()
        print(f"File '{loss_file_path}' train: {train_losses.shape}")
        print(f"File '{loss_file_path}' test: {test_losses.shape}")

        plt.title(loss_file_path + ' train')
        plt.plot(train_losses.flatten())
        plt.show()
        plt.title(loss_file_path + ' test')
        plt.plot(test_losses.flatten())
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loss Plotting Util')
    parser.add_argument('LOSSES', help='Path to pickled `LossHolder`. May be multiple files.', nargs='+')
    args = parser.parse_args()
    main(args.LOSSES)
