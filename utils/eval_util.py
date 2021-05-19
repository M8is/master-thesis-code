from contextlib import contextmanager

import torch


@contextmanager
def eval_mode(model):
    with torch.no_grad():
        was_training = model.training
        model.eval()
        try:
            yield
        finally:
            if was_training:
                model.train()
