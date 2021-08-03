from typing import Dict, Any

from tasks.train_log_reg import TrainLogReg
from tasks.train_polynomial import TrainPolynomial
from tasks.train_vae import TrainVAE
from tasks.trainer import StochasticTrainer

__TASKS = {
    'vae': TrainVAE,
    'logreg': TrainLogReg,
    'polynomial': TrainPolynomial
}


def get_trainer(task_tag: str, config: Dict[str, Any]) -> StochasticTrainer:
    if task_tag not in __TASKS:
        raise ValueError(f"Unknown task '{task_tag}'.")
    return __TASKS[task_tag.lower()](**config)
