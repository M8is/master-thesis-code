import shutil
from os import path


def clean(results_dir, **__):
    if path.exists(results_dir):
        shutil.rmtree(results_dir)
        print(f"Cleaned '{results_dir}'.")
