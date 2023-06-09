"""Run optimisation on different datasets and optimisation configurations."""
# pylint: disable=invalid-name, redefined-outer-name
import math
import os

from itertools import product
from pathlib import Path
from pprint import pprint

import jax.numpy as jnp

import pandas as pd

from bme_jax.main import run
from utils.data import load_yaml_config

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def run_test(cfg, params, dataset_folder):
    """_summary_

    Parameters
    ----------
    cfg : _type_
        _description_
    params : _type_
        _description_
    dataset_folder : _type_
        _description_
    """
    # Update the configuration according to params
    cfg.substitution_model, cfg.optimizer, cfg.learning_rate = params

    # Prepare the result file
    result_file = Path(
        dataset_folder,
        (
            f"all_scores_{Path(cfg.fasta_path).stem}_"
            f"s-{cfg.substitution_model}_"
            f"o-{cfg.optimizer}_"
            f"lr-{cfg.learning_rate}"
            ".npy"
        ),
    )

    print(f"Result file: {result_file}")

    if os.path.isfile(result_file):
        return

    pprint(cfg.__dict__)

    best_params, summary = run(cfg)

    # Save run
    summary_file = Path(dataset_folder, "summary.csv")

    if os.path.isfile(summary_file):
        df = pd.read_csv(summary_file, index_col=0).append(summary, ignore_index=True)
    else:
        df = pd.Series(summary).to_frame().T

    df.to_csv(summary_file)

    jnp.save(
        result_file,
        best_params["scores"],
    )


if __name__ == "__main__":
    cfg = load_yaml_config("cfg/bme_config_v3.yml", quiet=True)

    datasets = [
        # "primates.fa",
        # "yeast.fa",
        # "h3n2_na_20.fa",
        "DS1.fa",
        "DS2.fa",
        "DS3.fa",
        "DS4.fa",
        "DS5.fa",
        "DS6.fa",
        "DS7.fa",
        "DS8.fa",
        "DS9.fa",
        "DS10.fa",
        "DS11.fa",
        # "song2012_merged.fa",
    ]

    param_grid = {
        "substitution_model": ["JC69", "F81", "TN93"],
        "optimizer": ["adafactor", "adamw", "rmsprop", "sgd"],
        "learning_rate": [0.001, 0.01, 0.1, 1.0],
    }

    n_tests = math.prod([len(_) for _ in param_grid.values()])

    for dataset in datasets:
        dataset_folder = Path("results", Path(dataset).stem)
        os.makedirs(dataset_folder, exist_ok=True)

        cfg.fasta_path = dataset

        for _, params in enumerate(product(*param_grid.values())):
            print(f"Dataset: {dataset}; Search {_}/{n_tests}")
            run_test(cfg, params, dataset_folder)
