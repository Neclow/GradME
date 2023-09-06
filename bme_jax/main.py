# pylint: disable=invalid-name

"""Main script."""
import os

from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import pandas as pd

from ete3 import Tree

from bme_jax.data import load_data
from bme_jax.loss import bme_loss_log, bme_tree_loss, check_losses
from bme_jax.optim import get_optimizer, objective_log, optimize, step
from core import Phylo2Vec
from utils.data import load_yaml_config


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def parse_args():
    """Parse main arguments."""
    parser = ArgumentParser(description="GradME main arguments")

    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--fasta-path",
        type=str,
        default=None,
        help="Path of FASTA file to analyse (will override fasta_path in the .yml config file)",
    )

    return parser.parse_args()


def run(cfg):
    """Perform a run with GradME

    Parameters
    ----------
    cfg : types.SimpleNamespace
        Configuration parameters, loaded from a .yml file

    Returns
    -------
    best_params : dict
        Parameter setting that gave the best results and all optimisation losses
    summary : pd.Series
        Summary of the run. Contains:
            Configuration parameters
            Optimisation statistics
            Robinson-Foulds distances between the FastME/NJ/GradME trees
    """
    # Load data
    n_sites, D, names, fastme_newick, nj_newick = load_data(cfg)

    print(f"{n_sites} sites; {D.shape[1]} taxa.")

    # Integer-taxon mapping
    int2taxa = dict(enumerate(names))
    taxa2int = {v: k for (k, v) in int2taxa.items()}

    # Distance matrix
    dist_df = pd.DataFrame(D, index=names, columns=names)

    v_num = dist_df.shape[0]

    # Load FastME and NJ trees and rename their leaves
    fastme_tree = Tree(fastme_newick)
    nj_tree = Tree(nj_newick)

    for tree in [fastme_tree, nj_tree]:
        i = 0
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                node.name = str(i)
                i += 1

    # Check that both BME losses are consistent
    fastme_tree, _, _, fastme_D = check_losses(
        dist_df, fastme_tree, names, v_num, int2taxa, taxa2int, "fastme", cfg.rooted
    )

    nj_tree, _, _, nj_D = check_losses(
        dist_df, nj_tree, names, v_num, int2taxa, taxa2int, "nj", cfg.rooted
    )

    # Function to evaluate objective_log and its gradient
    value_and_grad_fun = jax.jit(jax.value_and_grad(objective_log, argnums=0))

    # Get optimiser
    optimizer = get_optimizer(cfg.optimizer, cfg.learning_rate)

    # Neighbour joining step
    W_end = step(cfg, nj_D, v_num, optimizer, value_and_grad_fun)
    W_discrete = jnp.eye(W_end.shape[0])[W_end.argmax(1)]
    sc_nj = bme_loss_log(W_discrete, nj_D, cfg.rooted)

    # FastME step
    W_end = step(cfg, fastme_D, v_num, optimizer, value_and_grad_fun)
    W_discrete = jnp.eye(W_end.shape[0])[W_end.argmax(1)]
    sc_fastme = bme_loss_log(W_discrete, fastme_D, cfg.rooted)

    # GradME Optimisation
    best_params, best_score = optimize(cfg, D, v_num, int2taxa, dist_df)

    # Load the best parameters in Phylo2Vec and build a tree
    phylo2vec = Phylo2Vec(v=best_params["W"].argmax(1), rooted=cfg.rooted)

    phylo2vec.taxa_dict = best_params["int2taxa"]

    opt_tree = phylo2vec.label_tree()

    opt_tree.ladderize()

    # Unroot trees for comparison
    if not cfg.rooted:
        nj_tree.unroot()
        fastme_tree.unroot()
        opt_tree.unroot()

    # Aggregate summary statistics
    # fmt: off
    summary = {
        # Configuration
        "substitution_model": cfg.substitution_model,
        "optimizer": cfg.optimizer,
        "learning_rate": cfg.learning_rate,
        # BME tree losses
        "tree_loss_nj": bme_tree_loss(dist_df, nj_tree, names),
        "tree_loss_fastme": bme_tree_loss(dist_df, fastme_tree, names),
        # BME scores
        "step_loss_nj": sc_nj,
        "step_loss_fastme": sc_fastme,
        "step_loss_opt_random": best_score,
        # Tree distances
        "rf_nj_opt": opt_tree.robinson_foulds(nj_tree, unrooted_trees=True)[0],
        "rf_fastme_opt": opt_tree.robinson_foulds(fastme_tree, unrooted_trees=True)[0],
        "rf_nj_fastme": nj_tree.robinson_foulds(fastme_tree, unrooted_trees=True)[0]
    }
    # fmt: on

    return best_params, pd.Series(summary)


if __name__ == "__main__":
    args = parse_args()

    cfg_ = load_yaml_config(args.config_path)

    # If fasta_path is provided in the args, overwrite the cfg argument
    if args.fasta_path is not None:
        cfg_.fasta_path = args.fasta_path

    run(cfg_)
