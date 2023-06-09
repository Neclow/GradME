# pylint: disable=invalid-name
"""BME continuous optimisation"""
import math

import jax
import jax.numpy as jnp
import optax

from jax import jit, value_and_grad
from tqdm import tqdm

from bme_jax.loss import bme_loss_log
from utils.tree import change_v, reroot_at_random


def optimize(cfg, D, v_num, int2taxa, dist_df):
    """Optimisation of the continuous BME function

    Parameters
    ----------
    cfg : types.SimpleNamespace
        Configuration parameters, loaded from a .yml file
    D : jax.numpy.ndarray
        Distance matrix (with ordered columns)
    v_num : int
        Number of leaves/taxa
    int2taxa : dict[int, str]
        An integer mapping of taxa
    dist_df : pandas.DataFrame
        Initial distance matrix

    Returns
    -------
    best_params : dict
        Parameter setting that gave the best results

    best_score : float
        Best optimisation score
    """
    # Function to evaluate objective_log and its gradient
    value_and_grad_fun = jit(value_and_grad(objective_log, argnums=0))

    # Get optimiser
    optimizer = get_optimizer(cfg.optimizer, lr=cfg.learning_rate)

    # Initial "best" score, set as an arbitrarily high values
    best_score = 1e8

    # The taxon ordering in the distance matrix, initially set as
    tmp_names = None

    # Initialise the best parameters
    best_params = {"D": D, "int2taxa": int2taxa, "score": best_score}

    # Progress bar
    pbar = tqdm(range(cfg.n_shuffles))

    # List of scores (BME losses) obtained during the optimisation
    scores = []

    for _ in pbar:
        # Perform a step
        W = step(cfg, D, v_num, optimizer, value_and_grad_fun)

        # Discretise and cast to a matrix
        v = W.argmax(1)

        W_discrete = jnp.eye(W.shape[0])[v]

        # Get the log-loss
        score = bme_loss_log(W_discrete, D, cfg.rooted)

        # Update the best score and params if we get a better score
        if score < best_score:
            best_score = score
            best_params.update(
                {
                    "D": D.copy(),
                    "int2taxa": int2taxa.copy(),
                    # if tmp_names is None
                    # else dict(enumerate(tmp_names)),
                    "score": best_score,
                    "W": W.copy(),
                }
            )

        # Reroot at random if we use unrooted trees
        v_change = v if cfg.rooted else reroot_at_random(v)

        # Queue Shuffle
        _, int2taxa = change_v("birth_death", v_change, int2taxa, shuffle_cols=True)

        # Gather the new taxon ordering and re-arrange the distance matrix
        tmp_names = [taxon for (_, taxon) in sorted(int2taxa.items())]

        D = jnp.asarray(dist_df.loc[tmp_names, tmp_names])

        # Add the current best score and update the progress bar
        scores.append(best_score)

        pbar.set_postfix({"\033[95m current best BME objective is ": best_score})

    best_params["scores"] = scores

    return best_params, best_score


def step(cfg, D, v_num, optimizer, value_and_grad_fun):
    """A single optimisation step of the continuous BME function

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration parameters, loaded from a .yml file
    D : jax.numpy.ndarray
        Distance matrix (with ordered columns)
    v_num : int
        Number of leaves/taxa
    optimizer : optax._src.base.GradientTransformation
        Optax optimizer
    value_and_grad_fun : Callable
        A function that calculates the BME objective value and its gradient

    Returns
    -------
    W : jax.numpy.ndarray
        Predicted tree distribution
    """
    params = jnp.ones(round(0.5 * v_num * (v_num - 1))) * 0.5

    state = optimizer.init(params)
    # state = optimizer.init_state(params, data=D, rooted=cfg.rooted)

    prev_loss = 1e8

    # TODO: lax.scan?
    for _ in range(cfg.n_steps):
        # loss, _ = optimizer.value_and_grad_fun(params, D=D, cfg=cfg.rooted)
        loss, gradients = value_and_grad_fun(params, D, cfg.rooted)

        if jnp.abs(loss - prev_loss) < cfg.tol:
            break
        else:
            prev_loss = loss

        # params, state = optimizer.update(params, state, data=D, rooted=cfg.rooted)
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)

    W = make_W(params)

    return W


@jit
def objective_log(params, D, rooted):
    """Log version of objective

    Parameters
    ----------
    params : jax.numpy.ndarray
        Flattened version of W
    D : jax.numpy.ndarray
        Distance matrix (with ordered columns)
    rooted : bool
        If True, indicates that the tree is/should be rooted

    Returns
    -------
    obj : float
        BME bjective value
    """
    W = make_W(params)

    obj = bme_loss_log(W, D, rooted)
    return obj


@jit
def make_W(params, eps=1e-8):
    """'Un-flatten' params to a W matrix representing the distribution of ordered trees

    Parameters
    ----------
    params : jax.numpy.ndarray
        Flattened version of W
    eps : float, optional
        Term added to improve numerical stability, by default 1e-8

    Returns
    -------
    W : jax.numpy.ndarray
        distribution of ordered trees
    """
    # Solution of quadratic equation: k^2 - k - 2*len(params)
    k = int((1 + math.sqrt(1 + 8 * len(params))) // 2) - 1

    W = jnp.zeros((k, k)).at[jnp.tril_indices(k)].set(jax.nn.softplus(params))

    return W / (jnp.tril(W).sum(1)[:, jnp.newaxis] + eps)


def get_optimizer(opt, lr):
    """Get an Optax optimizer from a string

    Parameters
    ----------
    opt : str
        Name of the optax optimizer
    lr : float
        Learning rate

    Returns
    -------
    optimizer : optax._src.base.GradientTransformation
        Optax optimizer
    """
    optimizer = getattr(optax, opt)

    return optimizer(learning_rate=lr)
