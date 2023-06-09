# pylint: disable=invalid-name
"""BME loss functions"""
from itertools import combinations

import jax.numpy as jnp

from jax import jit, lax
from jax.scipy.special import logsumexp

from utils.tree import change_v, newick2v


def bme_tree_loss(dist_df, tree, taxa_list):
    """'Tree' version of the BME loss function

    Parameters
    ----------
    tree : ete3.Tree
        Phylogenetic tree
    dist_df : pandas.DataFrame
        Distance matrix
    taxa_list : list
        List of taxa names

    Returns
    -------
    log_loss : float
        BME loss
    """
    loss = sum(
        dist_df.loc[t_i, t_j]
        * (2 ** -tree.get_distance(str(t_i), str(t_j), topology_only=True))
        for (t_i, t_j) in combinations(taxa_list, 2)
    )

    log_loss = jnp.log(loss)

    return log_loss


@jit
def bme_loss(W, D, rooted):
    """'Vanilla' version of the BME loss function

    Parameters
    ----------
    W : jax.numpy.array
        Ordered tree probability matrix
    D : jax.numpy.array
        Distance matrix
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    float
        BME loss
    """
    E = get_edges_exp(W, rooted=rooted)

    return jnp.log((D * E).sum())


@jit
def bme_loss_log(W, D, rooted):
    """Log version of the BME loss function

    Parameters
    ----------
    W : jax.numpy.array
        Ordered tree probability matrix
    D : jax.numpy.array
        Distance matrix
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    float
        BME loss
    """
    E = get_edges_exp_log(W, rooted)

    return logsumexp(E, b=D)


@jit
def get_edges_exp(W, rooted):
    """Calculate the expectation of the objective value of a tree drawn with distribution W.

    We calculate and update E_ij hroughout the left-to-right construction procedure

    Parameters
    ----------
    W : jax.numpy.array
        Tree distribution
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    E : jax.numpy.array
        Expected objective value of a tree drawn with distribution W

        E_ij = Expected value of 2^(e_{ij}) with e the path length between nodes i and j in the tree
    """
    n_leaves = len(W) + 1

    # Triangular indices
    trindx_x, trindx_y = jnp.tril_indices(n_leaves - 1, -1)

    # Initialize E
    E = jnp.zeros((n_leaves, n_leaves))
    E = E.at[1, 0].set(0.5 * E[0, 0] * W[0, 0] + 0.25 * (2 - rooted) * W[0, 0])
    E = E + jnp.transpose(E)

    # Masks of W and E
    mask_E = jnp.ones_like(E)
    mask_W = jnp.ones_like(W)

    def body(carry, _):
        E, i = carry

        E_new = jnp.zeros((n_leaves, n_leaves))

        trindx_x_i = jnp.where(trindx_x < i, trindx_x, 1)
        trindx_y_i = jnp.where(trindx_x < i, trindx_y, 0)

        indx = (trindx_x_i, trindx_y_i)

        # Fill E_new
        E_new = E_new.at[indx].set(
            E[indx] * (1 - 0.5 * (W[i - 1, indx[1]] + W[i - 1, indx[0]]))
        )

        # Apply mask to E
        mask_Ei = jnp.where(jnp.arange(E.shape[0]) >= i, 0, mask_E)
        mask_Eii = jnp.where(jnp.arange(E.shape[1]) >= i, 0, mask_Ei.T)

        Eii = jnp.multiply(E, mask_Eii)

        # Apply mask to W
        mask_Wii = jnp.where(jnp.arange(W.shape[1]) >= i, 0, mask_W[i - 1])

        Wii = jnp.multiply(W[i - 1], mask_Wii)

        # Fill E_new
        tmp = 0.5 * jnp.sum(Eii[: n_leaves - 1, : n_leaves - 1] * Wii, 1) + 0.25 * Wii
        E_new = E_new.at[i, : n_leaves - 1].set(tmp)

        # Update E
        E = E_new + jnp.transpose(E_new)

        return (E, i + 1), None

    # https://github.com/google/jax/issues/5454
    (E, _), _ = lax.scan(body, (E, 2), None, length=n_leaves - 2)

    return E


@jit
def get_edges_exp_log(W, rooted):
    """Log version of get_edges_exp

    Parameters
    ----------
    W : jax.numpy.array
        Tree distribution
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    E : jax.numpy.array
        Log of the expected objective value of a tree drawn with distribution W
    """
    # Add jnp.finfo(float).eps to W.tmp to avoid floating point errors with float32
    W_tmp = (
        jnp.pad(W, (0, 1), constant_values=jnp.finfo(float).eps) + jnp.finfo(float).eps
    )

    n_leaves = len(W) + 1

    E = jnp.zeros((n_leaves, n_leaves))

    trindx_x, trindx_y = jnp.tril_indices(n_leaves - 1, -1)

    E = E.at[1, 0].set(
        0.5 * E[0, 0] * W_tmp[0, 0] + jnp.log(0.25 * (2 - rooted) * W_tmp[0, 0])
    )

    E = E + E.T

    def body(carry, _):
        E, i = carry

        E_new = jnp.zeros((n_leaves, n_leaves))

        ##### j for loop #####
        trindx_x_i = jnp.where(trindx_x < i, trindx_x, 1)
        trindx_y_i = jnp.where(trindx_x < i, trindx_y, 0)

        indx = (trindx_x_i, trindx_y_i)

        E_new = E_new.at[indx].set(
            E[indx]
            + jnp.log(
                1
                - 0.5 * (W_tmp[i - 1, indx[1]] + W_tmp[i - 1, indx[0]])
                + jnp.finfo(float).eps
            )
        )
        ##### End j for loop #####

        ##### k for loop #####
        # exp array
        mask_Ei = jnp.where(jnp.arange(n_leaves) >= i, 0, 1)
        exp_array = E * jnp.where(jnp.arange(n_leaves) >= i, 0, mask_Ei.T)

        # coef array
        mask_Wi = jnp.where(jnp.arange(n_leaves) >= i, 0, 0.5 * W_tmp[i - 1])
        coef_array = (jnp.zeros_like(W_tmp) + mask_Wi).at[:, i].set(
            0.25 * W_tmp[i - 1]
        ) * (1 - jnp.eye(W_tmp.shape[0]))

        # logsumexp
        tmp = logsumexp(exp_array, b=coef_array, axis=-1) * mask_Ei

        E_new = E_new.at[i, :].set(tmp)

        # Update E
        E = E_new + E_new.T

        ##### End k for loop #####

        return (E, i + 1), None

    # https://github.com/google/jax/issues/5454
    (E, _), _ = lax.scan(body, (E, 2), None, length=n_leaves - 2)

    return E


def check_losses(dist_df, tree, names, v_num, int2taxa, taxa2int, method, rooted):
    """Check that BME losses are equivalent (using the distance matrix and the tree)

    Parameters
    ----------
    dist_df : pandas.DataFrame
        Distance matrix
    tree : ete3.Tree
        Tree
    names : list
        List of taxa names
    v_num : int
        Number of leaves/taxa
    int2taxa : dict[int, str]
        An integer mapping of taxa
    taxa2int : dict[str, int]
        The reverse mapping of int2taxa
    method : str
        The name of the method that yielded the tree
    rooted : bool
        If True, indicates that the tree is/should be rooted

    Returns
    -------
    tmp_tree : ete3.Tree
        The tree, rerooted at the best node
    tmp_int2taxa : dict[int, str]
        The new integer mapping (changed if tmp_tree != tree)
    tmp_names : list
        The new list of taxa names
    D : jax.numpy.array
        The reordered distance matrix (using tmp names), as an array
    """
    tmp_tree, tmp_tree_mid = tree.copy(), tree.copy()

    tmp_tree_mid.set_outgroup(tmp_tree_mid.get_midpoint_outgroup())

    midpoint_loss = bme_tree_loss(dist_df, tmp_tree_mid, names)

    best_loss = midpoint_loss

    losses = [best_loss]

    root = ""

    for node in tmp_tree.traverse("postorder"):
        if not node.is_root():
            tmp_tree.set_outgroup(node.name)

            score = bme_tree_loss(dist_df, tmp_tree, names)

            losses.append(score)

            if score < best_loss:
                root = node.name
                best_loss = score

    if min(losses) == midpoint_loss:
        print("root at midpoint")
        tmp_tree = tmp_tree_mid.copy()
    else:
        print(f"root at {root}")
        tmp_tree.set_outgroup(root)

    print(f"with score {best_loss}")

    tr = tmp_tree.copy()

    for node in tr.traverse("postorder"):
        if node.is_leaf():
            node.name = taxa2int[node.name]

    v = newick2v(tr.write(format=1))[1:]

    v_tmp, tmp_int2taxa = change_v("birth_death", v, int2taxa)

    tmp_names = [v for (_, v) in sorted(tmp_int2taxa.items())]

    D = jnp.asarray(dist_df.loc[tmp_names, tmp_names])

    W_discrete = jnp.eye(v_num - 1)[v_tmp]

    sc = bme_loss(W_discrete, D, rooted)

    eval_tree = tmp_tree.copy()

    if not rooted:
        eval_tree.unroot()

    sctmp = bme_tree_loss(dist_df, eval_tree, tmp_names)

    print("Checking our and the discrete BME objective match:")
    print(f"newick2v+reorder score: {sc}")
    print(f"{method} score: {sctmp}")
    print(f"diff: {jnp.abs(sctmp - sc)}")

    return tmp_tree, tmp_int2taxa, tmp_names, D
