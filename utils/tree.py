"""Utility functions for tree manipulation."""
# pylint: disable=invalid-name, line-too-long
import random
import re

import numpy as np

from ete3 import Tree

POS_INT_PATTERN = re.compile(r"\b\d+\b")


def label_tree(
    v, with_branch_lengths=False, rooted=True, ete3_format=8, taxa_dict=None
):
    """Label a tree from Phylo2Vec

    Parameters
    ----------
    v : numpy.ndarray or list
        Current Phylo2vec vector
    with_branch_lengths : bool, optional
        If True, add "dummy" branch lengths, by default False
        cf hc.utils.tree._get_dummy_branch_lengths
    rooted : bool, optional
        If False, unroot the tree, by default True
    ete3_format : int, optional
        Newick string format in ete3, by default 8
    taxa_dict : dict[int, str], optional
        Pass it to label the tree with the actual taxa names, by default None

    Returns
    -------
    M : numpy.ndarray
        1st column: parent
        2nd and 3rd column: children
    str
        Tree written in Newick format
    tree : ete3.Tree
        Tree representing the node ancestry (M) with branch_lengths
    """
    # Get ancestry
    M = _get_ancestry(v)

    # Build tree
    if with_branch_lengths:
        # Get dummy branch lengths
        branch_lengths = _get_dummy_branch_lengths(M)
        tree = _build_tree_with_branch_lengths(M, branch_lengths)
    else:
        # try:
        tree = _build_tree(M)  # (M.astype(str))
        # except Exception as e:
        #     raise ValueError('M should be astype str?') from e

    if not rooted:
        tree.unroot()

    if taxa_dict is not None:
        for leaf in tree.iter_leaves():
            leaf.name = taxa_dict[leaf.name]

    return M, tree.write(format=ete3_format), tree


def _get_ancestry(v):
    """Get pairs and ancestry for each node given a v-representation.

    Parameters
    ----------
    v : numpy.ndarray
        vector representation of a tree

    Returns
    -------
    M : numpy.ndarray
        1st column: parent
        2nd and 3rd column: children
    """
    k = len(v)

    assert np.all(v <= 2 * np.arange(k)), print(v, v <= 2 * np.arange(k))

    labels = np.tril([np.arange(k + 1)] * (k + 1))

    # Which nodes have not been processed?
    not_processed = np.ones(k, dtype="bool")

    M = np.zeros((k, 3), dtype=np.int32)

    for _ in range(k):
        cond = (v <= labels[:k, :].max(1)) & (not_processed)
        n = len(cond) - cond[::-1].argmax() - 1

        m = np.argmax(labels[n, :] == v[n])

        M[_, 0] = labels[-1, m]
        M[_, 1] = labels[-1, n + 1]

        labels[n:, m] = labels[n:, :].max(1) + 1

        M[_, 2] = labels[-1, m]

        not_processed[n] = False

    return np.flip(M)


# FIXME: numba crashes for long M arrays
# @nb.njit
def _build_newick(M):
    """Build a Newick rep of a tree from an "ancestry" array

    The input M should always be 3-dimensional with the following format:
    1st column: parent node
    2nd column: children 1
    3rd column: children 2

    M is processed such that we iteratively write a Newick string
    to describe the tree.

    Parameters
    ----------
    M : numpy.ndarray
        "Ancestry" array of size (n_leaves - 1, 3)

    Returns
    -------
    newick : str
        Newick representation of M (topology only)
    """

    # TODO: add flag to add internal nodes to newick or not (like format 8 or 9)
    # List of parent nodes
    parent_nodes = []

    # List of sub-Newicks
    sub_newicks = []

    for i in range(len(M) - 1, -1, -1):
        parent, child1, child2 = M[i, :]
        # Case 1: Both children are parent nodes, so we have sub-newicks for them
        if child1 in parent_nodes and child2 in parent_nodes:
            # Find their indices
            idx1 = parent_nodes.index(child1)
            idx2 = parent_nodes.index(child2)

            # Merge the sub-newicks and add the parent node
            sub_newicks[idx1] = f"({sub_newicks[idx1]},{sub_newicks[idx2]}){parent}"

            # Update the parent node for the 1st children
            parent_nodes[idx1] = parent

            # Discard info on 2nd children as merged with the 1st children
            sub_newicks.remove(sub_newicks[idx2])
            parent_nodes.remove(parent_nodes[idx2])

        # Case 2: only the first child is a parent node
        elif child1 in parent_nodes:
            # Find its index
            idx = parent_nodes.index(child1)

            # Update its sub-Newick:
            # (sub_child1.1, sub_child1.2)child_1 --> ((sub_child1.1, sub_child1.2)child_1, child_2)parent
            sub_newicks[idx] = "(" + sub_newicks[idx].replace(
                f"{child1}", f"{child1},{child2}){parent}"
            )

            # Update the parent node (first child is now just an internal node)
            parent_nodes[idx] = parent

        # Case 3: only the second child is a parent node (similar to Case 2)
        elif child2 in parent_nodes:
            idx = parent_nodes.index(child2)
            # (sub_child2.1, sub_child2.2)child_2 --> ((sub_child2.1, sub_child2.2)child_2, child_2)parent
            sub_newicks[idx] = "(" + sub_newicks[idx].replace(
                f"{child2}", f"{child2},{child1}){parent}"
            )
            parent_nodes[idx] = parent

        # Case 4: the children nodes have not been added yet
        else:
            # Add a new sub-Newick for this triplet
            sub_newicks.append(f"({child1},{child2}){parent}")

            # Append the parent node
            parent_nodes.append(parent)

    # If everything went well, only one "sub-newick" should be left, with only one parent: the root node
    newick = sub_newicks[0] + ";"

    return newick


def _build_tree(M):
    """Build a tree from an "ancestry" array

    The input M should always be 3-dimensional with the following format:
    1st column: parent node
    2nd column: children 1
    3rd column: children 2

    M is processed such that we iteratively write a Newick string
    to describe the tree.

    Parameters
    ----------
    M : numpy.ndarray
        "Ancestry" array of size (n_leaves - 1, 3)

    Returns
    -------
    ete3.TreeNode
        ete3 tree object built from a Newick string
    """
    newick = _build_newick(M)

    # Convert to ete3 (for legacy reasons)
    return Tree(newick, format=8)


def _build_tree_with_branch_lengths(M, branch_lengths):
    """Build a tree with branch lengths

    Parameters
    ----------
    M : numpy.ndarray
        1st column: parent
        2nd and 3rd column: children
    branch_lengths : dict
        key: "parent->child"
        Length of each tree edge

    Returns
    -------
    t : ete3.Tree
        Tree representing the node ancestry (M) with branch_lengths
    """
    t = Tree(name=M[0, 0])
    t.add_child(name=M[0, 1], dist=branch_lengths[f"{M[0, 0]}->{M[0, 1]}"])
    t.add_child(name=M[0, 2], dist=branch_lengths[f"{M[0, 0]}->{M[0, 2]}"])

    for i in range(1, M.shape[0]):
        current_node = t.search_nodes(name=M[i, 0])[0]

        current_node.add_child(
            name=M[i, 1], dist=branch_lengths[f"{M[i, 0]}->{M[i, 1]}"]
        )
        current_node.add_child(
            name=M[i, 2], dist=branch_lengths[f"{M[i, 0]}->{M[i, 2]}"]
        )

    return t


def _get_dummy_branch_lengths(M):
    """Assign dummy branch lengths given an ancestry

    The branch lengths from a parent to a child depends on how "deep" the parent is in the tree
    The root node has depth 0, and all leaf nodes have depth len(M).
    Depth of node N = len(M) - number of connections from root node to N

    Parameters
    ----------
    M : numpy.ndarray
        1st column: parent
        2nd and 3rd column: children

    Returns
    -------
    branch_lengths : dict
        key: "parent->child"
        Length of each tree edge
    """
    branch_lengths = {}
    node_depth = {}
    k = M.shape[0]

    for i in range(k):
        parent = M[i, 0]
        children = M[i, [1, 2]]

        for child in children:
            if child <= k:
                branch_lengths[f"{parent}->{child}"] = (
                    k - node_depth.get(parent, i)
                ) / k
            else:
                branch_lengths[f"{parent}->{child}"] = 1 / k
                node_depth[child] = node_depth.get(parent, i) + 1

    return branch_lengths


def newick2v(nw, n_leaves=None, ete3_format=1, reroot=False):
    """Convert a newick-format tree to its v representation

    Parameters
    ----------
    nw : str
        Newick representation of a tree
    n_leaves : int, optional
        Number of leaves, by default None
        (Saves some computation time if fed in advance)
    ete3_format : int, optional
        Newick string format in ete3, by default 1
    reroot : bool, optional
        If true, "re-root" the Newick string, by default False

    Returns
    -------
    v: numpy.ndarray
        v representation of nw
    """
    if ete3_format != 9 or reroot:
        # Rewrite to Newick format 9 (only leaf nodes)
        t = Tree(nw, format=ete3_format)
        if reroot:
            t.set_outgroup(t.get_midpoint_outgroup())
        nw = t.write(format=9)

        n_leaves = len(t.get_leaves())
    else:
        if n_leaves is None:
            # INFO: fastest when n_leaves is used, this is just in case
            # TODO: n_leaves as non-optional argument?
            n_leaves = max(int(s) for s in re.findall(POS_INT_PATTERN, nw)) + 1

    # Phylo2Vec vector
    v = np.zeros(n_leaves, dtype=np.int16)

    # Whether each leaf node has been processed or not
    processed = np.zeros(n_leaves, dtype=bool)

    # TODO: documentation
    vmin = np.zeros(n_leaves, dtype=np.int16)
    labels = np.arange(n_leaves, dtype=np.int16)

    try:
        for _ in range(n_leaves - 1):
            # Name of left leaf
            left_leaf = ""

            for i in range(n_leaves):
                if processed[n_leaves - i - 1] == 0:
                    # Find whether the node with the current label has a sister node
                    label = labels[n_leaves - i - 1]

                    # Is label on the left of a newick pair?
                    if nw.find(f"({label},") > -1:
                        left_sep = f"({label},"
                        right_sep = ")"
                        # Sister node = substring between last left_sep and first right_sep
                        left_leaf = nw.rpartition(left_sep)[2].partition(right_sep)[0]

                    # Is label on the right of a newick pair?
                    elif nw.find(f",{label})") > -1:
                        left_sep = "("
                        right_sep = f",{label})"
                        # Sister node = substring between last left_sep and first right_sep
                        left_leaf = nw.partition(right_sep)[0].rpartition(left_sep)[2]

                    # Otherwise --> it has no sister node No sister node --> we can skip it
                    else:
                        continue

                    # If the sister substring is an actual digit, we can stop
                    if left_leaf.isdigit():
                        break

                    # Reset the left_leaf if it wasn't a digit
                    else:
                        left_leaf = ""

            left_leaf_ind = np.arange(len(labels))[labels == int(left_leaf)][0]
            right_leaf = n_leaves - i - 1

            for n in range(right_leaf + 1, n_leaves):
                if not processed[n]:
                    if vmin[n] == 0:
                        vmin[n] = n
                    else:
                        vmin[n] += 1

            labels[left_leaf_ind] = labels.max() + 1

            if vmin[right_leaf] == 0:
                v[right_leaf] = left_leaf_ind
            else:
                v[right_leaf] = vmin[right_leaf]

            # Update the processed vector
            processed[right_leaf] = True

            # Update the Newick string
            nw = nw.replace(
                f"({left_leaf},{labels[right_leaf]})", str(labels[int(left_leaf_ind)])
            )
            nw = nw.replace(
                f"({labels[right_leaf]},{left_leaf})", str(labels[int(left_leaf_ind)])
            )
    except IndexError as e:
        raise IndexError(
            "Have you tried reroot=True? "
            "Are the Newick nodes integers (and not taxa)? "
            "If the error still persists, your tree might be unrooted or non-binary."
        ) from e
    return v


def change_v(reorder_method, v_old, taxa_dict_old, ete3_format=9, **kwargs):
    """Shuffle v by reordering leaf labels

    Current pipeline: _get_ancestry --> change M --> _build_tree --> newick2v

    cf. gd.utils.tree._reorder for more details.

    Parameters
    ----------
    v_old : numpy.ndarray or list
        Current Phylo2vec vector
    taxa_dict_old : dict[int, str]
        Current mapping of node label (integer) to taxa
    ete3_format : int, optional
        Newick string format in ete3, by default 9

    Returns
    -------
    v_new : numpy.ndarray or list
        New Phylo2vec vector
    taxa_dict_new : dict[int, str]
        Updated mapping after shuffling
    """
    # TODO: make this function inplace?
    # Get ancestry
    M_old = _get_ancestry(v_old)

    # Reorder M
    M_new, taxa_dict_new = _reorder(reorder_method, M_old, taxa_dict_old, **kwargs)

    # Build tree
    tree = _build_tree(M_new.astype(str))

    # Update v via newick2v
    v_new = newick2v(
        tree.write(format=ete3_format), ete3_format=ete3_format, n_leaves=len(v_old) + 1
    )[1:]

    return v_new, taxa_dict_new


def _reorder(method, M_old, taxa_dict_old_, **kwargs):
    """Reorder nodes according to a specific method

    Parameters
    ----------
    method : str
        Name of reordering method. Available: 'birth_death', 'bfs', 'chronological'.
    M_old : numpy.ndarray
        Current "ancestry" array of size (n_leaves - 1, 3)
    taxa_dict_old_ : dict[int, str]
        Current mapping of node label (integer) to taxa

    Returns
    -------
    M_new : numpy.ndarray
       New "ancestry" array
    taxa_dict_new : dict[int, str]
        New mapping of node label (integer) to taxa
    """
    if method == "birth_death":
        return _reorder_birth_death(M_old, taxa_dict_old_, **kwargs)
    elif method == "bfs":
        return _reorder_bfs(M_old, taxa_dict_old_)
    elif method == "chronological":
        return _reorder_chronological(M_old, taxa_dict_old_, **kwargs)
    elif method == "custom":
        return _reorder_custom(M_old, taxa_dict_old_, **kwargs)
    else:
        raise ValueError(
            "`method` must be 'birth_death', 'bfs', 'chronological', or 'custom'."
        )


def _reorder_birth_death(
    M_old, taxa_dict_old, reorder_internal=True, shuffle_cols=False
):
    """Reorder leaf nodes via its "ancestry" array

    The input M should always be 3-dimensional with the following format:
    1st column: parent node
    2nd column: children 1
    3rd column: children 2

    Structure of the algorithm:
     * Traverse the tree in a level-order fashion (BFS)
     * Reorder the internal labels according to their depth
       --> the root will be labelled R=2*(n_leaves - 1), its descendants R-1, R-2, etc.
     * Reorder the leaf nodes in the following fashion:

    Example:
                   ////-3
             ////6|
       ////7|      \\\\-2
      |     |
    -8|      \\\\-1
      |
      |      ////-4
       \\\\5|
             \\\\-0

    The ancestry array of this tree is:
    [8, 7, 5]
    [7, 6, 1]
    [6, 3, 2]
    [5, 4, 0]

    Unrolled, it becomes:
    8 7 5 6 1 3 2 4 0

    We encode the nodes as it:
    Start by encoding the first two non-root nodes as 0, 1
    For the next pairs:
     * The left member takes the label was the previous parent node
     * The right member increments the previous right member by 1

    Ex:
    8 7 5 6 1 3 2 4 0
      0 1 0 2

    then

    8 7 5 6 1 3 2 4 0
      0 1 0 2 1 3

    then

    8 7 5 6 1 3 2 4 0
      0 1 0 2 1 3 0 4

    The code for the leaf nodes (0, 1, 2, 3, 4) is their new label

    Parameters
    ----------
    M : numpy.ndarray
        Current "ancestry" array of size (n_leaves - 1, 3)
    taxa_dict_old : dict[int, str]
        Current mapping of node label (integer) to taxa

    Returns
    -------
    M_new : numpy.ndarray
       New "ancestry" array
    taxa_dict_new : dict[int, str]
        New mapping of node label (integer) to taxa
    """
    # Copy old M
    M_new = M_old.copy()

    # Internal nodes to visit (2*len(M_old) = root label)
    to_visit = [2 * len(M_old)]

    # Number of visits
    visits = 1

    # Internal labels
    internal_labels = list(range(len(M_old) + 1, 2 * len(M_old)))

    # Leaf "code"
    node_code = []

    # List of all visited nodes
    visited = []

    # List of visited internal nodes
    visited_internals = []

    # Taxa dict to be updated
    taxa_dict_new = {}

    # col_orders = np.array(
    #     [
    #         [0, 1],
    #         [1, 0],
    #         [1, 0],
    #         [0, 1],
    #         [1, 0],
    #         [0, 1],
    #         [1, 0]
    #     ],
    #     dtype=np.int16
    # )

    while len(to_visit) > 0:
        row = 2 * len(M_old) - to_visit.pop(0)

        # some trickery
        if node_code:
            next_pair = [node_code[visited.index(visited_internals.pop(0))], visits]
        else:
            next_pair = [0, 1]

        if shuffle_cols:
            col_order = 2 * random.randint(0, 1) - 1
            # col_order = -2 * col_orders[row, 0] + 1
            # print([int(-0.5*col_order+0.5), int(1-(-0.5*col_order+0.5))])
            M_old[row, 1:] = M_old[row, 1:][::col_order]
            next_pair = next_pair[::col_order]

        for i, child in enumerate(M_old[row, 1:]):
            if child < len(M_old) + 1:
                # Update taxa dict (not sure that's correct but hey)
                taxa_dict_new[next_pair[i]] = taxa_dict_old[child]

                # Update M_new (23.01: ugly)
                M_new[row, i + 1] = next_pair[i]

            # Not a leaf node --> add it to the visit list
            else:
                visited_internals.append(child)
                if reorder_internal:
                    # Basically, flip the nodes
                    # Ex: relabel 7 in M_old as 9 in M_new
                    # Then relabel 9 in M_old as 7 in M_new
                    internal_node = internal_labels.pop()
                    M_new[row, i + 1] = internal_node
                    M_new[2 * len(M_new) - M_old[row, i + 1], 0] = M_new[row, i + 1]

                to_visit.append(child)

        visited.extend(M_old[row, 1:])

        node_code.extend(next_pair)
        visits += 1

    # Re-sort M such that the root node R is the first row, then internal nodes R-1, R-2, ...
    return M_new[M_new[:, 0].argsort()[::-1]], taxa_dict_new


def _reorder_bfs(M_old, taxa_dict_old):
    """Reorder leaf nodes via its "ancestry" array (UNUSED)

    The input M should always be 3-dimensional with the following format:
    1st column: parent node
    2nd column: children 1
    3rd column: children 2

    This is essentially an adaptation of breadth-first search for M.

    The leaf orders are labelled according to their depth
    i.e., leaf 0 has the fewest connections to the root,
    and the last leaf label is the deepest.

    Parameters
    ----------
    M : numpy.ndarray
        Current "ancestry" array of size (n_leaves - 1, 3)
    taxa_dict_old : dict[int, str]
        Current mapping of node label (integer) to taxa

    Returns
    -------
    M_new : numpy.ndarray
       New "ancestry" array
    taxa_dict_new : dict[int, str]
        New mapping of node label (integer) to taxa
    """
    # Copy old M
    M_new = M_old.copy()

    # Internal nodes to visit (2*len(M_old) = root label)
    to_visit = [2 * len(M_old)]

    # Leaf order
    order = []

    # Taxa dict to be updated
    taxa_dict_new = {}

    while len(to_visit) > 0:
        # Current row of M
        row = 2 * len(M_old) - to_visit.pop(0)

        for i, child in enumerate(M_old[row, 1:]):
            # Leaf node
            if child < len(M_old) + 1:
                order.append(child)

                # Update taxa dict
                taxa_dict_new[len(order) - 1] = taxa_dict_old[child]

                # Update M_new
                M_new[row, i + 1] = len(order) - 1

            # Not a leaf node --> add it to the visit list
            else:
                to_visit.append(child)

    return M_new, taxa_dict_new


def _reorder_custom(M_old, taxa_dict_old, order):
    taxa_dict_new = dict(enumerate(order))

    taxa_dict_new_reversed = {v: k for k, v in taxa_dict_new.items()}

    key_changes = {k: taxa_dict_new_reversed[v] for k, v in taxa_dict_old.items()}

    M_new = np.vectorize(lambda x: key_changes.get(x, x))(M_old)

    return M_new, taxa_dict_new


def _reorder_chronological(M_old, taxa_dict_old, dates):
    dates_sorted = {
        k: v for (k, v) in sorted(dates.items(), key=lambda x: (x[1], random.random()))
    }
    # this is the ugliest fix
    taxa_dict_new = dict(enumerate(dates_sorted.keys()))

    inv_taxa_dict_old = {v: k for (k, v) in taxa_dict_old.items()}
    inv_taxa_dict_new = {v: k for (k, v) in taxa_dict_new.items()}

    for taxa_old, taxa_new in zip(inv_taxa_dict_old.keys(), inv_taxa_dict_new.keys()):
        old_taxa_key = inv_taxa_dict_old[taxa_old]
        new_taxa_key = inv_taxa_dict_new[taxa_new]

        if old_taxa_key != new_taxa_key:
            old_idx = M_old == old_taxa_key
            new_idx = M_old == new_taxa_key
            M_old[old_idx], M_old[new_idx] = new_taxa_key, old_taxa_key

    # Copy old M
    M_new = M_old.copy()

    # Internal nodes to visit (2*len(M_old) = root label)
    to_visit = [2 * len(M_old)]

    # Leaf order
    order = []

    while len(to_visit) > 0:
        # Current row of M
        row = 2 * len(M_old) - to_visit.pop(0)

        for i, child in enumerate(M_old[row, 1:]):
            # Leaf node
            if child < len(M_old) + 1:
                order.append(child)

                # Update M_new
                M_new[row, i + 1] = len(order) - 1

            # Not a leaf node --> add it to the visit list
            else:
                to_visit.append(child)

    return M_new, taxa_dict_new


def reroot_at_random(v):
    """Reroot a tree (via its Phylo2Vec vector v) at a random node

    Parameters
    ----------
    v : numpy.ndarray
        v representation of a tree

    Returns
    -------
    numpy.ndarray
        rerooted v
    """
    _, _, ete3_tree = label_tree(v)

    ete3_tree.set_outgroup(f"{random.randint(0, 2 * len(v) - 1)}")

    newick = ete3_tree.write(format=9)

    return newick2v(newick, n_leaves=len(v) + 1)[1:]
