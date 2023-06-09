# pylint: disable=invalid-name
"""Phylo2Vec definition."""
import random
import re
import warnings

import numpy as np

from ete3 import Tree

from utils.data import parse_dates, read_fasta

POS_INT_PATTERN = re.compile(r"\b\d+\b")


class Phylo2Vec:
    """A Phylo2Vec object.

    Parameters
    ----------
    n_leaves : int, optional
        Number of leaf nodes/taxa, by default None
    msa_path : str, optional
        Path to a FASTA file for an MSA, by default None
    v : numpy.ndarray, optional
        v representation of a tree, by default None
    newick : str
        Newick representation of a tree, by default None
    date_path : str, optional
        Path to a CSV file containing dates for each taxon, by default None
        cf. treetime
    rooted : bool, optional
        If False, unroot the tree, by default True
    """

    def __init__(
        self,
        n_leaves=None,
        msa_path=None,
        v=None,
        newick=None,
        date_path=None,
        rooted=True,
    ):
        self.n_leaves = n_leaves
        self.data = None
        self.v = v
        self.newick = newick
        self.rooted = rooted

        if self.n_leaves is None:
            # Find the number of taxa/leaves using a predefined v
            if self.v is not None:
                self.v = v
                self.n_leaves = len(self.v) + 1

            # Find the number of taxa/leaves using a predefined Newick string
            if self.newick is not None:
                # TODO add taxa dict from Newick
                if v is not None:
                    warnings.warn(
                        "Value of 'v' will be overwritten using 'newick'.", UserWarning
                    )

                self.v = self.newick2v(self.newick, n_leaves=self.n_leaves)
                self.n_leaves = len(self.v) + 1

            # Find the number of taxa/leaves using an MSA
            if msa_path is not None:
                self.data = read_fasta(msa_path)
                self.taxa_dict = dict(enumerate(self.data.columns))
                self.n_leaves = self.data.shape[1]

                self._check_msa_and_v()
                self._check_msa_and_newick()
            elif self.n_leaves is None:
                raise ValueError(
                    "At least one of `msa_path`, `n_leaves`, "
                    "`newick`, or `v` must be not `None`."
                )
            else:
                self.taxa_dict = {i: f"t{i}" for i in range(self.n_leaves)}

        else:
            # Sample a random v
            self.v = self.sample()
            self.taxa_dict = {i: f"t{i}" for i in range(self.n_leaves)}

        self.n_nodes = 2 * self.n_leaves - 1

        # Chronological data
        if date_path is not None:
            _dates = parse_dates(date_path)
            # Sort with random ties
            self.dates = dict(
                sorted(_dates.items(), key=lambda x: (x[1], random.random()))
            )
        else:
            self.dates = None

    def __repr__(self):
        format_string = f"{self.__class__.__name__}("

        for item in ["v", "newick", "n_leaves", "taxa_dict", "dates"]:
            format_string += "\n"
            # TODO: pprint if dict?
            format_string += f"\t{item}={repr(self.__getattribute__(item))},"

        format_string = format_string[:-1] + "\n)"

        return format_string

    def sample(self):
        """Sample a random tree via Phylo2Vec

        Returns
        -------
        numpy.ndarray
            Phylo2Vec vector where v_i in {0, 1, ..., 2*i}
        """
        return np.array([random.randint(0, 2 * i) for i in range(self.n_leaves - 1)])

    def reroot_at_random(self):
        """Reroot v by setting a random node as the outgroup."""
        ete3_tree = self.label_tree(with_taxa_labels=False)

        ete3_tree.set_outgroup(f"{random.randint(0, self.n_nodes - 2)}")

        self.newick = ete3_tree.write(format=9)

        self.v = self.newick2v(self.newick, n_leaves=self.n_leaves)[1:]

    def label_tree(self, with_taxa_labels=True):
        """Label a tree from Phylo2Vec

        Returns
        -------
        tree : ete3.Tree
            Tree representing the node ancestry (M) with branch_lengths
        """
        M = self._get_ancestry()

        # Build tree
        # if with_branch_lengths:
        #     # Get dummy branch lengths
        #     branch_lengths = _get_dummy_branch_lengths(M)
        #     tree = _build_tree_with_branch_lengths(M, branch_lengths)
        # else:
        tree = self._build_tree(M.astype(str))

        if not self.rooted:
            tree.unroot()

        if with_taxa_labels:
            for leaf in tree.iter_leaves():
                leaf.name = self.taxa_dict[int(leaf.name)]

        return tree

    # def reorder(self, method, **kwargs):
    #     """Reorder nodes according to a specific method

    #     Parameters
    #     ----------
    #     method : str
    #         Name of reordering method. Available: 'birth_death', 'bfs', 'chronological'.
    #     **kwargs : kwargs
    #         Optional keyword arguments

    #     Returns
    #     -------
    #     M_new : numpy.ndarray
    #         New "ancestry" array
    #     taxa_dict_new : dict[int, str]
    #         New mapping of node label (integer) to taxa
    #     """
    #     M_old = self._get_ancestry()

    #     if method == "birth_death":
    #         M_new, taxa_dict_new = self._reorder_birth_death(
    #             M_old, self.taxa_dict, **kwargs
    #         )
    #     elif method == "bfs":
    #         M_new, taxa_dict_new = self._reorder_bfs(M_old, self.taxa_dict)
    #     elif method == "chronological":
    #         M_new, taxa_dict_new = self._reorder_custom(
    #             M_old, self.taxa_dict, order=list(self.dates.keys())
    #         )
    #     elif method == "custom":
    #         M_new, taxa_dict_new = self._reorder_custom(M_old, self.taxa_dict, **kwargs)
    #     else:
    #         raise ValueError(
    #             "`method` must be 'birth_death', 'bfs', 'chronological', or 'custom'."
    #         )

    #     tree = self._build_tree(M_new.astype(str))

    #     self.v = self.newick2v(
    #         tree.write(format=9), ete3_format=9, n_leaves=self.n_leaves
    #     )[1:]

    #     self.taxa_dict = taxa_dict_new

    # def get_newick(self, ete3_format=8):
    #     return self.label_tree().write(
    #         format=ete3_format
    #     )  # if self.newick is None else self.newick

    def _check_msa_and_v(self):
        if self.v is None:
            self.v = self.sample()  # FIXME

        return len(self.v) + 1 == self.data.shape[1]

    def _check_msa_and_newick(self):
        if self.newick is None:
            return True

        return set(Tree(self.newick).get_leaf_names()) == set(self.data.columns)

    def _get_ancestry(self):
        """Get pairs and ancestry for each node given a v-representation.

        Returns
        -------
        M : numpy.ndarray
            1st column: parent
            2nd and 3rd column: children
        """
        k = len(self.v)

        assert np.all(self.v <= 2 * np.arange(k)), print(
            self.v, self.v <= 2 * np.arange(k)
        )

        labels = np.tril([np.arange(k + 1)] * (k + 1))

        # Which nodes have not been processed?
        not_processed = np.ones(k, dtype="bool")

        M = np.zeros((k, 3), dtype=np.int32)

        for _ in range(k):
            cond = (self.v <= labels[:k, :].max(1)) & (not_processed)
            n = len(cond) - cond[::-1].argmax() - 1

            m = np.argmax(labels[n, :] == self.v[n])

            M[_, 0] = labels[-1, m]
            M[_, 1] = labels[-1, n + 1]

            labels[n:, m] = labels[n:, :].max(1) + 1

            M[_, 2] = labels[-1, m]

            not_processed[n] = False

        return np.flip(M)

    @staticmethod
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
        # List of parent nodes
        par_nodes = []

        # List of Newick sub-strings
        sub_newicks = []

        for i in reversed(range(len(M))):
            par, ch1, ch2 = M[i, :]
            # Case 1: Both children are parent nodes, so we have sub-newicks for them
            if ch1 in par_nodes and ch2 in par_nodes:
                # Find their indices
                idx1 = par_nodes.index(ch1)
                idx2 = par_nodes.index(ch2)

                # Merge the sub-newicks and add the parent node
                sub_newicks[idx1] = f"({sub_newicks[idx1]},{sub_newicks[idx2]}){par}"

                # Update the parent node for the 1st children
                par_nodes[idx1] = par

                # Discard info on 2nd children as merged with the 1st children
                sub_newicks.remove(sub_newicks[idx2])
                par_nodes.remove(par_nodes[idx2])

            # Case 2: only the first child is a parent node
            elif ch1 in par_nodes:
                # Find its index
                idx = par_nodes.index(ch1)

                # Update its sub-Newick:
                # (sub_child1.1, sub_child1.2)child_1 becomes:
                # ((sub_child1.1, sub_child1.2)child_1, child_2)parent
                sub_newicks[
                    idx
                ] = f"({sub_newicks[idx].replace(ch1, f'{ch1},{ch2}){par}')}"

                # Update the parent node (first child is now just an internal node)
                par_nodes[idx] = par

            # Case 3: only the second child is a parent node (similar to Case 2)
            elif ch2 in par_nodes:
                idx = par_nodes.index(ch2)
                # (sub_child2.1, sub_child2.2)child_2 becomes:
                # ((sub_child2.1, sub_child2.2)child_2, child_2)parent
                sub_newicks[
                    idx
                ] = f"({sub_newicks[idx].replace(ch2, f'{ch2},{ch1}){par}')}"
                par_nodes[idx] = par

            # Case 4: the children nodes have not been added yet
            else:
                # Add a new sub-Newick for this triplet
                sub_newicks.append(f"({ch1},{ch2}){par}")

                # Append the parent node
                par_nodes.append(par)

        # Only one Newick sub-string should be left, with only one parent: the root node
        newick = sub_newicks[0] + ";"

        # Convert to ete3 (for legacy reasons)
        return Tree(newick, format=8)

    @staticmethod
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
        elif n_leaves is None:
            # Faster when n_leaves is used, this is just in case
            # TODO: n_leaves as non-optional argument?
            n_leaves = max(int(s) for s in re.findall(POS_INT_PATTERN, nw)) + 1

        # Phylo2Vec vector
        v = np.zeros(n_leaves, dtype=np.int16)

        # Whether each leaf node has been processed or not
        processed = np.zeros(n_leaves, dtype=bool)

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
                            left_leaf = nw.rpartition(left_sep)[2].partition(right_sep)[
                                0
                            ]

                        # Is label on the right of a newick pair?
                        elif nw.find(f",{label})") > -1:
                            left_sep = "("
                            right_sep = f",{label})"
                            # Sister node = substring between last left_sep and first right_sep
                            left_leaf = nw.partition(right_sep)[0].rpartition(left_sep)[
                                2
                            ]

                        # Otherwise --> it has no sister node No sister node --> we can skip it
                        else:
                            continue

                        # If the sister substring is an actual digit, we can stop
                        if left_leaf.isdigit():
                            break

                        # Reset the left_leaf if it wasn't a digit
                        else:
                            left_leaf = ""

                # TODO: finish documentation
                left_idx = np.arange(len(labels))[labels == int(left_leaf)][0]
                right_leaf = n_leaves - i - 1

                for n in range(right_leaf + 1, n_leaves):
                    if not processed[n]:
                        if vmin[n] == 0:
                            vmin[n] = n
                        else:
                            vmin[n] += 1

                labels[left_idx] = labels.max() + 1

                if vmin[right_leaf] == 0:
                    v[right_leaf] = left_idx
                else:
                    v[right_leaf] = vmin[right_leaf]

                # Update the processed vector
                processed[right_leaf] = True

                # Update the Newick string
                nw = nw.replace(
                    f"({left_leaf},{labels[right_leaf]})", str(labels[int(left_idx)])
                )
                nw = nw.replace(
                    f"({labels[right_leaf]},{left_leaf})", str(labels[int(left_idx)])
                )
        except IndexError as e:
            raise IndexError(
                "Have you tried reroot=True? "
                "Are the Newick nodes integers (and not taxa)? "
                "If the error still persists, your tree might be unrooted or non-binary."
            ) from e
        return v

    @staticmethod
    def _reorder_birth_death(M_old, taxa_dict_old, reorder_internal=True):
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

        while len(to_visit) > 0:
            row = 2 * len(M_old) - to_visit.pop(0)

            # some trickery
            if node_code:
                next_pair = [node_code[visited.index(visited_internals.pop(0))], visits]
            else:
                next_pair = [0, 1]

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

    @staticmethod
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

    @staticmethod
    def _reorder_custom(M_old, taxa_dict_old, order):
        taxa_dict_new = dict(enumerate(order))

        taxa_dict_new_reversed = {v: k for k, v in taxa_dict_new.items()}

        key_changes = {k: taxa_dict_new_reversed[v] for k, v in taxa_dict_old.items()}

        M_new = np.vectorize(lambda x: key_changes.get(x, x))(M_old)

        return M_new, taxa_dict_new
