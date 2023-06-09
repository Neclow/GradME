"""Data loading."""
from pathlib import Path

import jax.numpy as jnp
import rpy2.robjects as ro

from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


def load_data(cfg):
    """Load data according to a configuration file

    Data = distance matrix and Newick-format trees produced using R for FastME and NJ

    Parameters
    ----------
    cfg : types.SimpleNamespace
        Configuration parameters, loaded from a .yml file

    Returns
    -------
    n_sites : int
        Number of sites in each aligned sequence
    D : jax.numpy.ndarray
        Distance matrix (with ordered columns)
    names : list
        List of taxon names in the multiple sequence alignment
    fastme_newick : str
        Newick-format tree from FastME for the alignment of interest
    nj_newick : str
        Newick-format tree from NJ for the alignment of interest
    """
    with localconverter(ro.default_converter + numpy2ri.converter):
        importr("phangorn")

        ro.globalenv["Rpath"] = str(Path(cfg.repo_path, "data", cfg.fasta_path))
        ro.globalenv["model"] = cfg.substitution_model

        # DNA Evolution model: F81 + Gamma
        data = ro.r(
            """
            aln <- read.FASTA(Rpath, type = "DNA")

            if (model != "JC69" && model != "F81") {
                dm <- dist.dna(aln, model = model)
                if (sum(is.na(dm)) > 0) {
                    # In case dist.dna doesn't work
                    warning(paste("NA found using dist.dna with model = ", model, ". Using F81"))

                    dm <- dist.ml(aln, model = "F81")
                }
            } else {
                dm <- dist.ml(aln, model = model)
            }

            D <- as.matrix(dm)

            ape_tree <- fastme.bal(dm, spr = TRUE)
            nj_tree <- bionj(dm)

            list(
                n_sites = length(aln[[1]]),
                D = D,
                names = colnames(D),
                fastme_newick = write.tree(ape_tree),
                nj_newick = write.tree(nj_tree)
            )
            """
        )

    n_sites = int(data["n_sites"])
    D = jnp.asarray(data["D"])
    names = list(data["names"])
    fastme_newick = data["fastme_newick"][0]
    nj_newick = data["nj_newick"][0]

    return n_sites, D, names, fastme_newick, nj_newick
