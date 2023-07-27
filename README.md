# Leaping through tree space: continuous phylogenetic inference for rooted and unrooted trees

This repo hosts a minimal implementation of ```GradME```
* ```bme_jax/```: Balanced Minimum Evolution and distance-based optimisation with Phylo2Vec in Jax
* ```cfg/```: Example configuration files
* ```utils/```: Utility functions for manipulation of sequence and tree data.

## Environment setup
1. Setup the ```gradme``` environment using conda/mamba and activate the environment:
```
conda env create -f env.yml
conda activate gradme
```
2. Optional: if you have GPUs/TPUs, you might need to update your installation of Jax. Follow the instructions at https://github.com/google/jax
3. Install ```phangorn``` in R (4.2.2 or above):
```
install.packages("phangorn")
```

## Accessing data
The following datasets were used:

| Dataset   | Sites        | Taxa | Type       | Taxonomic rank                    | Access                                                                                             | TreeBASE ID |
|-----------|--------------|------|------------|-----------------------------------|----------------------------------------------------------------------------------------------------|-------------|
| DS1       | 1,949        | 27   | rRNA (18S) | Tetrapods                         | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2017  |
| DS2       | 2,520        | 29   | rRNA (18S) | Acanthocephalans                  | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2131  |
| DS3       | 1,812        | 36   | mtDNA      | Mammals; mainly Lemurs            | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M127   |
| DS4       | 1,137        | 41   | rDNA (18S) | Fungi; mainly Ascomycota          | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M487   |
| DS5       | 378          | 50   | DNA        | Lepidoptera                       | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2907  |
| DS6       | 1,133        | 50   | rDNA (28S) | Fungi; mainly Diaporthales        | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M220   |
| DS7       | 1,824        | 59   | mtDNA      | Mammals; mainly Lemurs            | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2449  |
| DS8       | 1,008        | 64   | rDNA (28S) | Fungi; mainly Hypocreales         | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2261  |
| DS9       | 955          | 67   | DNA        | Poaecae (grasses)                 | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2389  |
| DS10      | 1,098        | 67   | DNA        | Fungi; mainly Ascomycota          | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2152  |
| DS11      | 1,082        | 71   | DNA        | Lichen                            | [[1]](https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/) | M2274  |
| Eutherian | 1,338,678    | 37   | DNA        | Eutherian Mammals                 | [[2]](https://datadryad.org/stash/dataset/doi:10.5061/dryad.3629v) |
| Jawed     | 1,460-18,406 | 99   | AA         | Gnathostomata (jawed vertebrates) | [[3]](https://datadryad.org/stash/dataset/doi:10.5061/dryad.r2n70) |
| Primates  | 232          | 14   | mtDNA      | Mammals; mainly Primates          | [[4]](https://evolution.gs.washington.edu/book/datasets.html) |

* [1] https://bitbucket.org/XavMeyer/coevrj/src/master/data/adaptiveTreeProp/alignments/TreeBase/
* [2] https://datadryad.org/stash/dataset/doi:10.5061/dryad.3629v
* [3] https://datadryad.org/stash/dataset/doi:10.5061/dryad.r2n70
* [4] https://evolution.gs.washington.edu/book/datasets.html


DS1-DS8 are also available at: https://github.com/zcrabbit/vbpi-gnn/tree/main/data/hohna_datasets_fasta
DS1-DS11 should also be available on [TreeBASE](https://treebase.org/treebase-web/home.html) using the TreeBASE, but the site was down on June 9, 2023.

Sources: see manuscript

## Running GradME
1. Download the datasets (in the FASTA format) mentioned above and place them in a ```data/``` folder (e.g., in the repo)
2. Update the configuration file ```cfg/bme_config_v3.yml```, especially ```repo_path``` and ```fasta_path```
3. Run the main optimisation script: ```python -m bme_jax.main``` or use the ```demo.ipynb``` notebook
