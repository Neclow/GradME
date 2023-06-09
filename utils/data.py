# pylint: disable=invalid-name, line-too-long
"""Data loading functions."""
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace

import pandas as pd
import yaml

# MAPPING = {"a": 0, "c": 1, "g": 2, "t": 3, "n": 4, "-": 5}


def read_fasta(file_path):
    """Read a FASTA file into DataFrame.

    Parameters
    ----------
    file_path : str, path object, file-like object
        String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function

    Returns
    -------
    pandas.DataFrame
        Two-dimensional data structure where:
        Columns = Species/Taxa.
        1 row = 1 nucleotide/aa site.
    """
    sequences = {}

    with open(file_path, "r", encoding="utf-8") as f:
        current_seq_name = None
        current_seq_values = ""
        for line in f:
            line = line.strip()
            if line[0] == ">":
                if current_seq_name:
                    sequences[current_seq_name] = current_seq_values.lower()
                current_seq_name = line[1:]
                current_seq_values = ""
            else:
                current_seq_values += line

        # Add the final species to the dictionary
        sequences[current_seq_name] = current_seq_values.lower()

    return pd.DataFrame({key: list(val) for key, val in sequences.items()})


def load_yaml_config(file_path, quiet=False):
    """Read a configuration yml file into simple namespace

    Parameters
    ----------
    file_path : str, path object, file-like object implementing a write() function
        Path to config file

    Returns
    -------
    types.SimpleNamespace
        namespace version of the dict constructed by yaml.safe_load
    """
    cfg = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))

    if not quiet:
        pprint(cfg)

    return SimpleNamespace(**cfg)


def parse_dates(file_path, name_col="name", date_col="date"):
    return pd.read_csv(file_path).set_index(name_col)[date_col].to_dict()
