"""
Used for creating the input matrices for the experiments where the data
aren't modified; this prevents some data prep/loading redundancies later on.
"""
from config import Config
from data import PosDataModule

import sys


def prepare_vanilla_data(config):
    with open(config.tagset_path, encoding="utf8") as f:
        pos2idx = {}
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pos2idx[line] = i
    dm = PosDataModule(config, pos2idx)
    dm.prepare_data()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 0.data_prep.py PATH_TO_CONFIG")
        sys.exit(0)

    config = Config()
    config.load(sys.argv[1])
    print(config)
    prepare_vanilla_data(config)
