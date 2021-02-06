import pathlib
import os
from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR


file = MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/1/000100_depth.bin')
with open(file, 'r') as f:
    pass