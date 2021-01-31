import os
from pathlib import Path

RELATIVE_BASE_PATH = "../../"
_ROOT_PATH = Path(__file__).parent.parent.parent

ROOT_DIR = _ROOT_PATH
LOGS_DIR = _ROOT_PATH.joinpath('logs')
SAVED_MODELS_DIR = _ROOT_PATH.joinpath('saved_models')
SRC_DIR = _ROOT_PATH.joinpath('src')

HANDSEG_DATASET_DIR = _ROOT_PATH.joinpath("datasets/handseg150k")


