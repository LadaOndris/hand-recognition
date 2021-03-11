from datetime import datetime
import os
from src.utils.paths import LOGS_DIR, SAVED_MODELS_DIR


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_timestamped_dir(path: str) -> str:
    """
    Creates a new directory with a name of a current timestamp
    in a location defined by 'path'.
    Parameters
    ----------
    path    string : Location of the new directory.

    Returns
    -------
    Returns path to the new directory.
    """
    timestamp = get_current_timestamp()
    subdir = path.joinpath(timestamp)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def make_log_dir() -> str:
    """
    Creates a new directory with a timestamp in the logs directory.
    """
    if not os.path.isdir(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    return make_timestamped_dir(LOGS_DIR)


def compose_ckpt_path(log_dir: str):
    ckpt_folder = os.path.join(log_dir, 'train_ckpts')
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)
    return os.path.join(ckpt_folder, 'weights.{epoch:02d}.h5')


def compose_model_path(prefix='', suffix=''):
    if not os.path.isdir(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
    timestamp = get_current_timestamp()
    posixpath = SAVED_MODELS_DIR.joinpath(F"{prefix}{timestamp}{suffix}.h5")
    return str(posixpath)
