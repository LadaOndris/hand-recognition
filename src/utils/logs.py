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
        os.mkdir(subdir)
    return subdir


def make_log_dir() -> str:
    """
    Creates a new directory with a timestamp in the logs directory.
    """
    return make_timestamped_dir(LOGS_DIR)


def compose_ckpt_path(log_dir: str):
    return os.path.join(log_dir, 'train_ckpts', 'weights.{epoch:02d}.h5')


def compose_model_path(prefix='', suffix=''):
    timestamp = get_current_timestamp()
    return SAVED_MODELS_DIR.joinpath(F"{prefix}{timestamp}{suffix}.h5")
