from datetime import datetime
import os

from src.utils.paths import LOGS_DIR, SAVED_MODELS_DIR


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_log_dir():
    """
    Creates a new directory with a timestamp in the logs directory.
    """
    timestamp = get_current_timestamp()
    log_subdir = LOGS_DIR.joinpath(timestamp)
    if not os.path.isdir(log_subdir):
        os.mkdir(log_subdir)
    return log_subdir


def compose_ckpt_path(log_dir: str):
    return os.path.join(log_dir, 'train_ckpts', 'weights.{epoch:02d}.h5')


def compose_model_path(prefix='', suffix=''):
    timestamp = get_current_timestamp()
    return SAVED_MODELS_DIR.joinpath(F"{prefix}{timestamp}{suffix}.h5")
