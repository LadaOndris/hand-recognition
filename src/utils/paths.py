import os

RELATIVE_BASE_PATH = "../../"


def logs_path(base_path=RELATIVE_BASE_PATH):
    return os.path.join(base_path, 'logs')

