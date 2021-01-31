import os
from paths import LOGS_DIR


def plot_yolo_training(path):
    """
    Plots metrics during training the YOLO model.
    """
    print(path)


if __name__ == "__main__":
    summary_path = os.path.join(LOGS_DIR, '20201016-125612')
    plot_yolo_training(summary_path)
