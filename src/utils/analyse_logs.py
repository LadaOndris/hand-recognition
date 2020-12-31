import os
import paths


def plot_yolo_training(path):
    print(path)


if __name__ == "__main__":
    summary_path = os.path.join(paths.logs_path(), '20201016-125612')
    plot_yolo_training(summary_path)
