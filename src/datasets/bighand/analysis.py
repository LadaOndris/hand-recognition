from src.utils.paths import BIGHAND_DATASET_DIR, DOCS_DIR
import numpy as np
from PIL import Image
from src.datasets.bighand.dataset import BighandDataset
from src.utils.camera import Camera
from src.utils.plots import plot_joints_2d
import tensorflow as tf


def get_line(file):
    with open(file, 'r') as f:
        return f.readline()


def show_sample_from_each_folder(save_fig_location_pattern=None):
    ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=1.0, batch_size=1, shuffle=False)
    camera = Camera('bighand')
    samples_per_file = 1
    for i, file in enumerate(ds.train_annotation_files):
        fig_location = str(save_fig_location_pattern).format(i)
        with open(file, 'r') as f:
            for i in range(samples_per_file):
                line = f.readline()
                image, joints = ds._prepare_sample(line)
                image = tf.squeeze(image)
                joints2d = camera.world_to_pixel(joints)
                plot_joints_2d(image, joints2d, fig_location=fig_location)
        print(F"{i}:", file)


if __name__ == '__main__':
    fig_location_pattern = DOCS_DIR.joinpath('images/datasets/BigHandSampleImage{}.png')
    show_sample_from_each_folder(fig_location_pattern)
