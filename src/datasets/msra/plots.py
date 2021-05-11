from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR, DOCS_DIR
from src.utils.plots import plot_image_with_skeleton
from src.datasets.msra.dataset import read_images
from src.utils.camera import Camera


def plot_image_with_annotation(subject_gesture_folder, image_index, show_fig, fig_location):
    images, bbox_coords, joints = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath(subject_gesture_folder))
    cam = Camera('msra')
    joints1_2d = cam.world_to_pixel(joints[image_index])
    plot_image_with_skeleton(images[image_index], joints1_2d[..., :2] - bbox_coords[image_index, ..., :2],
                             fig_location=fig_location, show_fig=show_fig)


def plot_images_with_annotations(show_fig=True):
    idx = 50
    path = str(DOCS_DIR.joinpath('figures/datasets/MSRASampleImage{}.png'))
    plot_image_with_annotation('P0/5', idx, show_fig, path.format(1))
    plot_image_with_annotation('P0/2', idx, show_fig, path.format(2))
    plot_image_with_annotation('P0/L', idx, show_fig, path.format(3))
    plot_image_with_annotation('P0/TIP', idx, show_fig, path.format(4))


if __name__ == '__main__':
    plot_images_with_annotations()
