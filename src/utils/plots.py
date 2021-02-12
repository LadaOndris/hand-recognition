import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.detection.yolov3.utils import draw_centroid
import numpy as np

depth_image_cmap = 'gist_yarg'


def plot_depth_image(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap=depth_image_cmap)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_joints(image, bbox, joints):
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    y = np.linspace(top, bottom, height)
    x = np.linspace(left, right, width)
    xx, yy = np.meshgrid(x, y)

    image3d = np.stack([xx, yy, image], axis=2)
    image3d[..., 2] = np.where(image3d[..., 2] == 0, np.max(image), image3d[..., 2])

    ax.contourf(image3d[..., 0], image3d[..., 1], image3d[..., 2], levels=64, cmap='gist_gray',
                linewidth=0, edgecolor='none')

    ax.scatter(joints[..., 0], joints[..., 1], -joints[..., 2], marker='o', s=20, c="red", alpha=1)

    """
    ax.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.axis('off')
    """

    ax.set_xlim([0, 360])
    ax.set_ylim([0, 360])
    fig.tight_layout()
    plt.show()


def plot_joints_only(joints):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(joints[..., 0], joints[..., 1], joints[..., 2], marker='o', s=20, c="red", alpha=1)

    # ax.view_init(azim=0.0, elev=0.0)
    # ax.dist = 0

    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)

    ax.set_xlim([0, 360])
    ax.set_ylim([0, 360])
    fig.tight_layout()
    plt.show()
