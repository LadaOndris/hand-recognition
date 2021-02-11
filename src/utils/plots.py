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


def plot_joints(image, joints):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    # fig, ax = plt.subplots(1)
    # ax = Axes3D(fig)
    xx, yy = np.meshgrid(np.linspace(0, image.shape[1], image.shape[1]), np.linspace(0, image.shape[0], image.shape[0]))
    ax.plot_surface(xx, yy, image, facecolors=cm.BrBG(image / image.max()), alpha=0.5)
    # ax.imshow(image, cmap=depth_image_cmap)

    ax.scatter(joints[..., 0], joints[..., 1], -joints[..., 2], marker='o', s=20, c="red", alpha=1)

    ax.view_init(azim=-90.0, elev=30.0)  # aligns the 3d coord with the camera view

    # for joint in joints:
    #    draw_centroid(ax, joint[:2])
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    # fig.tight_layout()
    plt.show()


def plot_joints_only(image, joints):
    fig, ax = plt.subplots(1)
    image = np.zeros(shape=(240, 320))
    ax.imshow(image, cmap=depth_image_cmap)
    for joint in joints:
        draw_centroid(ax, joint[:2])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.tight_layout()
    plt.show()
