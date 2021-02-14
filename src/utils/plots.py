import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.detection.yolov3.utils import draw_centroid
from src.acceptance.base import best_fitting_hyperplane
import numpy as np

depth_image_cmap = 'gist_yarg'


def plot_depth_image(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap=depth_image_cmap)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_joints(image, bbox, joints, show_norm=False):
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

    ax.contourf(image3d[..., 0], image3d[..., 1], image3d[..., 2], levels=64, cmap='gist_gray', linewidths=0)

    joints[..., 2] = -joints[..., 2]  # Convert negative Z axis to positive
    ax.scatter(joints[..., 0], joints[..., 1], joints[..., 2], marker='o', s=20, c="red", alpha=1)

    if show_norm:
        # 0: wrist,
        # 1-4: index_mcp, index_pip, index_dip, index_tip,
        # 5-8: middle_mcp, middle_pip, middle_dip, middle_tip,
        # 9-12: ring_mcp, ring_pip, ring_dip, ring_tip,
        # 13-16: little_mcp, little_pip, little_dip, little_tip,
        # 17-20: thumb_mcp, thumb_pip, thumb_dip, thumb_tip

        palm_joints = np.take(joints, [0, 1, 5, 9, 13, 17], axis=0)
        norm, mean = best_fitting_hyperplane(palm_joints)
        direction = mean + norm
        ax.quiver(mean[0], mean[1], mean[2], norm[0], norm[1], norm[2],
                  pivot='tail', length=np.std(joints), arrow_length_ratio=0.2)
    # np.std(joints) / joints.shape[0]
    """
    ax.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.axis('off')
    """

    max_z = np.max(joints[..., 2]) * 1.2
    ax.set_xlim([0, max_z])
    ax.set_ylim([0, max_z])
    ax.set_zlim([0, max_z])
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
