import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.detection.yolov3.utils import draw_centroid
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection
from src.acceptance.base import hand_orientation, joint_relation_errors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Tuple
import numpy as np
import matplotlib.colors as colors
import matplotlib.ticker as tick
from src.utils.camera import Camera

depth_image_cmap = 'gist_yarg'


def plot_depth_image(image, fig_location=None, figsize=(3, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    # fig.tight_layout()
    save_show_fig(fig, fig_location, show_figure=True)


class FixZorderCollection(Poly3DCollection):
    _zorder = -1

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        pass


def plot_two_hands_diff(hand1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        hand2: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        cam: Camera,
                        show_norm=False, show_joint_errors=False,
                        fig_location=None, show_fig=True):
    """
    Plots an image with a skeleton of the first hand and
    shows major differences from the second hand.

    Parameters
    ----------
    hand1 A tuple of three ndarrays: image, bbox, joints
    hand2 A tuple of three ndarrays: image, bbox, joints
    show_norm   bool True if vectors of the hand orientations should be displayed.
    show_diffs  bool True if differences between the hand poses should be displayed.
    """
    # plot the first skeleton
    # plot the second skeleton
    # compute the error between these skeletons
    # aggregate the errors for each joint
    # display the error for each joint
    image1, bbox1, joints1 = hand1
    image2, bbox2, joints2 = hand2

    if show_joint_errors:
        errors = joint_relation_errors(np.expand_dims(joints1, axis=0), np.expand_dims(joints2, axis=0))[0, 0, :]
    else:
        errors = None

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 5),
                             gridspec_kw={"width_ratios": [5.0 / 6, 1.5 / 6]})
    hand_axis = axes[0]
    bar_axis = axes[1]
    # hand_axis.set_position([0, 0, 0.9, 1])
    # bar_axis.set_position([0.9, 0, 1, 1])
    _plot_depth_image(hand_axis, image1)
    joints1_2d = cam.world_to_pixel(joints1)
    # The image is cropped. Amend the joint coordinates to match the depth image
    joints1_2d = joints1_2d[..., :2] - bbox1[..., :2]
    _plot_hand_skeleton_2d(hand_axis, joints1_2d, wrist_index=0, scatter=False)
    _plot_joint_errors_2d(fig, hand_axis, bar_axis, joints1_2d, joint_errors=errors, color='blue')

    if show_norm:
        norm, mean = hand_orientation(joints1)
        norm_2d, mean_2d = cam.world_to_pixel(np.stack([mean + 20 * norm, mean]))
        norm_2d = norm_2d[..., :2] - bbox1[..., :2]
        mean_2d = mean_2d[..., :2] - bbox1[..., :2]
        hand_axis.arrow(mean_2d[0], mean_2d[1], dx=norm_2d[0] - mean_2d[0], dy=norm_2d[1] - mean_2d[1],
                        color='orange', head_length=5, shape='full', head_width=4, zorder=1000,
                        width=1)
    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_axis_off()
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def cmap_subset(cmap, min, max):
    """ Create a subset of a cmap. """
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min, b=max),
        cmap(np.linspace(min, max, 256)))


def _plot_joint_errors_2d(fig, hand_axis, bar_axis, joints, joint_errors, color='r'):
    plt.rcParams.update({"font.size": 18})
    min_err, max_err = joint_errors.min(), joint_errors.max()
    min_s, max_s = min_err * 4, max_err * 4  # 20., 200.
    scaled_errors = joint_errors / (max_err / (max_s - min_s)) + min_s

    cmap = cmap_subset(cm.get_cmap('Reds'), 0.4, 1.0)
    hand_axis.scatter(joints[..., 0], joints[..., 1],
                      c=joint_errors, cmap=cmap, s=scaled_errors, alpha=1, zorder=100)
    ax_inset = inset_axes(bar_axis, width="100%", height="100%", loc='upper center',
                          bbox_to_anchor=(0, 0.35, 0.15, 0.4), bbox_transform=bar_axis.transAxes)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=ax_inset, format='%.2f')
    colorbar_tick_labels = np.linspace(min_err, max_err, 5, dtype=float)
    colorbar_tick_labels = np.round(colorbar_tick_labels, 1)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels(colorbar_tick_labels)
    cbar.ax.set_ylabel('Joint relation error [mm]', labelpad=15)
    # cbar.ax.xaxis.set_label_position('top')


def _plot_hand_skeleton_2d(ax, joints, wrist_index=0, s=20, alpha=1, marker='o', scatter=True,
                           finger_colors='rmcgb', linewidth=2, linestyle='solid'):
    joints = np.squeeze(joints)  # get rid of surplus dimensions
    if joints.ndim != 2:
        raise ValueError(F"joints.ndim should be 2, but is {joints.ndim}")
    fingers_bases = np.arange(wrist_index + 1, wrist_index + 20, 4)
    wrist_joint = joints[wrist_index]
    if scatter:
        ax.scatter(wrist_joint[..., 0], wrist_joint[..., 1], c=finger_colors[-1], marker=marker, s=s, alpha=alpha)
    for i, finger_base in enumerate(fingers_bases):
        finger_joints = joints[finger_base:finger_base + 4]
        if scatter:
            ax.scatter(finger_joints[..., 0], finger_joints[..., 1], c=finger_colors[i], marker=marker, s=s,
                       alpha=alpha)
        xs = np.concatenate([wrist_joint[0:1], finger_joints[:, 0]])
        ys = np.concatenate([wrist_joint[1:2], finger_joints[:, 1]])
        # if joints.shape[-1] == 3:
        #     zs = np.concatenate([wrist_joint[2:3], finger_joints[:, 2]])
        #     ax.plot(xs, ys, zs=zs, c=finger_colors[i])
        # else:
        ax.plot(xs, ys, c=finger_colors[i], linewidth=linewidth, linestyle=linestyle)


def plot_norm(joints_3d):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(joints_3d[..., 0], joints_3d[..., 1], zs=joints_3d[..., 2])
    norm, mean = hand_orientation(joints_3d)
    ax.quiver(mean[0], mean[1], mean[2], norm[0], norm[1], norm[2],
              pivot='tail', length=np.std(joints_3d), arrow_length_ratio=0.1)

    ax.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
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

    for i in range(len(ax.collections)):
        ax.collections[i].__class__ = FixZorderCollection

    _plot_hand_skeleton_2d(ax, joints, wrist_index=0, scatter=False)
    _plot_joint_errors_2d(fig, ax, joints)

    if show_norm:
        norm, mean = hand_orientation(joints)
        ax.quiver(mean[0], mean[1], mean[2], norm[0], norm[1], norm[2],
                  pivot='tail', length=np.std(joints), arrow_length_ratio=0.1)

    ax.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
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


def _plot_depth_image(ax, image):
    ax.imshow(image, cmap=depth_image_cmap)


def plot_joints_2d(image, joints2d, show_fig=True, fig_location=None, figsize=(4, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    _plot_hand_skeleton_2d(ax, joints2d)
    ax.set_axis_off()
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def plot_joints_with_annotations(image, joints_pred, joints_true, show_fig=True, fig_location=None, figsize=(4, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    first_color = np.full(shape=[5], fill_value="orange")
    second_color = np.full(shape=[5], fill_value="red")
    _plot_hand_skeleton_2d(ax, joints_true, scatter=False, linewidth=2, linestyle=(0, (2, 2)))
    _plot_hand_skeleton_2d(ax, joints_pred, scatter=False, linewidth=2)
    ax.set_axis_off()
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def plot_skeleton_with_label(image, joints2d, label, show_fig=True, fig_location=None, figsize=(4, 3)):
    width, height = image.shape[1], image.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    _plot_hand_skeleton_2d(ax, joints2d)
    ax.text(30, 30, label)
    ax.set_axis_off()
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def plot_joints_2d_from_3d(image, joints3d, cam: Camera, show_norm=False):
    fig, ax = plt.subplots(1)
    _plot_depth_image(ax, image)
    # project joint coords through pinhole camera
    joints2d = cam.world_to_pixel(joints3d)
    _plot_hand_skeleton_2d(ax, joints2d)
    if show_norm:
        norm, mean = hand_orientation(joints3d)
        norm_2d, mean_2d = cam.world_to_pixel(np.stack([mean + norm, mean]))
        ax.quiver(mean_2d[0], mean_2d[1], norm_2d[0], norm_2d[1], pivot='tail')
    fig.tight_layout()
    plt.show()


def plot_proportion_below_threshold(proportions, show_figure=True, fig_location=None):
    if np.max(proportions) <= 1:
        proportions *= 100
    fig, ax = plt.subplots(1, 1)
    ax.plot(proportions)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Proportion of frames within distnace (%)')
    ax.set_xlabel('Max joint error threshold (mm)')
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_figure)


def plot_depth_image_histogram(image, show_fig=True, fig_location=None):
    plt.rcParams.update({"font.size": 24})
    fig = plt.figure(figsize=(10, 7))
    min, max = np.min(image), np.max(image)
    plt.hist(image, bins=np.arange(min, max + 1, step=1), histtype='stepfilled')
    plt.xlabel('Depth [mm]', labelpad=20)
    plt.ylabel('Frequency', labelpad=20)
    plt.margins(x=0, y=0)
    plt.tick_params(axis='x', pad=15)
    plt.tick_params(axis='y', pad=15)
    plt.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def plot_image_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    _plot_depth_image(ax1, original)
    ax1.set_title('original')
    ax1.axis('off')
    _plot_depth_image(ax2, filtered)
    ax2.set_title(filter_name)
    ax2.axis('off')
    fig.show()


def plot_bounding_cube(image, bcube, cam: Camera, fig_location=None, show_fig=True):
    def get_four_points(P, dx, dy):
        P = np.array(P)
        P1 = P.copy()
        P2 = P.copy()
        P3 = P.copy()
        P4 = P.copy()
        P2[0] += dx
        P3[1] += dy
        P4[:2] += [dx, dy]
        Ps_xyz = np.array([P1, P2, P4, P3])
        Ps_uv = cam.world_to_pixel(Ps_xyz)[:, :2]
        return Ps_uv

    Au, Av, Az, Bu, Bv, Bz = bcube
    A = np.array([Au, Av, Az])
    B = np.array([Bu, Bv, Bz])
    A = cam.pixel_to_world(A[np.newaxis, ...])[0]
    B = cam.pixel_to_world(B[np.newaxis, ...])[0]
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    As = get_four_points(A, dx, dy)
    Bs = get_four_points(B, -dx, -dy)
    rect1 = np.concatenate([As, [As[0]]])
    rect2 = np.concatenate([Bs, [Bs[0]]])

    fig, ax = plt.subplots(1, figsize=(3, 3))
    _plot_depth_image(ax, image)
    color = '#c61732'
    ax.plot(rect1[:, 0], rect1[:, 1], c=color)
    ax.plot(rect2[:, 0], rect2[:, 1], c=color)
    ax.plot([As[0, 0], Bs[2, 0]], [As[0, 1], Bs[2, 1]], c=color)
    ax.plot([As[1, 0], Bs[3, 0]], [As[1, 1], Bs[3, 1]], c=color)
    ax.plot([As[2, 0], Bs[0, 0]], [As[2, 1], Bs[0, 1]], c=color)
    ax.plot([As[3, 0], Bs[1, 0]], [As[3, 1], Bs[1, 1]], c=color)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save_show_fig(fig, fig_location, show_fig)


def save_show_fig(fig, fig_location, show_figure):
    if fig_location:
        fig.savefig(fig_location)
    if show_figure:
        fig.show()
