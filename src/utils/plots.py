import functools

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils.camera import Camera

depth_image_cmap = 'gist_yarg'


def plotlive(func):
    """
    A decorator that updates the current figure
    instead of plotting into a new one.
    It requires that the figure is created before calling
    the plot function that is decorated.
    The figure should be passed as a parameter to the plot function.
    """
    plt.ion()

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # Clear current axis
        # plt.cla()

        # Clear all axes in the current figure.
        axes = plt.gcf().get_axes()
        for axis in axes:
            axis.cla()

        result = func(*args, **kwargs)

        plt.draw()
        plt.pause(0.01)

        return result

    return new_func


def plot_depth_image(image, fig_location=None, figsize=(3, 3)):
    """
    Plots the depth image in a new figure.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    # fig.tight_layout()
    save_show_fig(fig, fig_location, show_figure=True)


@plotlive
def plot_skeleton_with_jre_live(fig, axes, image, joints, jres, label=None,
                                norm_vec=None, mean_vec=None):
    """
    Plots the depth image, hand's skeleton, and a colorbar
    for the user, showing Joint Relation Errors.
    Replaces the contents of the currently active plot.
    """
    _plot_skeleton_with_jre(fig, axes, image, joints, jres, label, norm_vec, mean_vec)


def plot_skeleton_with_jre(image, joints, jres, label=None, show_fig=True, fig_location=None,
                           norm_vec=None, mean_vec=None):
    """
    Creates a new figure into which it plots the depth image,
    hand's skeleton, and a colorbar for the user, showing Joint Relation Errors.
    """
    fig, axes = plot_skeleton_with_jre_subplots()
    _plot_skeleton_with_jre(fig, axes, image, joints, jres, label, norm_vec, mean_vec)
    save_show_fig(fig, fig_location, show_fig)


def plot_skeleton_with_jre_subplots():
    """
    Creates a new figure for plotting the result
    of gesture recognition.
    The left axis contains the estimated hand pose,
    and the right axis displays colorbar.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 5),
                             gridspec_kw={"width_ratios": [5.0 / 6, 1.5 / 6]})
    bar_axis = axes[1]
    ax_inset = inset_axes(bar_axis, width="100%", height="100%", loc='upper center',
                          bbox_to_anchor=(0, 0.35, 0.15, 0.4), bbox_transform=bar_axis.transAxes)
    axes = [axes[0], axes[1], ax_inset]
    return fig, axes


def _plot_skeleton_with_jre(fig, axes, image, joints, jres, label=None,
                            norm_vec=None, mean_vec=None):
    """
    Plots the depth image, hand's skeleton, and a colorbar
    for the user, showing Joint Relation Errors.
    """
    hand_axis = axes[0]
    bar_axis = axes[2]
    _plot_depth_image(hand_axis, image)
    _plot_hand_skeleton(hand_axis, joints, wrist_index=0, scatter=False)
    _plot_joint_errors(fig, hand_axis, bar_axis, joints, joint_errors=jres)
    if norm_vec is not None and mean_vec is not None:
        _plot_hand_orientation(hand_axis, mean_vec, norm_vec)
    if label is not None:
        hand_axis.set_title(label)

    for ax in axes[:2]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_axis_off()


def _plot_hand_orientation(ax, mean, norm):
    """
    Plots an arrow (a vector) from the 'mean' position
    in the direction of the 'norm' vector.
    """
    ax.arrow(mean[0], mean[1], dx=norm[0] - mean[0], dy=norm[1] - mean[1],
             color='orange', head_length=5, shape='full', head_width=4, zorder=1000,
             width=1)


def cmap_subset(cmap, min, max):
    """ Create a subset of a cmap. """
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min, b=max),
        cmap(np.linspace(min, max, 256)))


def _plot_joint_errors(fig, hand_axis, bar_axis, joints, joint_errors):
    """
    Plots a colorbar in the given axis, displaying the
    range of Joint Relation Errors.
    """
    plt.rcParams.update({"font.size": 18})
    min_err, max_err = joint_errors.min(), joint_errors.max()
    min_s, max_s = min_err * 4, max_err * 4  # 20., 200.
    scaled_errors = joint_errors / (max_err / (max_s - min_s)) + min_s

    cmap = cmap_subset(cm.get_cmap('Reds'), 0.4, 1.0)
    hand_axis.scatter(joints[..., 0], joints[..., 1],
                      c=joint_errors, cmap=cmap, s=scaled_errors, alpha=1, zorder=100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=bar_axis, format='%.2f')
    colorbar_tick_labels = np.linspace(min_err, max_err, 5, dtype=float)
    colorbar_tick_labels = np.round(colorbar_tick_labels, 1)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels(colorbar_tick_labels)
    cbar.ax.set_ylabel('Joint relation error [mm]', labelpad=15, fontsize=18)


def _plot_hand_skeleton(ax, joints, wrist_index=0, s=20, alpha=1, marker='o', scatter=True,
                        finger_colors='rmcgb', linewidth=2, linestyle='solid'):
    """
    Plots the hand's skeleton from joints in image coordinates.
    """
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
        ax.plot(xs, ys, c=finger_colors[i], linewidth=linewidth, linestyle=linestyle)


def _plot_depth_image(ax, image):
    """
    Plot a depth image in an existing axis.
    """
    ax.imshow(image, cmap=depth_image_cmap)


@plotlive
def _plot_depth_image_live(ax, image):
    """
    Plot a depth image in an existing axis
    by replacing its previous content.
    """
    ax.imshow(image, cmap=depth_image_cmap)


def plot_image_with_skeleton(image, joints2d, show_fig=True, fig_location=None, figsize=(4, 3)):
    """
    Plots the depth image and the skeleton in a new figure.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_image_with_skeleton(fig, ax, image, joints2d)
    save_show_fig(fig, fig_location, show_fig)


@plotlive
def plot_image_with_skeleton_live(fig, ax, image, joints2d):
    """
    Plots the depth image and the skeleton in an existing figure
    by replacing its content.
    """
    _plot_image_with_skeleton(fig, ax, image, joints2d)


def _plot_image_with_skeleton(fig, ax, image, joints2d):
    """
    Plots the depth image and the skeleton in an existing figure.
    """
    _plot_depth_image(ax, image)
    _plot_hand_skeleton(ax, joints2d)
    ax.set_axis_off()
    fig.tight_layout()


def plot_joints_with_annotations(image, joints_pred, joints_true, show_fig=True, fig_location=None, figsize=(4, 3)):
    """
    Plots a depth image with predicted and ground truth skeletons.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_depth_image(ax, image)
    _plot_hand_skeleton(ax, joints_true, scatter=False, linewidth=2, linestyle=(0, (2, 2)))
    _plot_hand_skeleton(ax, joints_pred, scatter=False, linewidth=2)
    ax.set_axis_off()
    fig.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


@plotlive
def plot_skeleton_with_label_live(fig, ax, image, joints2d, label):
    """
    Plots a depth image with skeleton and
    a label above the axis.
    """
    _plot_depth_image(ax, image)
    _plot_hand_skeleton(ax, joints2d)
    ax.set_title(label)
    ax.set_axis_off()
    fig.tight_layout()


def plot_proportion_below_threshold(proportions, show_figure=True, fig_location=None):
    """
    Plots the Proportion of joints below threshold metric.
    """
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
    """
    Plots a histogram of depth values in the image.
    """
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
    """
    Plots two depth images next to each other.
    """
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
    """
    Projects a bounding cube in 3d coordinates
    onto the given depth image.
    """

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


def plot_scores(x, y, labels=None, fig_location=None, show_fig=True):
    """
    Plots evaluation metrics dependent on a threshold.
    """
    figsize = (8, 6)
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size": 24})

    fig = plt.figure(figsize=figsize)
    for y_score in y:
        plt.plot(x, y_score)
    if labels is not None:
        if len(labels) != len(y):
            raise ValueError('y shape is not the same as labels shape')
        plt.legend(labels=labels)
    plt.xlabel('Threshold', labelpad=20)
    plt.ylabel('Score', labelpad=20)
    plt.margins(x=0, y=0)
    plt.tick_params(axis='x', pad=15)
    plt.tick_params(axis='y', pad=15)
    plt.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def precision_recall_curve(precision, recall, fig_location=None, show_fig=True):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size": 24})
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    plt.plot(recall, precision)
    plt.xlabel('Recall', labelpad=20)
    plt.ylabel('Precision', labelpad=20)
    plt.margins(x=0, y=0)
    plt.tick_params(axis='x', pad=15)
    plt.tick_params(axis='y', pad=15)
    plt.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def roc_curve(fnr, fpr, fig_location=None, show_fig=True):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size": 24})
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    plt.plot(fpr, fnr)
    plt.xlabel('False positive rate', labelpad=20)
    plt.ylabel('False negative rate', labelpad=20)
    plt.margins(x=0, y=0)
    plt.tick_params(axis='x', pad=15)
    plt.tick_params(axis='y', pad=15)
    plt.tight_layout()
    save_show_fig(fig, fig_location, show_fig)


def histogram(y, label, fig_location=None, show_fig=True, **kwargs):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size": 24})
    fig = plt.figure(figsize=(8, 6))
    plt.hist(y, bins=50, density=True, **kwargs)
    _set_pad_and_labels(label, 'Density')
    save_show_fig(fig, fig_location, show_fig)


def _set_pad_and_labels(xlabel, ylabel):
    plt.xlabel(xlabel, labelpad=20)
    plt.ylabel(ylabel, labelpad=20)
    plt.margins(x=0, y=0)
    plt.tick_params(axis='x', pad=15)
    plt.tick_params(axis='y', pad=15)
    plt.tight_layout()


def save_show_fig(fig, fig_location, show_figure):
    if fig_location:
        fig.savefig(fig_location)
    if show_figure:
        fig.show()
