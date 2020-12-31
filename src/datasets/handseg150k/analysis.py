from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from pathlib import Path
import numpy as np
import tensorflow as tf
import os
from dataset import HandsegDataset
from dataset_bboxes import HandsegDatasetBboxes

dirname = os.path.dirname(__file__)


def foo():
    depth_im = np.array(Image.open(os.path.join(dirname, 'images/user-4.00000003.png')))  # Loading depth image
    mask_im = np.array(Image.open(os.path.join(dirname, 'masks/user-4.00000003.png')))  # Loading mask image
    depth_im = depth_im.astype(np.float32)  # Converting to float
    mean = np.mean(depth_im)
    print(Counter(depth_im.flatten()))
    # plt.hist(mask_im, bins=100)
    # plt.show()
    mean_depth_ims = 10000.0  # Mean value of the depth images
    depth_im /= mean_depth_ims  # Normalizing depth image
    plt.imshow(depth_im)
    plt.title('Depth Image')
    plt.show()  # Displaying Depth Image
    plt.imshow(mask_im)
    plt.title('Mask Image')
    plt.show()  # Displaying Mask Image

    print(depth_im.shape)


def depth_hist():
    dataset = HandsegDatasetBboxes(batch_size=16, dataset_path=handseg_path, type='train', train_size=0.8, )
    for batch_images, batch_bboxes in dataset.batch_iterator:
        plt.hist(batch_images.numpy().flatten(), np.linspace(1, tf.reduce_max(batch_images), 100))
        break

    # depth_im = np.array(Image.open(os.path.join(handseg_path, 'images/user-4.00000010.png')))
    # plt.hist(depth_im.flatten(), np.linspace(1, depth_im.max(), 100))
    plt.yscale('log')
    plt.show()


def show_images_from_handseg_dataset(num, dataset_path, save_fig_loc=None):
    dataset = HandsegDataset(dataset_path=dataset_path)
    offset = 9
    images, masks = dataset.load_images(start_index=offset, end_index=offset + num)

    fig, ax = plt.subplots(nrows=2, ncols=num, figsize=(8, 4), constrained_layout=True)
    for i in range(num):
        ax[0, i].imshow(images[i], cmap='gist_yarg')
        ax[1, i].imshow(masks[i], cmap='gist_yarg')

        ax[0, i].xaxis.set_visible(False)
        ax[0, i].yaxis.set_visible(False)

        ax[1, i].xaxis.set_visible(False)
        ax[1, i].yaxis.set_visible(False)
        # ax[0, i].axis('off')
        # ax[1, i].axis('off')
    if save_fig_loc:
        plt.savefig(save_fig_loc)
    plt.show()


def show_images_with_bboxes(dataset):
    for batch_images, batch_bboxes in dataset.batch_iterator:
        for image, bboxes in zip(batch_images, batch_bboxes):
            fig, ax = plt.subplots(1)
            ax.imshow(np.squeeze(image, axis=-1))
            for bbox in bboxes:
                draw_bounding_box(ax, bbox)
            plt.title('Resized depth image with bounding boxes');
            plt.show()
        break


def get_bbox_from_mask(mask, hand_label):
    hand_coords = np.where(mask == hand_label)
    if len(hand_coords[0]) == 0:
        return None

    x_coords = hand_coords[1]
    y_coords = hand_coords[0]
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    return x_min, y_min, x_max, y_max


def draw_bounding_box(axis, bbox):
    if bbox is not None:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle((bbox[0], bbox[1]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        axis.add_patch(rect)


def generate_bounding_boxes(print_images=False, save_to_file=False, bboxes_filename='', max_count=None):
    filenames = os.listdir('masks/')

    lines = []
    count = 0
    for filename in filenames:
        if count % 10000 == 0:
            print(F"Generating bounding boxes: {count}/{len(filenames)}")
        if count == max_count:
            break

        image = np.array(Image.open(os.path.join(dirname, 'images', filename)))
        mask = np.array(Image.open(os.path.join(dirname, 'masks', filename)))
        # print(mask.shape)
        # print(np.histogram(mask, bins=[0,1,2,3]))

        first_bbox = get_bbox_from_mask(mask, 1)
        second_bbox = get_bbox_from_mask(mask, 2)

        line = filename
        if first_bbox is not None:
            line += F" {first_bbox[0]} {first_bbox[1]} {first_bbox[2]} {first_bbox[3]}"
        if second_bbox is not None:
            line += F" {second_bbox[0]} {second_bbox[1]} {second_bbox[2]} {second_bbox[3]}"
        line += '\n'
        lines.append(line)

        # print(first_bbox, second_bbox)

        if print_images:
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            draw_bounding_box(ax, first_bbox)
            draw_bounding_box(ax, second_bbox)
            plt.title('Depth image with bounding boxes')
            plt.show()

        count += 1

    if save_to_file:
        with open(bboxes_filename, 'w') as bboxes_file:
            bboxes_file.writelines(lines)


def analyse_pixel_distance():
    dataset = HandsegDatasetBboxes(batch_size=16, dataset_path=handseg_path, type='train', train_size=0.8, )
    for batch_images, batch_bboxes in dataset.batch_iterator:
        ones = np.ones(batch_images.shape)
        zeros = np.zeros(batch_images.shape)
        mask = np.where(batch_images > 0, ones, zeros)
        print(batch_images[mask == 1])
        print(np.min(batch_images), np.max(batch_images))
        break


if __name__ == '__main__':
    base_path = '../../../'
    handseg_path = base_path + "datasets/handseg150k"
    # depth_hist()
    # analyse_pixel_distance()
    # dataset = HandsegDatasetBboxes(batch_size=5)
    # show_images_with_bboxes(dataset)
    # generate_bounding_boxes(print_images=False, save_to_file=False,
    #                        bboxes_filename='bounding_boxes.txt')
    show_images_from_handseg_dataset(num=3, dataset_path=handseg_path,
                                     save_fig_loc=base_path + 'documentation/images/handseg_samples.png')
