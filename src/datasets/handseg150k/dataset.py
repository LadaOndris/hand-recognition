from PIL import Image
import numpy as np
import os
import glob
from src.utils.paths import HANDSEG_DATASET_DIR

HUGE_INT = 2147483647


class HandsegDataset:

    def __init__(self, dataset_path):
        self.image_shape = (480, 640)
        self.dataset_path = dataset_path
        return

    """
    Load a subset of images of the dataset.
    The subset is defined by start_index and end_index.
    
    If start_index is set to None, the subset starts at the beginning of the dataset.
    If end_index is set to None, the subset ends at the end of the dataset.
    If both set to None, selects the whole dataset. Not recommended.
    """

    def load_images(self, start_index=0, end_index=99):
        all_images_paths = glob.glob(os.path.join(self.dataset_path, 'images/*'))
        masks_paths = []

        if start_index is not None and end_index is not None:
            images_paths = all_images_paths[start_index:end_index]
        elif start_index is None and end_index is not None:
            images_paths = all_images_paths[:end_index]
        elif start_index is not None and end_index is None:
            images_paths = all_images_paths[start_index:]
        else:
            images_paths = all_images_paths

        for image_path in images_paths:
            _, filename = os.path.split(image_path)
            masks_paths.append(os.path.join(self.dataset_path, 'masks', filename))

        images = np.array([np.array(Image.open(filename)) for filename in images_paths])
        for image in images:
            image[image > 9000] = 0
        masks = [np.array(Image.open(filename)) for filename in masks_paths]
        return images, masks

    """
    Reads a depth image and its mask at the specified index.
    """

    def load_image(self, index):
        all_images_paths = glob.glob(os.path.join(self.dataset_path, 'images/*'))
        image_path = all_images_paths[index]

        _, filename = os.path.split(image_path)
        mask_path = os.path.join(self.dataset_path, 'masks', filename)

        image = np.array(Image.open(image_path))
        image[image == 0] = HUGE_INT
        mask = np.array(Image.open(mask_path))
        return image, mask


d = HandsegDataset(HANDSEG_DATASET_DIR)
d.load_image(0)
