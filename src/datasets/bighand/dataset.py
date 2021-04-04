import tensorflow as tf
import os
import glob
from src.utils.paths import BIGHAND_DATASET_DIR
from src.utils import plots
from src.utils.camera import Camera
import matplotlib.pyplot as plt
from PIL import Image


class BighandDataset:

    def __init__(self, dataset_path, train_size, batch_size=16, shuffle=True):
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")
        if batch_size < 1:
            raise ValueError("Batch size has to be greater than 0.")

        self.dataset_path = dataset_path
        self.train_size = train_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0

        self.train_annotation_files, self.test_annotation_files = self._load_annotations()
        self.train_annotations = self._count_annotations(self.train_annotation_files)
        self.test_annotations = self._count_annotations(self.test_annotation_files)
        self.num_train_batches = int(self.train_annotations // self.batch_size)
        self.num_test_batches = int(self.test_annotations // self.batch_size)

        self.train_dataset = self._build_dataset(self.train_annotation_files)
        self.test_dataset = self._build_dataset(self.test_annotation_files)

    def _count_annotations(self, annotation_files):
        def file_lines(filename):
            with open(filename) as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        counts = [file_lines(filename) for filename in annotation_files]
        return sum(counts)

    def _load_annotations(self):
        subject_dirs = [f.stem for f in self.dataset_path.iterdir() if f.is_dir()]
        annotation_files = []
        for subject_dir in subject_dirs:
            pattern = F"full_annotation/{subject_dir}/[!README]*.txt"
            full_pattern = os.path.join(self.dataset_path, pattern)
            annotation_files += glob.glob(full_pattern)

        boundary_index = int(len(annotation_files) * self.train_size)
        return annotation_files[:boundary_index], annotation_files[boundary_index:]

    def _build_dataset_one_sample(self, annotation_files):
        file = annotation_files[0]
        tf.print(file)
        with open(file, 'r') as f:
            annotation_line = f.readline()
        line = tf.constant(annotation_line, dtype=tf.string)
        line = tf.reshape(line, shape=[1])
        ds = tf.data.Dataset.from_tensor_slices(line)
        ds = ds.repeat()
        ds = ds.map(self._prepare_sample)
        ds = ds.batch(1)
        return ds

    def _build_dataset(self, annotation_files):
        """ Read specified files """
        # dataset = tf.data.Dataset.from_tensor_slices(annotations)

        """ Read all available annotations """
        # pattern = os.path.join(self.dataset_path, 'full_annotation/*/*.txt')
        # dataset = tf.data.Dataset.list_files(pattern)

        """ Convert to Tensor and shuffle the files """
        annotation_files = tf.constant(annotation_files, dtype=tf.string)
        if self.shuffle:
            annotation_files = tf.random.shuffle(annotation_files)

        dataset = tf.data.TextLineDataset(annotation_files)
        """ Reshuffle the dataset each iteration """
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=131072, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._prepare_sample)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _prepare_sample(self, annotation_line):
        """ Each line contains 64 values: file_name, 21 (joints) x 3 (coords) """

        """ If the function processes a single line """
        splits = tf.strings.split(annotation_line, sep='\t', maxsplit=63)  # Split by whitespaces
        filename, labels = tf.split(splits, [1, 63], 0)
        joints = tf.strings.to_number(labels, tf.float32)
        joints = tf.reshape(joints, [21, 3])
        """
        # If the function processes a batch
        splits = splits.to_tensor()
        filename, labels = tf.split(splits, [1, 63], 1)
        """

        """ Compose a full path to the image """
        image_paths = tf.strings.join([tf.constant(str(self.dataset_path)), filename], separator=os.sep)
        """ Squeeze the arrays dimension if necessary"""
        image_paths = tf.squeeze(image_paths)

        """ Read and decode image (tf doesn't support for more than a single image)"""
        depth_image = tf.io.read_file(image_paths)
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        depth_image.set_shape([480, 640, 1])

        reorder_idx = tf.constant([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20
        ], dtype=tf.int32)
        joints = tf.gather(joints, reorder_idx)
        # tf.print("annot", annotation_line)
        # tf.print("joint", joints)
        return depth_image, joints


if __name__ == '__main__':
    cam = Camera('bighand')
    ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=10, shuffle=True)
    iterator = iter(ds.train_dataset)
    batch_images, batch_labels = next(iterator)

    for image, joints in zip(batch_images, batch_labels):
        image = tf.squeeze(image)
        joints2d = cam.world_to_pixel(joints)
        plots.plot_joints_2d(image, joints2d)
        pass
