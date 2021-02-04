import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
from src.utils.paths import BIGHAND_DATASET_DIR


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

        self.train_annotations, self.test_annotations = self._load_annotations()
        self.num_train_batches = int(len(self.train_annotations) // self.batch_size)
        self.num_test_batches = int(len(self.test_annotations) // self.batch_size)

        self.train_dataset = self._build_dataset(self.train_annotations)
        self.test_dataset = self._build_dataset(self.test_annotations)

    def _load_annotations(self):
        subject_dirs = [f.stem for f in self.dataset_path.iterdir() if f.is_dir()]
        annotation_files = []
        for subject_dir in subject_dirs:
            pattern = F"full_annotation/{subject_dir}/[!README]*.txt"
            full_pattern = os.path.join(self.dataset_path, pattern)
            annotation_files += glob.glob(full_pattern)

        print(annotation_files)
        boundary_index = int(len(annotation_files) * self.train_size)
        return annotation_files[:boundary_index], annotation_files[boundary_index:]

    def _build_dataset(self, annotation_files):
        """
        1. read files in directory - tf.data.Dataset.list_files
        2. TextLineDataset for each of these files
        3. interleave these TextLineDatasets
        4. read the depth image and return it together with labels te
        """

        """ Read specified files """
        # dataset = tf.data.Dataset.from_tensor_slices(annotations)

        """ Read all available annotations """
        # pattern = os.path.join(self.dataset_path, 'full_annotation/*/*.txt')
        # dataset = tf.data.Dataset.list_files(pattern)
        annotation_files = tf.constant(annotation_files, dtype=tf.string)
        annotation_files = tf.random.shuffle(annotation_files)

        """ Create a TextLineDataset for each annotations file """
        # datasets = tf.map_fn(lambda x: tf.data.TextLineDataset(x), annotation_files)
        """ Randomly interleave all those files """
        # dataset = tf.data.experimental.sample_from_datasets(datasets, weights=None)

        # files_dataset = tf.data.Dataset.from_tensor_slices(annotation_files)
        # dataset = files_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).map(self._prepare_sample))
        dataset = tf.data.TextLineDataset(annotation_files)
        """ Reshuffle the dataset each iteration """
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=16384, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        # shapes = (tf.TensorShape([480, 640, 1]), tf.TensorShape([None, ]))
        # dataset = dataset.padded_batch(self.batch_size, padded_shapes=([480, 640, 1], [None]))
        dataset = dataset.map(self._prepare_sample)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _prepare_sample(self, annotation_line):
        """ Each line contains 64 values: file_name, 21 (joints) x 3 (coords) """
        tf.print(annotation_line)
        splits = tf.strings.split(annotation_line, sep='\t', maxsplit=63)  # Split by whitespaces
        filename, labels = tf.split(splits, [1, 63], 0)
        """
        # If the function processes a batch
        splits = splits.to_tensor()
        filename, labels = tf.split(splits, [1, 63], 1)
        """
        image_paths = tf.strings.join([tf.constant(str(self.dataset_path)), filename], separator=os.sep)
        image_paths = tf.squeeze(image_paths)

        depth_image = tf.io.read_file(image_paths)
        image = tf.io.decode_image(depth_image, channels=1)
        image.set_shape([480, 640, 1])

        return image, labels


if __name__ == '__main__':
    ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=2)
    iterator = iter(ds.train_dataset)
    batch = next(iterator)
    print(batch)
    for image, labels in batch:
        plt.imshow(image)
        plt.show()
        print(labels)
