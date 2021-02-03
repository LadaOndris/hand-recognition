import tensorflow as tf
import os
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

        self.train_batch_iterator = self._build_iterator(self.train_annotations)
        self.test_batch_iterator = self._build_iterator(self.test_annotations)

    def _load_annotations(self):
        """
        1. read files in directory - tf.data.Dataset.list_files
        2. TextLineDataset for each of these files
        3. interleave these TextLineDatasets
        4. read the depth image and return it together with labels te
        """
        first = 'full_annotation/Subject_1/76 150_loc_shift_made_by_qi_20180112_v2.txt'
        annotations_path = os.path.join(self.dataset_path, first)
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        boundary_index = int(len(annotations) * self.train_size)
        return annotations[:boundary_index], annotations[boundary_index:]

    def _build_iterator(self, annotations):
        """ Read specified files """
        # dataset = tf.data.Dataset.from_tensor_slices(annotations)

        """ Read all available annotations """
        # pattern = os.path.join(self.dataset_path, 'full_annotation/*/*.txt')
        # dataset = tf.data.Dataset.list_files(pattern)

        """ Create a TextLineDataset for each annotations file """
        datasets = tf.map_fn(lambda x: tf.data.TextLineDataset(x), annotations)
        """ Randomly interleave all those files """
        dataset = tf.data.experimental.sample_from_datasets(datasets, weights=None)
        """ Reshuffle the dataset each iteration """
        if self.shuffle:
            dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        shapes = (tf.TensorShape([480, 640, 1]), tf.TensorShape([None, ]))
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.map(self._prepare_sample)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _prepare_sample(self, annotation_line):
        """ Each line contains 65 values: file_name, 21 (joints) x 3 (coords) """
        splits = tf.strings.split(annotation_line) # Split by whitespaces
        filename = splits[:, 0]
        labels = splits[:, 1:]
        image = tf.io.decode_image(filename)
        return image, labels

if __name__ == '__main__':
    ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1)
    batch = next(ds.train_batch_iterator)
    for image, labels in batch:
        plt.imshow(image)
        plt.show()
        print(labels)