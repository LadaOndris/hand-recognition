import tensorflow as tf
import numpy as np
import src.utils.plots as plots

from src.utils.paths import CUSTOM_DATASET_DIR


class CustomDataset:

    def __init__(self, dataset_path, batch_size, shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        sequence_annotations = self._get_sequence_annotations()
        filepaths, labels = self._get_line_annotations(sequence_annotations)
        self.num_batches = int(len(filepaths) // batch_size)
        self.dataset = self._build_dataset(filepaths, labels)

    def _get_sequence_annotations(self):
        path = self.dataset_path.joinpath('annotations.txt')
        annotations = np.genfromtxt(path, delimiter=' ', dtype=str)
        return annotations

    def _get_line_annotations(self, annotations):
        folder_names = annotations[..., 0]
        gesture_label = annotations[..., 1]
        file_names = []
        file_labels = []
        for i in range(annotations.shape[0]):
            folder_path = self.dataset_path.joinpath(folder_names[i])
            files = np.array(list(folder_path.iterdir()), dtype=str)
            labels = np.full(shape=[files.shape[0]], fill_value=gesture_label[i])
            file_names.append(files)
            file_labels.append(labels)
        file_names = np.concatenate(file_names)
        file_labels = np.concatenate(file_labels)
        return file_names, file_labels

    def _build_dataset(self, images, labels):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
        ds = ds.map(self._prepare_sample)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=1)
        return ds

    def _prepare_sample(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=1, dtype=tf.uint16)
        image.set_shape([480, 640, 1])
        return image, label


if __name__ == '__main__':
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=4, shuffle=True)
    iterator = iter(ds.dataset)
    batch_images, batch_labels = next(iterator)

    for image, label in zip(batch_images, batch_labels):
        image = tf.squeeze(image)
        plots.plot_depth_image(image)
        print(label)
