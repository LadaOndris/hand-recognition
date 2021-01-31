import tensorflow as tf
import os


class HandsegDatasetBboxes:

    def __init__(self, dataset_path, train_size, batch_size=16, shuffle=True):
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")

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
        annotations_path = os.path.join(self.dataset_path, 'bounding_boxes.txt')
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        boundary_index = int(len(annotations) * self.train_size)
        return annotations[:boundary_index], annotations[boundary_index:]

    def _build_iterator(self, annotations):
        dataset = tf.data.Dataset.from_tensor_slices(annotations)
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.map(self._prepare_sample)
        shapes = (tf.TensorShape([416, 416, 1]), tf.TensorShape([None, 4]))
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    # annotation line consists of depth_image file name and bounding boxes coordinates
    @tf.function()
    def _prepare_sample(self, annotation):
        annotation = tf.strings.strip(annotation)  # remove white spaces causing FileNotFound
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])

        # depth_image = tf.keras.preprocessing.image.load_img(image_file_name, color_mode='grayscale')
        depth_image = tf.io.read_file(image_file_path)
        # loads depth images and converts values to fit in dtype.uint8
        depth_image = tf.io.decode_image(depth_image, channels=1)
        # replace outliers above value 15000
        depth_image = tf.where(depth_image < 35, depth_image, 0)

        # depth_image = tf.reshape(depth_image, shape=[480, 640, 1])
        # depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.int32)
        depth_image.set_shape([480, 640, 1])

        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)

        # resize image
        depth_image = tf.image.resize_with_pad(depth_image, 416, 416)

        # resize bboxes to [416, 416]
        m = tf.tile(tf.constant([[416 / 640, 416 / 640, 416 / 640, 416 / 640]], dtype=tf.float32), [len(bboxes), 1])
        a = tf.tile(tf.constant([[0, 52, 0, 52]], dtype=tf.float32), [len(bboxes), 1])
        bboxes = tf.math.multiply(bboxes, m)
        bboxes = tf.math.add(bboxes, a)

        return depth_image, bboxes
