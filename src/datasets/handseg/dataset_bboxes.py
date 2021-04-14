import tensorflow as tf
import os


class HandsegDatasetBboxes:

    def __init__(self, dataset_path, train_size, model_input_shape, batch_size=16, shuffle=True):
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")

        self.dataset_path = str(dataset_path)
        self.train_size = train_size
        self.model_input_shape = model_input_shape
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
        if self.shuffle:
            dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
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

        depth_image.set_shape([480, 640, 1])
        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)

        depth_image, bboxes = self.crop(depth_image, bboxes)
        # depth_image, bboxes = self.pad(depth_image, bboxes)
        return depth_image, bboxes

    def crop(self, depth_image, bboxes):
        depth_image = depth_image[tf.newaxis, ...]
        depth_image = tf.image.crop_and_resize(depth_image, [[0, 80 / 640.0, 480 / 480.0, 560 / 640.0]],
                                               [0], self.model_input_shape[:2])
        depth_image = depth_image[0]
        # crop bboxes
        bboxes = bboxes - tf.constant([80, 0, 80, 0], dtype=tf.float32)[tf.newaxis, :]
        # crop out of bounds boxes
        bboxes = tf.where(bboxes < 0., 0., bboxes)
        bboxes = tf.where(bboxes >= 480., 479., bboxes)
        # remove too narrow boxes because of the crop
        bboxes_mask_indices = tf.where(bboxes[..., 2] - bboxes[..., 0] > 5.)
        bboxes = tf.gather_nd(bboxes, bboxes_mask_indices)
        # resize bboxes
        bboxes *= self.model_input_shape[0] / 480
        return depth_image, bboxes

    def pad(self, depth_image, bboxes):
        depth_image = tf.image.resize_with_pad(depth_image, 416, 416)
        m = tf.tile(tf.constant([[416 / 640, 416 / 640, 416 / 640, 416 / 640]], dtype=tf.float32),
                    [tf.shape(bboxes)[0], 1])
        a = tf.tile(tf.constant([[0, 52, 0, 52]], dtype=tf.float32), [tf.shape(bboxes)[0], 1])
        bboxes = tf.math.multiply(bboxes, m)
        bboxes = tf.math.add(bboxes, a)
        return depth_image, bboxes
