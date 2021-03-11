import tensorflow as tf
import os
import glob
from src.utils.paths import BIGHAND_DATASET_DIR
from src.utils import plots
from src.utils.camera import Camera


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

        boundary_index = int(len(annotation_files) * self.train_size)
        return annotation_files[:boundary_index], annotation_files[boundary_index:]

    def _build_dataset(self, annotation_files):
        """ Read specified files """
        # dataset = tf.data.Dataset.from_tensor_slices(annotations)

        """ Read all available annotations """
        # pattern = os.path.join(self.dataset_path, 'full_annotation/*/*.txt')
        # dataset = tf.data.Dataset.list_files(pattern)

        """ Convert to Tensor and shuffle the files """
        annotation_files = tf.constant(annotation_files, dtype=tf.string)
        annotation_files = tf.random.shuffle(annotation_files)

        dataset = tf.data.TextLineDataset(annotation_files)
        """ Reshuffle the dataset each iteration """
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=16384, reshuffle_each_iteration=True)
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
        tf.print("annot", annotation_line)
        tf.print("joint", joints)
        return depth_image, joints


if __name__ == '__main__':
    """
    annot "Subject_1/ego/image_D00000956.png\t57.3375\t12.5278\t358.6355\t67.0202\t13.6015\t368.7855\t24.5080\t-20.8626\t416.0751\t10.6481\t-23.4054\t396.6474\t6.0920\t-20.5528\t378.4237\t0.3275\t-18.8273\t361.4121\t61.1928\t-40.3427\t382.2544\t64.2384\t-63.7942\t405.4462\t64.5719\t-86.9980\t420.0651\t13.1114\t-66.1421\t400.7969\t5.9757\t-94.4926\t391.2309\t1.2620\t-113.8722\t387.9937\t-1.1292\t-70.6526\t385.2515\t-8.5380\t-100.3752\t378.0826\t-12.0724\t-121.9319\t375.7346\t-22.9689\t-58.4097\t380.0795\t-41.8077\t-82.9507\t381.1528\t-48.1331\t-103.0207\t383.8523\t-29.3093\t-48.2683\t362.1296\t-45.1400\t-63.9945\t362.5128\t-56.7198\t-80.5820\t369.2659"
    joint [[57.3375 12.5278 358.635498]
     [67.0202 13.6015 368.785492]
     [61.1928 -40.3427 382.254395]
     ...
     [-29.3093 -48.2683 362.129608]
     [-45.14 -63.9945 362.512787]
     [-56.7198 -80.582 369.2659]]
    """
    cam = Camera('bighand')
    from PIL import Image
    #
    # img = Image.open(os.path.join(BIGHAND_DATASET_DIR, 'Subject_1/ego/image_D00004312.png'))
    # plt.imshow(img)
    # str = "52.6535\t17.4918\t244.7902\t53.4700\t10.5999\t232.5524\t-7.5528\t-25.6909\t246.3567\t-7.9223\t-11.1448\t265.4427\t-0.7686\t2.2768\t276.8307\t4.5669\t14.8456\t288.6272\t-0.5667\t11.4204\t213.3530\t-34.0246\t11.9284\t201.4653\t-54.3765\t6.6205\t183.8623\t-53.0614\t-32.1009\t256.1785\t-79.4255\t-35.8143\t261.8684\t-98.8141\t-40.6814\t264.8109\t-53.3633\t-16.1438\t287.2915\t-82.5084\t-19.3500\t301.3050\t-102.7916\t-23.0986\t308.8707\t-32.0517\t32.4398\t292.7742\t-9.4679\t34.0019\t275.2228\t4.7097\t42.8654\t262.1639\t-18.2654\t11.7553\t325.9233\t-31.7898\t9.9248\t348.0151\t-46.1367\t5.7294\t363.2272"
    # jnts = tf.strings.split(tf.constant(str, tf.string), sep='\t', maxsplit=63)
    # jnts = tf.strings.to_number(jnts, tf.float32)
    # jnts = tf.reshape(jnts, [21, 3])
    # jnts2d = cam.world_to_pixel(jnts)
    # plt.scatter(jnts2d[..., 0], jnts2d[..., 1], c='r', marker='o', s=20)
    # plt.show()

    ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=10)
    iterator = iter(ds.train_dataset)
    batch_images, batch_labels = next(iterator)

    for image, joints in zip(batch_images, batch_labels):
        image = tf.squeeze(image)
        # plots.plot_depth_image(image)
        joints2d = cam.world_to_pixel(joints)
        plots.plot_joints_2d(image, joints2d)
        pass
