import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.estimation.configuration import Config
from src.estimation.preprocessing_com import ComPreprocessor
from src.utils.camera import Camera
from src.utils.imaging import resize_bilinear_nearest_batch, resize_images


class DatasetPreprocessor:
    """
    DatasetGenerator takes an iterator of a hand pose dataset
    as its input, and preprocesses the joints coordinates
    to match the expected output of JGR-J2O network.
    """

    def __init__(self, dataset_iterator, image_in_size, image_out_size, camera: Camera, config: Config):
        """
        Parameters
        ----------
        dataset_iterator
        depth_image_size    (Width, Height)
        image_in_size
        image_out_size
        camera
        return_xyz
        dataset_includes_bboxes
            If dataset returns bboxes, then the images are expected to be already cropped.
        """
        self.iterator = dataset_iterator
        self.image_in_size = [image_in_size, image_in_size]
        self.image_out_size = image_out_size
        self.camera = camera
        self.return_xyz = config.return_xyz
        self.augment = config.augment
        self.cube_size = (config.cube_size, config.cube_size, config.cube_size)
        self.thresholding = config.thresholding
        self.refine_iters = config.refine_iters
        self.bboxes = None  # These are used to position the cropped coordinates into global picture
        self.resize_coeffs = None  # These are used to invert image resizing
        self.cropped_imgs = None
        self.com_preprocessor = ComPreprocessor(self.camera, config.thresholding, config.use_center_of_image,
                                                config.ignore_threshold_otsus)

    def __iter__(self):
        return self

    def postprocess(self, y_pred):
        """
        Denormalizes predicted coordinates.
        Applies inverted resizing.
        Puts the coordinates in the global picture by
        adding bounding box offsets.

        Returns
        -------
            Global UVZ coordinates.
        """
        normalized_uvz = y_pred[0]
        normalized_uv = normalized_uvz[..., :2]
        normalized_z = normalized_uvz[..., 2:3]

        resized_uv = normalized_uv * self.image_in_size
        global_z = normalized_z * self.cube_size[2] + self.bcubes[..., tf.newaxis, 2:3]
        uv_local = 1.0 / self.resize_coeffs[:, tf.newaxis, :] * resized_uv
        uv_global = uv_local + tf.cast(self.bboxes[:, tf.newaxis, :2], dtype=tf.float32)
        uvz_global = tf.concat([uv_global, global_z], axis=-1)
        return uvz_global

    def preprocess(self, images, bboxes):
        images = tf.cast(images, tf.float32)
        self.bcubes = self.com_preprocessor.refine_bcube_using_com(images, bboxes,
                                                                   cube_size=self.cube_size,
                                                                   refine_iters=self.refine_iters)
        # path = DOCS_DIR.joinpath('figures/design/bounding_cube1.png')
        # plots.plot_bounding_cube(images[0], self.bcubes[0], self.camera, fig_location=path)
        # Crop the area defined by bcube from the orig image
        self.cropped_imgs = self.com_preprocessor.crop_bcube(images, self.bcubes)
        if self.thresholding:
            self.cropped_imgs = self.com_preprocessor.apply_otsus_thresholding(
                self.cropped_imgs,
                plot_image_after_thresholding=False,
                plot_image_before_thresholding=False,
                plot_histogram=False)
        self.cropped_imgs = self.clear_hand(self.cropped_imgs)  # find countours
        self.bcubes = tf.cast(self.bcubes, tf.float32)
        self.bboxes = tf.concat([self.bcubes[..., 0:2], self.bcubes[..., 3:5]], axis=-1)
        resized_imgs = resize_images(self.cropped_imgs, target_size=self.image_in_size)
        resized_imgs = tf.where(resized_imgs < self.bcubes[..., tf.newaxis, tf.newaxis, 2:3], 0, resized_imgs)
        self.resize_coeffs = self.get_resize_coeffs(self.bboxes, target_size=self.image_in_size)
        self.normalized_imgs = self.normalize_imgs(resized_imgs, self.bcubes)
        return self.normalized_imgs

    def __next__(self):
        """
        The dataset iterator returns images of arbitrary size -
        either original size, e.g. 640x480 (BigHand dataset), or cropped images using
        the returned bounding boxes (MSRA dataset).
        These images are cropped if no bounding boxes are returned from
        the iterator.
        Then, they are resized to shape (96, 96), which is the
        input shape of the JGR-J20 network.
        The dataset iterator should return joint coordinates in global
        coordinate system (XYZ). They are also cropped, resized.
        Both, images and coordinates are normalized to range [0, 1].
        Also, offsets are computed from the normalized coordinates.
        """
        images, self.xyz_global = self.iterator.get_next()
        uv_global = self.camera.world_to_pixel(self.xyz_global)[..., :2]

        # Rotate in plane
        if self.augment:
            images, uv_global = self.augment_batch(images, uv_global)
        images = tf.cast(images, tf.float32)

        self.bboxes = self.extract_bboxes(uv_global)
        self.bboxes = self.square_bboxes(self.bboxes, tf.shape(images)[1:3])
        self.bcubes = self.com_preprocessor.refine_bcube_using_com(images, self.bboxes,
                                                                   cube_size=self.cube_size)
        if self.augment:
            # Translate 3D - move bcubes
            self.bcubes = self.bcubes_translate3d(self.bcubes, stddev=8)
            # Scale 3D - increase or descrease the size of bcubes
            self.bcubes = self.bcubes_scale3d(self.bcubes, stddev=0.08)

        # Crop the area defined by bcube from the orig image
        self.cropped_imgs = self.com_preprocessor.crop_bcube(images, self.bcubes)
        if self.thresholding:
            self.cropped_imgs = self.com_preprocessor.apply_otsus_thresholding(
                self.cropped_imgs,
                plot_image_after_thresholding=False,
                plot_image_before_thresholding=False,
                plot_histogram=False)
        self.bcubes = tf.cast(self.bcubes, tf.float32)
        self.bboxes = tf.concat([self.bcubes[..., :2], self.bcubes[..., 3:5]], axis=-1)

        # Resize images and remove values out of bcubes caused by billinear resizing
        resized_imgs = resize_images(self.cropped_imgs, target_size=self.image_in_size)
        resized_imgs = tf.where(resized_imgs < self.bcubes[..., tf.newaxis, tf.newaxis, 2:3], 0, resized_imgs)

        uv_cropped = uv_global - tf.cast(self.bboxes[:, tf.newaxis, :2], dtype=tf.float32)
        self.resize_coeffs = self.get_resize_coeffs(self.bboxes, target_size=self.image_in_size)
        resized_uv = self.resize_coords(uv_cropped, self.resize_coeffs)

        z = self.xyz_global[..., 2:3]
        self.normalized_imgs = self.normalize_imgs(resized_imgs, self.bcubes)
        normalized_uvz = self.normalize_coords(resized_uv, z, self.bcubes)
        offsets = self.compute_offsets(self.normalized_imgs, normalized_uvz)
        return self.normalized_imgs, [normalized_uvz, offsets]

    def clear_hand(self, images):

        cleared = tf.map_fn(self.find_largest_countour, elems=images,
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, 1], dtype=images.dtype))
        return cleared

    def find_largest_countour(self, image):
        image_tensor = image.to_tensor()
        image_casted_uint16 = tf.cast(image_tensor, dtype=tf.uint16).numpy()
        # image_casted = tf.cast(image_casted_uint16, dtype=tf.uint8).numpy()
        # image_casted = tf.image.convert_image_dtype(image_casted_uint16, dtype=tf.uint8).numpy()
        # image_casted = cv2.convertScaleAbs(image_casted_uint16, alpha=(255.0 / 65535.0))
        image_casted = image_casted_uint16

        # plot_image_comparison(image_casted_uint16, image_casted, 'uint8')
        # Use morphological closing to connect disconnected fingers
        # caused by a depth camera
        image_morph_closed = cv2.morphologyEx(image_casted, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # image_morph_opened = cv2.morphologyEx(image_casted, cv2.MORPH_OPEN,
        #                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # plot_image_comparison(image_casted, image_morph_closed, 'closing')
        # plot_image_comparison(image_casted, image_morph_opened, 'opening')
        # plots.plot_depth_image(image_casted, DOCS_DIR.joinpath('figures/design/largest_contour_original.png'))
        # plots.plot_depth_image(image_morph_closed, DOCS_DIR.joinpath('figures/design/largest_contour_after_morphing.png'))

        # Convert dtype to uint8
        image_morph_closed = cv2.convertScaleAbs(image_morph_closed)

        # Find largest contour in intermediate image
        cnts, _ = cv2.findContours(image_morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)

        # Output
        out = np.zeros(image_casted.shape, np.uint8)
        cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)

        out_tensor = tf.convert_to_tensor(out)
        mask = out_tensor > 0
        cleared_img = tf.where(mask, image_tensor, 0)

        # plots.plot_depth_image(out_tensor, DOCS_DIR.joinpath('figures/design/largest_contour_mask.png'))
        # plots.plot_depth_image(cleared_img, DOCS_DIR.joinpath('figures/design/largest_contour_result.png'))

        # cleared_img = tf.convert_to_tensor(cleared_img[..., np.newaxis])
        # cleared_img = tf.image.convert_image_dtype(cleared_img, tf.uint16)
        # cleared_img = tf.cast(cleared_img, tf.float32)
        return tf.RaggedTensor.from_tensor(cleared_img, ragged_rank=2)

    def bcubes_translate3d(self, bcubes, stddev):
        """
        Translates bounding cubes. Moves the XYZ coordinates
        of the bounding cube by a value given by a normal
        distribution N(0, stddev),

        Parameters
        ----------
        bcubes
        stddev : float
            The standard deviation of the normal distribution.

        Returns
        -------

        """
        translation_offset = tf.random.truncated_normal(shape=[bcubes.shape[0], 3], mean=0,
                                                        stddev=stddev)  # shape [batch_size, 3]
        translation_offset = tf.cast(translation_offset, dtype=tf.int32)
        bcubes_translation = tf.concat([translation_offset, translation_offset], axis=-1)  # shape [batch_size, 6]
        bcubes_translated = bcubes + bcubes_translation
        return bcubes_translated

    def bcubes_scale3d(self, bcubes, stddev):
        """
        Scales the size of the bounding cubes.
        The cube is shrinked or enlarged with a normal distribution N(1, stddev).

        Parameters
        ----------
        bcubes : tf.Tensor of shape [batch_size, 6]
        stddev : float
            The standard deviation of the normal distribution.

        Returns
        -------
        scaled_bcubes tf.Tensor of shape [batch_size, 6]
            A scaled tensor of bcubes.
        """
        scales = tf.random.truncated_normal(shape=[bcubes.shape[0], 1], mean=0, stddev=stddev)
        half_cube_size = tf.constant(self.cube_size, dtype=tf.float32) / 2
        shift = half_cube_size * scales  # shape [batch_size, 3]
        shift = tf.cast(shift, dtype=tf.int32)
        shift_both_directions = tf.concat([shift, -shift], axis=-1)  # shape [batch_size, 6]
        bcubes_scaled = bcubes + shift_both_directions
        return bcubes_scaled

    def get_resize_coeffs(self, bboxes, target_size):
        cropped_imgs_sizes_u = bboxes[..., 2] - bboxes[..., 0]  # right - left
        cropped_imgs_sizes_v = bboxes[..., 3] - bboxes[..., 1]  # bottom - top
        cropped_imgs_sizes = tf.stack([cropped_imgs_sizes_u, cropped_imgs_sizes_v], axis=-1)
        return tf.cast(target_size / cropped_imgs_sizes, dtype=tf.float32)

    def resize_coords(self, coords_uv, resize_coeffs):
        resized_uv = resize_coeffs[:, tf.newaxis, :] * coords_uv
        return resized_uv

    def normalize_imgs(self, images, bcubes):
        z_cube_size = (bcubes[..., tf.newaxis, 5:6] - bcubes[..., tf.newaxis, 2:3])
        normalized_imgs = (images - bcubes[..., tf.newaxis, tf.newaxis, 2:3])
        # Ignore negative values - because it was the background of zeroes.
        normalized_imgs = (tf.math.abs(normalized_imgs) + normalized_imgs) / 2
        normalized_imgs = normalized_imgs / z_cube_size[..., tf.newaxis]
        return normalized_imgs

    def normalize_coords(self, joints_uv, joints_z, bcubes):
        z_cube_size = (bcubes[..., tf.newaxis, 5:6] - bcubes[..., tf.newaxis, 2:3])
        normalized_uv = joints_uv / self.image_in_size
        normalized_z = (joints_z - bcubes[..., tf.newaxis, 2:3]) / z_cube_size
        normalized_uvz = tf.concat([normalized_uv, normalized_z], axis=-1)
        return normalized_uvz

    def augment_batch(self, images, uv_joints):
        images = tf.cast(images, dtype=tf.float32)
        im_height, im_width = images.shape[1], images.shape[2]
        image_center = [im_width / 2, im_height / 2]
        images, joints = self.rotate(images, uv_joints, image_center)
        return images, joints

    def rotate(self, images, uv_joints, image_center):
        degrees = tf.random.uniform(shape=[tf.shape(images)[0]], minval=-180, maxval=180)
        radians = np.deg2rad(degrees)
        # Transposed rotation matrix
        rotation_matrix = np.array([[np.cos(radians), np.sin(radians)],
                                    [-np.sin(radians), np.cos(radians)]])
        new_images = tfa.image.rotate(images, radians)
        # Set the image center at [96/2, 96/2] and rotate around the center
        uv_joints = uv_joints - image_center
        new_joints = tf.matmul(tf.transpose(rotation_matrix, [2, 0, 1]), tf.transpose(uv_joints, [0, 2, 1]))
        new_joints = tf.transpose(new_joints, [0, 2, 1])
        # Set the center at [0, 0] of the image
        new_joints = new_joints + image_center
        return new_images, new_joints

    def extract_bboxes(self, uv_global):
        """
        Parameters
        ----------
        images      tf.Tensor of shape [None, height, width]
        uv_global   tf.Tensor of shape [None, n_joints, 2]

        Returns
        -------
            tf.Tensor of shape [None, 4].
            The four values are [left, top, right, bottom].
        """
        # Extract u and v coordinates
        u = uv_global[..., 0]
        v = uv_global[..., 1]

        # Find min and max over joints axis.
        u_min, u_max = tf.reduce_min(u, axis=1), tf.reduce_max(u, axis=1)
        v_min, v_max = tf.reduce_min(v, axis=1), tf.reduce_max(v, axis=1)

        # Move min and max to make the bbox slighty larger
        shift_coeff = 0.2
        width = u_max - u_min
        height = v_max - v_min
        u_shift = width * shift_coeff
        v_shift = height * shift_coeff
        u_min, u_max = u_min - u_shift, u_max + u_shift
        v_min, v_max = v_min - v_shift, v_max + v_shift

        # The bounding box is represented by four coordinates
        bboxes = tf.stack([u_min, v_min, u_max, v_max], axis=-1)
        bboxes = tf.cast(bboxes, dtype=tf.int32)
        return bboxes

    def square_bboxes(self, bboxes, image_size):
        u_min, v_min, u_max, v_max = tf.unstack(bboxes, axis=-1)
        width = u_max - u_min
        height = v_max - v_min

        # Make bbox square in the U axis
        u_min_new = u_min + tf.cast(width / 2 - height / 2, dtype=u_min.dtype)
        u_max_new = u_min + tf.cast(width / 2 + height / 2, dtype=u_min.dtype)
        u_min_raw = tf.where(width < height, u_min_new, u_min)
        u_max_raw = tf.where(width < height, u_max_new, u_max)
        # Crop out of bounds
        u_min = tf.math.maximum(u_min_raw, 0)
        u_max = tf.math.minimum(u_max_raw, image_size[0])

        # Make bbox square in the V axis
        v_min_new = v_min + tf.cast(height / 2 - width / 2, dtype=v_min.dtype)
        v_max_new = v_min + tf.cast(height / 2 + width / 2, dtype=v_min.dtype)
        v_min_raw = tf.where(height < width, v_min_new, v_min)
        v_max_raw = tf.where(height < width, v_max_new, v_max)
        # Crop out of bounds
        v_min = tf.math.maximum(v_min_raw, 0)
        v_max = tf.math.minimum(v_max_raw, image_size[1])

        return tf.stack([u_min, v_min, u_max, v_max], axis=-1)

    def compute_offsets(self, images, joints):
        """
            Computes offsets for each joint coordinate.
            All offsets are normalized to [-1, 1] if the given
            UVZ coords are normalized to [0, 1], and if the
            input images are also normalized to range [0, 1].

            !!! Offsets have to be computed from cropped resized
            normalized coordinates. !!!
        Parameters
        ----------
        joints  tf.Tensor of shape [None, n_joints, 3]

        Returns
        -------
            tf.Tensor of shape [None, out_size, out_size, n_joints, 3]
        """
        n_joints = joints.shape[1]
        x = self.offset_coord(joints[..., 0], n_joints)  # shape = [None, out_size, n_joints]
        y = self.offset_coord(joints[..., 1], n_joints)  # shape = [None, out_size, n_joints]

        u_offsets = x[:, tf.newaxis, :, :]
        u_offsets = tf.tile(u_offsets, [1, self.image_out_size, 1, 1])

        v_offsets = y[:, :, tf.newaxis, :]
        v_offsets = tf.tile(v_offsets, [1, 1, self.image_out_size, 1])

        # z_im = tf.image.resize(images, [self.image_out_size, self.image_out_size],
        #                        method=tf.image.ResizeMethod.BILINEAR)
        z_im = resize_bilinear_nearest_batch(images, [self.image_out_size, self.image_out_size])

        z_coords = joints[..., 2]
        z_coords = z_coords[:, tf.newaxis, tf.newaxis, :]
        z_offsets = z_coords - z_im
        return tf.stack([u_offsets, v_offsets, z_offsets], axis=-1)

    def offset_coord(self, joints_single_coord, n_joints):
        x = tf.linspace(0, 1, num=self.image_out_size)  # [0, 1]
        x = tf.cast(x, tf.float32)
        x = x[tf.newaxis, :, tf.newaxis]  # shape = [1, out_size, 1]
        x = tf.tile(x, [1, 1, n_joints])  # shape = [1, out_size, n_joints]
        x -= tf.expand_dims(joints_single_coord, 1)  # [0 - u, 1 - u], shape = [None, out_size, n_joints]
        x *= -1  # [u, u - 1]
        return x

    def convert_coords_to_local(self, uvz_coords):
        uv_local = uvz_coords[..., :2] - self.bboxes[..., tf.newaxis, :2]
        return uv_local
