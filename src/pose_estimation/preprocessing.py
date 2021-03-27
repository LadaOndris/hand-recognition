import tensorflow as tf
from src.utils.camera import Camera
import matplotlib.pyplot as plt


class ComPreprocessor:

    def __init__(self, camera: Camera):
        self.camera = camera

    def refine_bcube_using_com(self, full_image, bbox, refine_iters=3, cube_size=(250, 250, 250)):
        """
        Refines the bounding box of the detected hand
        by iteratively finding its center of mass and
        cropping it in all three axes.

        Parameters
        ----------
        full_image
        bbox
        refine_iters
        cube_size

        Returns
        -------
        Tuple having (cropped_images, bcubes)
            cropped_images is a tf.RaggedTensor(shape=[batch_size, None, None, 1])
            bcubes is a tf.Tensor(shape=[batch_size, 6])
        """
        cropped = self.crop_bbox(full_image, bbox)
        coms = self.compute_coms(cropped, offsets=bbox[..., :2])
        # plt.imshow(full_image[0])
        # plt.scatter(coms[0, 0], coms[0, 1])
        # plt.show()
        coms = self.refine_coms(full_image, coms, iters=refine_iters, cube_size=cube_size)

        # Get the cube in UVZ around the center of mass
        bcube = self.com_to_bcube(coms, size=cube_size)
        return bcube

    def compute_coms(self, images, offsets):
        """
        Calculates the center of mass of the given image.
        Does not take into account the actual values of the pixels,
        but rather treats the pixels as either background, or something.

        Parameters
        ----------
        images : tf.RaggedTensor of shape [batch_size, None, None, 1]
            A batch of images.

        Returns
        -------
        center_of_mass : tf.Tensor of shape [batch_size, 3]
            Represented in UVZ coordinates.
        """
        com_local = tf.map_fn(self.center_of_mass, images,
                              fn_output_signature=tf.TensorSpec(shape=[3], dtype=tf.float32))

        # Adjust the center of mass coordinates to orig image space (add U, V offsets)
        com_uv_global = com_local[..., :2] + tf.cast(offsets, tf.float32)
        com_z = com_local[..., 2:3]
        com_z = tf.where(tf.experimental.numpy.isclose(com_z, 0.), 300, com_z)
        coms = tf.concat([com_uv_global, com_z], axis=-1)
        return coms

    def center_of_mass(self, image):
        """
        Calculates the center of mass of the given image.
        Does not take into account the actual values of the pixels,
        but rather treats the pixels as either background, or something.

        Parameters
        ----------
        image : tf.Tensor of shape [width, height, 1]

        Returns
        -------
        center_of_mass : tf.Tensor of shape [3]
            Represented in UVZ coordinates.
            Returns [0,0,0] for zero-sized image, which happens for cropped images .

        """
        if type(image) is tf.RaggedTensor:
            image = image.to_tensor()
        # Create all coordinate pairs
        im_width, im_height = tf.shape(image)[:2]
        x = tf.range(im_width)
        y = tf.range(im_height)
        xx, yy = tf.meshgrid(x, y, indexing='ij')
        xx = tf.reshape(xx, [-1])
        yy = tf.reshape(yy, [-1])
        # Stack along a new axis to create pairs in the last dimension
        coords = tf.stack([xx, yy], axis=-1)
        coords = tf.cast(coords, tf.float32)  # [im_width * im_height, 2]

        image_mask = tf.cast(image > 0, dtype=tf.float32)
        image_mask_flat = tf.reshape(image_mask, [im_width * im_height, 1])
        # The total mass of the depth
        total_mass = tf.reduce_sum(image)
        total_mass = tf.cast(total_mass, dtype=tf.float32)
        nonzero_pixels = tf.math.count_nonzero(image_mask, dtype=tf.float32)
        # Multiply the coords with volumes and reduce to get UV coords
        volumes_vu = tf.reduce_sum(image_mask_flat * coords, axis=0)
        volumes_uvz = tf.stack([volumes_vu[1], volumes_vu[0], total_mass], axis=0)
        com_uvz = tf.math.divide_no_nan(volumes_uvz, nonzero_pixels)
        return com_uvz

    def refine_coms(self, full_image, com, iters, cube_size):
        for i in range(iters):
            # Get the cube in UVZ around the center of mass
            bcube = self.com_to_bcube(com, size=cube_size)
            # fig, ax = plt.subplots()
            # ax.imshow(full_image[0])
            # ax.scatter(com[0, 0], com[0, 1])
            # r = patches.Rectangle(bcube[0, :2], bcube[0, 3] - bcube[0, 0], bcube[0, 4] - bcube[0, 1],
            # facecolor='none', edgecolor='r', linewidth=2)
            #
            # ax.add_patch(r)
            # plt.show()
            # Crop the area defined by bcube from the orig image
            cropped = self.crop_bcube(full_image, bcube)
            # plt.imshow(cropped[0].to_tensor())
            # plt.show()
            # Compute center of mass again from the new cropped image
            com = self.compute_coms(cropped, offsets=bcube[..., :2])
        return com

    def com_to_bcube(self, com, size):
        """
        For the given center of mass (UVZ),
        computes a bounding cube in UVZ coordinates.

        Projects COM to the world coordinates,
        adds size offsets and projects back to image coordinates.

        Parameters
        ----------
        com : Center of mass
            Z coordinate cannot be zero, otherwise projection fails.
        size : Size of the bounding cube
        """
        com_xyz = self.camera.pixel_to_world(com)
        half_size = tf.constant(size, dtype=tf.float32) / 2
        # Do not subtact Z coordinate yet
        # The Z coordinates must stay the same for both points
        # in order for the projection to image plane to be correct
        half_size_xy = tf.stack([half_size[0], half_size[1], 0], axis=0)
        bcube_start_xyz = com_xyz - half_size_xy
        bcube_end_xyz = com_xyz + half_size_xy
        bcube_start_uv = self.camera.world_to_pixel(bcube_start_xyz)[..., :2]
        bcube_end_uv = self.camera.world_to_pixel(bcube_end_xyz)[..., :2]
        bcube_start_z = com[..., 2:3] - half_size[2]
        bcube_end_z = com[..., 2:3] + half_size[2]
        bcube = tf.concat([bcube_start_uv, bcube_start_z, bcube_end_uv, bcube_end_z], axis=-1)
        return tf.cast(bcube, dtype=tf.int32)

    def crop_bcube(self, images, bcubes):
        """
        Crops the image using a bounding cube. It is
        similar to cropping with a bounding box, but
        a bounding cube also defines the crop in Z axis.
        Pads the cropped area on out of bounds.


        Parameters
        ----------
        image  Image to crop from.
        bcube  Bounding cube in UVZ coordinates.
             Zero sized bcubes produce errors.

        Returns
        -------
            Cropped image as defined by the bounding cube.
        """

        def crop(elems):
            image, bcube = elems
            x_start, y_start, z_start, x_end, y_end, z_end = bcube
            # Modify bcube because it is invalid to index with negatives.
            x_start_bound = tf.maximum(x_start, 0)
            y_start_bound = tf.maximum(y_start, 0)
            x_end_bound = tf.minimum(image.shape[1], x_end)
            y_end_bound = tf.minimum(image.shape[0], y_end)
            cropped_image = image[y_start_bound:y_end_bound, x_start_bound:x_end_bound]
            z_start = tf.cast(z_start, tf.float32)
            z_end = tf.cast(z_end, tf.float32)
            cropped_image = tf.where(tf.math.logical_and(cropped_image < z_start, cropped_image != 0),
                                     z_start, cropped_image)
            cropped_image = tf.where(cropped_image > z_end, 0., cropped_image)

            # Pad the cropped image if we were out of bounds
            padded_image = tf.pad(cropped_image, [[y_start_bound - y_start, y_end - y_end_bound],
                                                  [x_start_bound - x_start, x_end - x_end_bound],
                                                  [0, 0]])
            return tf.RaggedTensor.from_tensor(padded_image, ragged_rank=2)

        cropped = tf.map_fn(crop, elems=[images, bcubes],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, 1], dtype=images.dtype))
        return cropped

    def crop_bbox(self, images, bboxes):
        """
        Crop images using bounding boxes.
        Pads the cropped area on out of bounds.

        Parameters
        ----------
        images  tf.Tensor shape=[None, 480, 640]
        bboxes  tf.Tensor shape=[None, 4]
            The last dim contains [left, top, right, bottom].

        Returns
        -------
        Cropped image
            The cropped image is of the same shape as the bbox.
        """

        def crop(elems):
            image, bbox = elems
            x_start, y_start, x_end, y_end = bbox
            cropped_image = image[y_start:y_end, x_start:x_end]

            # Pad the cropped image if we were out of bounds
            x_start_bound = tf.maximum(x_start, 0)
            y_start_bound = tf.maximum(y_start, 0)
            x_end_bound = tf.minimum(image.shape[1], x_end)
            y_end_bound = tf.minimum(image.shape[0], y_end)

            padded_image = tf.pad(cropped_image, [[y_start_bound - y_start, y_end - y_end_bound],
                                                  [x_start_bound - x_start, x_end - x_end_bound]])

            return tf.RaggedTensor.from_tensor(padded_image, ragged_rank=2)

        cropped = tf.map_fn(tf.autograph.experimental.do_not_convert(crop), elems=[images, bboxes],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, 1], dtype=images.dtype))
        return cropped
