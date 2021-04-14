import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def resize_images(cropped_imgs, target_size):
    def _resize(img):
        if type(img) is tf.RaggedTensor:
            img = img.to_tensor()
        return resize_bilinear_nearest(img, target_size)

    return tf.map_fn(_resize, cropped_imgs,
                     fn_output_signature=tf.TensorSpec(shape=(target_size[0], target_size[1], 1),
                                                       dtype=tf.float32))


def tf_resize_image(depth_image, shape, resize_mode: str):
    """

    Parameters
    ----------
    depth_image
    shape
    resize_mode : str
       "crop" - Crop mode first crops the image to a square and then resizes
            with a combination of bilinear and nearest interpolation.
       "pad"  - In pad mode, the image is resized and and the rest is padded
            to retain the aspect ration of the original image.
    Returns
    -------
    """
    # convert the values to range 0-255 as tf.io.read_file does
    # depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.uint8)
    # resize image
    type = depth_image.dtype
    if resize_mode == 'pad':
        depth_image = tf.image.resize_with_pad(depth_image, shape[0], shape[1])
    elif resize_mode == 'crop':
        height, width, channels = depth_image.shape
        if height > width:
            offset = tf.cast((height - width) / 2, tf.int32)
            cropped = depth_image[offset:height - offset, :, :]
        elif width > height:
            offset = tf.cast((width - height) / 2, tf.int32)
            cropped = depth_image[:, offset:width - offset, :]
        else:
            cropped = depth_image
        depth_image = resize_bilinear_nearest(cropped, shape)
        # depth_image = depth_image[tf.newaxis, ...]
        # depth_image = tf.image.crop_and_resize(depth_image, [[0, 80 / 640.0, 480 / 480.0, 560 / 640.0]],
        #                                        [0], shape)
        # depth_image = depth_image[0]
    else:
        raise ValueError(F"Unknown resize mode: {resize_mode}")
    # depth_image = tf.where(depth_image > 2000, 0, depth_image)
    depth_image = tf.cast(depth_image, dtype=type)
    return depth_image


def resize_bilinear_nearest(image_in, shape):
    img_height, img_width = image_in.shape[:2]
    width, height = shape[:2]

    image = tf.cast(image_in, tf.float32)
    image = tf.reshape(image, [-1])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    xx = tf.range(width)
    yy = tf.range(height)
    x, y = tf.meshgrid(xx, yy)
    x, y = tf.reshape(x, [-1]), tf.reshape(y, [-1])
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    inter_points_x = tf.cast(x_ratio * x, tf.float32)
    inter_points_y = tf.cast(y_ratio * y, tf.float32)

    x_l = tf.math.floor(inter_points_x)
    y_l = tf.math.floor(inter_points_y)
    x_h = tf.math.ceil(inter_points_x)
    y_h = tf.math.ceil(inter_points_y)

    x_weight = inter_points_x - x_l
    y_weight = inter_points_y - y_l

    x_l = tf.cast(x_l, tf.int32)
    y_l = tf.cast(y_l, tf.int32)
    x_h = tf.cast(x_h, tf.int32)
    y_h = tf.cast(y_h, tf.int32)

    a = tf.gather(image, y_l * img_width + x_l)
    b = tf.gather(image, y_l * img_width + x_h)
    c = tf.gather(image, y_h * img_width + x_l)
    d = tf.gather(image, y_h * img_width + x_h)

    bilinear = a * (1 - x_weight) * (1 - y_weight) + \
               b * x_weight * (1 - y_weight) + \
               c * y_weight * (1 - x_weight) + \
               d * x_weight * y_weight

    # Find nearest for each set of points point
    ab_nearest = tf.where(x_weight < .5, a, b)
    cd_nearest = tf.where(x_weight < .5, c, d)
    nearest = tf.where(y_weight < .5, ab_nearest, cd_nearest)

    # Find points where either of a and b is zero
    mask_is_zero = tf.math.logical_or(
        tf.math.logical_or(a == 0, b == 0),
        tf.math.logical_or(c == 0, d == 0))

    # Apply nearest interpolation
    resized = tf.where(mask_is_zero, nearest, bilinear)

    return tf.reshape(resized, [height, width, 1])


def resize_bilinear_nearest_batch(images_in, shape):
    """
    Input images are expected to be the same shape.
    """
    batch_size, img_height, img_width, channels = tf.shape(images_in)
    width, height = shape[:2]

    images = tf.cast(images_in, tf.float32)
    images = tf.reshape(images, [batch_size, -1])

    x_ratio = tf.cast(img_width - 1, tf.float64) / (width - 1) if width > 1 else 0
    y_ratio = tf.cast(img_height - 1, tf.float64) / (height - 1) if height > 1 else 0

    xx = tf.range(width)
    yy = tf.range(height)
    x, y = tf.meshgrid(xx, yy)
    x, y = tf.reshape(x, [-1]), tf.reshape(y, [-1])
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    inter_points_x = tf.cast(x_ratio * x, tf.float32)
    inter_points_y = tf.cast(y_ratio * y, tf.float32)

    x_l = tf.math.floor(inter_points_x)
    y_l = tf.math.floor(inter_points_y)
    x_h = tf.math.ceil(inter_points_x)
    y_h = tf.math.ceil(inter_points_y)

    x_weight = inter_points_x - x_l
    y_weight = inter_points_y - y_l

    x_l = tf.cast(x_l, tf.int32)
    y_l = tf.cast(y_l, tf.int32)
    x_h = tf.cast(x_h, tf.int32)
    y_h = tf.cast(y_h, tf.int32)

    a_indices = y_l * img_width + x_l
    b_indices = y_l * img_width + x_h
    c_indices = y_h * img_width + x_l
    d_indices = y_h * img_width + x_h

    # batches = tf.range(batch_size)
    a_indices = tf.tile(a_indices[tf.newaxis, :], [batch_size, 1])
    b_indices = tf.tile(b_indices[tf.newaxis, :], [batch_size, 1])
    c_indices = tf.tile(c_indices[tf.newaxis, :], [batch_size, 1])
    d_indices = tf.tile(d_indices[tf.newaxis, :], [batch_size, 1])
    a = tf.gather(images, a_indices, axis=1, batch_dims=1)
    b = tf.gather(images, b_indices, axis=1, batch_dims=1)
    c = tf.gather(images, c_indices, axis=1, batch_dims=1)
    d = tf.gather(images, d_indices, axis=1, batch_dims=1)

    bilinear = a * (1 - x_weight) * (1 - y_weight) + \
               b * x_weight * (1 - y_weight) + \
               c * y_weight * (1 - x_weight) + \
               d * x_weight * y_weight

    # Find nearest for each set of points point
    ab_nearest = tf.where(x_weight < .5, a, b)
    cd_nearest = tf.where(x_weight < .5, c, d)
    nearest = tf.where(y_weight < .5, ab_nearest, cd_nearest)

    # Find points where either of a and b is zero
    mask_is_zero = tf.math.logical_or(
        tf.math.logical_or(a == 0, b == 0),
        tf.math.logical_or(c == 0, d == 0))

    # Apply nearest interpolation
    resized = tf.where(mask_is_zero, nearest, bilinear)

    return tf.reshape(resized, [batch_size, height, width, 1])


if __name__ == "__main__":
    # Check bilinear nearest resizing
    im = tf.zeros([5, 194, 195, 1])
    res = resize_bilinear_nearest_batch(im, [96, 96])
    tf.print(res.shape)
