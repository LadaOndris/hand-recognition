import numpy as np
import tensorflow as tf


class Camera:

    def __init__(self, dataset: str):
        if dataset == 'bighand':
            self.set_intel_realsense_sr300()
        elif dataset == 'MSRA':
            self.set_intel_creative_interactive()
        else:
            raise NotImplementedError('Unknown dataset')

    def set_intel_creative_interactive(self):
        self.focal_length = [241.42, 241.42]
        self.principal_point = [160, 120]
        self.image_size = [320, 240]

    def set_intel_realsense_sr300(self):
        self.focal_length = [475.065948, 475.065857]  # [476.0068, 476.0068]  # [588.235, 587.084]
        self.principal_point = [315.944855, 245.287079]
        self.image_size = [640, 480]

        # More information on intrinsic and extrinsic parameters
        # can be found at: https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
        self.cam_extr = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1]], dtype=tf.float32)

        self.cam_intr = tf.constant([[475.065948, 0, 315.944855],
                                     [0, 475.065857, 245.287079],
                                     [0, 0, 1]], dtype=tf.float32)
        # self.set_M(cam_extr, cam_intr)

    def set_M(self, extr, intr):
        self.M = np.dot(intr, extr)

    def project_onto_2d_plane(self, points_3d):
        """
        Projects the given points through pinhole camera on a plane at a distance of focal_length.
        Returns
        -------
            2-D coordinates
        """
        if tf.rank(points_3d) == 2:
            points_3d = points_3d[tf.newaxis, ...]
        new_shape = [points_3d.shape[0], points_3d.shape[1], 1]
        points = tf.concat([points_3d, tf.ones(new_shape, dtype=points_3d.dtype)], axis=-1)
        extr_points = tf.matmul(self.cam_extr, points, transpose_b=True)
        extr_points = tf.transpose(extr_points, [0, 2, 1])[..., :3]
        intr_points = tf.matmul(self.cam_intr, extr_points, transpose_b=True)
        intr_points = tf.transpose(intr_points, [0, 2, 1])
        proj_points = intr_points[..., :2] / intr_points[..., 2:3]
        return proj_points

        res = tf.matmul(tf.cast(self.M, dtype=tf.float32), points[:, :, tf.newaxis])
        res = tf.squeeze(res)
        return res[:, :2] / res[:, 2:3]

        coefficient = self.focal_length / points_3d[..., 2:3]
        xy_moved_to_middle = points_3d[..., 0:2] - self.principal_point
        return coefficient * xy_moved_to_middle + self.principal_point

    def convert_UVZ_to_3D(points):
        pass
