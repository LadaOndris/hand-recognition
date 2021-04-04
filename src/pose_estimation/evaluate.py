from src.datasets.MSRAHandGesture.dataset import MSRADataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.pose_estimation.metrics import MeanJointErrorMetric, DistancesBelowThreshold
from src.utils.paths import LOGS_DIR
from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR
from src.utils.camera import Camera
import matplotlib.pyplot as plt
import numpy as np


def plot_proportions_below_threshold():
    vals = """
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 1.17702448e-04
     1.17702448e-04 1.17702448e-04 4.70809793e-04 1.41242938e-03
     3.29566855e-03 6.12052731e-03 1.18879473e-02 2.08333333e-02
     3.23681733e-02 5.29661017e-02 7.49764595e-02 1.01930320e-01
     1.30061205e-01 1.64665725e-01 1.95739171e-01 2.29519774e-01
     2.63064972e-01 3.04143126e-01 3.47222222e-01 3.88418079e-01
     4.26671375e-01 4.68808851e-01 5.08121469e-01 5.45080038e-01
     5.81214689e-01 6.11817326e-01 6.40065913e-01 6.64665725e-01
     6.87853107e-01 7.11040490e-01 7.36111111e-01 7.55649718e-01
     7.73893597e-01 7.89194915e-01 8.02966102e-01 8.14147834e-01
     8.26859699e-01 8.37452919e-01 8.48399247e-01 8.60640301e-01
     8.72057439e-01 8.81002825e-01 8.89595104e-01 8.98305085e-01
     9.06544256e-01 9.15136535e-01 9.20668550e-01 9.26082863e-01
     9.32321092e-01 9.36676083e-01 9.42208098e-01 9.46680791e-01
     9.51271186e-01 9.54331450e-01 9.58804143e-01 9.63983051e-01
     9.66219397e-01 9.69750471e-01 9.71986817e-01 9.74223164e-01
     9.76224105e-01 9.78695857e-01 9.80932203e-01 9.82344633e-01
     9.83992467e-01 9.85993409e-01 9.87405838e-01 9.88818267e-01
     9.89524482e-01 9.90348399e-01 9.91172316e-01 9.91643126e-01
     9.92113936e-01 9.92702448e-01 9.93290960e-01 9.93526365e-01
     9.93879473e-01 9.94585687e-01 9.95056497e-01 9.95645009e-01
     9.96115819e-01 9.96233522e-01 9.96939736e-01 9.97175141e-01
     9.97881356e-01 9.97999058e-01 9.98352166e-01 9.98705273e-01
     9.98705273e-01 9.98940678e-01 9.99058380e-01 9.99293785e-01
     9.99293785e-01 9.99529190e-01 9.99529190e-01 9.99529190e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99646893e-01
     9.99646893e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
     9.99882298e-01 9.99882298e-01 9.99882298e-01 9.99882298e-01
     """

    proportions = np.fromstring(vals, sep=' ')

    fig, ax = plt.subplots()
    ax.plot(proportions[:120])
    fig.show()


def evaluate(dataset: str, weights_path: str):
    if dataset != 'msra':
        raise ValueError("Invalid dataset")

    network = JGR_J2O()
    cam = Camera(dataset)

    ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=8, shuffle=False)
    test_ds_gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                   camera=cam, dataset_includes_bboxes=True)

    model = network.graph()
    model.load_weights(weights_path)
    mje_metric = MeanJointErrorMetric()
    threshold_metric = DistancesBelowThreshold(max_thres=200)

    for batch_idx in range(ds.num_test_batches):
        normalized_images, y_true = next(test_ds_gen)
        y_pred = model.predict(normalized_images)
        # y_pred and y_true are normalized values from/for the model
        uvz_true = test_ds_gen.postprocess(y_true)
        uvz_pred = test_ds_gen.postprocess(y_pred)
        xyz_pred = cam.pixel_to_world(uvz_pred)
        xyz_true = cam.pixel_to_world(uvz_true)
        threshold_metric.update_state(xyz_true, xyz_pred)
        mje_metric.update_state(xyz_true, xyz_pred)
    return threshold_metric.result(), mje_metric.result()


if __name__ == '__main__':
    # weights = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')
    # weights = LOGS_DIR.joinpath("20210323-160416/train_ckpts/weights.10.h5")
    # weights = LOGS_DIR.joinpath("20210324-203043/train_ckpts/weights.12.h5")
    weights = LOGS_DIR.joinpath('20210403-191540/train_ckpts/weights.09.h5')  # msra
    thres, mje = evaluate('msra', weights)
    print(mje)
