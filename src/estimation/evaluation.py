import matplotlib.pyplot as plt
import numpy as np

import src.estimation.configuration as configs
from src.datasets.bighand.dataset import BighandDataset
from src.datasets.msra.dataset import MSRADataset
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.estimation.jgrp2o import JGR_J2O
from src.estimation.metrics import DistancesBelowThreshold, MeanJointErrorMetric
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR, LOGS_DIR, MSRAHANDGESTURE_DATASET_DIR


def plot_proportions_below_threshold():
    vals = """
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 1.17702448e-04 2.35404896e-04 8.23917137e-04 4.59039548e-03
 9.06308851e-03 1.80084746e-02 2.81308851e-02 4.37853107e-02
 6.36770245e-02 9.16902072e-02 1.23234463e-01 1.60546139e-01
 2.00682674e-01 2.45762712e-01 2.89548023e-01 3.32980226e-01
 3.78413371e-01 4.20550847e-01 4.63865348e-01 5.07062147e-01
 5.47787194e-01 5.86393597e-01 6.22881356e-01 6.55838041e-01
 6.84204331e-01 7.09863465e-01 7.34698682e-01 7.58121469e-01
 7.80249529e-01 7.98022599e-01 8.16854991e-01 8.34157250e-01
 8.48046139e-01 8.62170433e-01 8.76883239e-01 8.87476460e-01
 8.98540490e-01 9.07721281e-01 9.15254237e-01 9.23964218e-01
 9.29967043e-01 9.37146893e-01 9.42443503e-01 9.46092279e-01
 9.50329567e-01 9.54449153e-01 9.58333333e-01 9.61158192e-01
 9.65160075e-01 9.68338041e-01 9.71869115e-01 9.74458569e-01
 9.77165725e-01 9.79402072e-01 9.82344633e-01 9.83757062e-01
 9.84934087e-01 9.86817326e-01 9.88582863e-01 9.89759887e-01
 9.90936911e-01 9.91525424e-01 9.92231638e-01 9.93173258e-01
 9.94114878e-01 9.94938795e-01 9.96351224e-01 9.96822034e-01
 9.96939736e-01 9.97410546e-01 9.97645951e-01 9.97999058e-01
 9.98116761e-01 9.98705273e-01 9.99058380e-01 9.99058380e-01
 9.99293785e-01 9.99411488e-01 9.99411488e-01 9.99529190e-01
 9.99529190e-01 9.99529190e-01 9.99529190e-01 9.99529190e-01
 9.99529190e-01 9.99529190e-01 9.99529190e-01 9.99646893e-01
 9.99646893e-01 9.99646893e-01 9.99646893e-01 9.99764595e-01
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
 9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
 9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
 9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99764595e-01
 9.99764595e-01 9.99764595e-01 9.99764595e-01 9.99882298e-01
 9.99882298e-01 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
     """

    proportions = np.fromstring(vals, sep=' ')

    fig, ax = plt.subplots()
    ax.plot(proportions[:120])
    fig.show()


def get_evaluation_dataset_generator(dataset_name: str, network, batch_size: int) -> DatasetPreprocessor:
    cam = Camera(dataset_name)
    if dataset_name == 'bighand':
        dataset = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=batch_size, shuffle=True)
        test_generator = DatasetPreprocessor(iter(dataset.test_dataset), network.input_size,
                                             network.out_size, camera=cam, config=configs.EvaluateBighandConfig())
    elif dataset_name == 'msra':
        dataset = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        test_generator = DatasetPreprocessor(iter(dataset.test_dataset), network.input_size,
                                             network.out_size, camera=cam, config=configs.EvaluateMsraConfig())
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return dataset, test_generator


def evaluate(dataset_name: str, weights_path: str, model_features=128):
    network = JGR_J2O(n_features=model_features)
    cam = Camera(dataset_name)
    dataset, generator = get_evaluation_dataset_generator(dataset_name, network, batch_size=8)

    model = network.graph()
    model.load_weights(weights_path)
    mje_metric = MeanJointErrorMetric()
    threshold_metric = DistancesBelowThreshold(max_thres=200)

    for batch_idx in range(dataset.num_test_batches):
        normalized_images, y_true = next(generator)
        y_pred = model.predict(normalized_images)
        # y_pred and y_true are normalized values from/for the model
        uvz_true = generator.postprocess(y_true)
        uvz_pred = generator.postprocess(y_pred)
        xyz_pred = cam.pixel_to_world(uvz_pred)
        xyz_true = cam.pixel_to_world(uvz_true)
        threshold_metric.update_state(xyz_true, xyz_pred)
        mje_metric.update_state(xyz_true, xyz_pred)
    return threshold_metric.result(), mje_metric.result()


def evaluate_msra():
    # weights = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')  # msra, first time, 31 MJE
    # weights = LOGS_DIR.joinpath("20210323-160416/train_ckpts/weights.10.h5")
    # weights = LOGS_DIR.joinpath("20210324-203043/train_ckpts/weights.12.h5")
    # weights = LOGS_DIR.joinpath('20210403-191540/train_ckpts/weights.09.h5')  # msra
    # weights = LOGS_DIR.joinpath('20210407-172551/train_ckpts/weights.13.h5')  # msra, second time, 18 MJE
    weights = LOGS_DIR.joinpath('20210421-221853/train_ckpts/weights.22.h5')  # msra, third time, 14.87 MJE
    thres, mje = evaluate('msra', weights)
    print(mje)


def evaluate_bighand():
    weights = LOGS_DIR.joinpath("20210426-125059/train_ckpts/weights.25.h5")  # bighand
    thres, mje = evaluate('bighand', weights, model_features=196)
    print(mje)


if __name__ == '__main__':
    evaluate_bighand()
