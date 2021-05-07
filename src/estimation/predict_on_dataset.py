import src.utils.plots as plots
from src.estimation.common import get_generator_for_dataset_prediction
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.estimation.jgrp2o import JGR_J2O
from src.estimation.metrics import MeanJointErrorMetric
from src.utils.camera import Camera
from src.utils.paths import LOGS_DIR


def prepare_joints2d(y, generator: DatasetPreprocessor):
    if y is None:
        raise ValueError("y cannot be None")
    uvz = generator.postprocess(y)
    joints2d = generator.convert_coords_to_local(uvz)
    return joints2d


def plot_estimators_predictions(y_pred, generator: DatasetPreprocessor):
    """
    Postprocesses predicted coordinates, and plots them in a cropped image.

    Parameters
    ----------
    y_pred
    generator
    y_true
        Plots ground truth annotation if set.
    """
    joints2d = prepare_joints2d(y_pred, generator)

    for image, joints in zip(generator.cropped_imgs, joints2d):
        plots.plot_joints_2d(image.to_tensor(), joints, figsize=(3, 3))


def plot_predictions_with_annotations(y_pred, y_true, generator: DatasetPreprocessor, camera: Camera):
    uvz_pred = generator.postprocess(y_pred)
    joints2d_pred = generator.convert_coords_to_local(uvz_pred)
    xyz_pred = camera.pixel_to_world(uvz_pred)

    uvz_true = generator.postprocess(y_true)
    joints2d_true = generator.convert_coords_to_local(uvz_true)
    xyz_true = camera.pixel_to_world(uvz_true)

    mje_metric = MeanJointErrorMetric()
    mjes = mje_metric.mean_joint_error(xyz_true, xyz_pred)

    for i in range(generator.cropped_imgs.shape[0]):
        image = generator.cropped_imgs[i]
        joints_pred = joints2d_pred[i]
        joints_true = joints2d_true[i]
        mje = mjes[i]

        # fig_location = DOCS_DIR.joinpath(F"images/estimation/msra/msra_{mje:.2f}.png")
        plots.plot_joints_with_annotations(image.to_tensor(), joints_pred, joints_true, figsize=(3, 3),
                                           fig_location=None)


def infer_on_dataset(dataset: str, weights_path: str, model_features=128):
    network = JGR_J2O(n_features=model_features)
    model = network.graph()
    model.load_weights(weights_path)
    gen = get_generator_for_dataset_prediction(dataset, network, batch_size=4, augment=False)

    for batch_images, y_true in gen:
        y_pred = model.predict(batch_images)
        plot_predictions_with_annotations(y_pred, y_true, gen, Camera('msra'))


def try_dataset_pipeline(dataset: str):
    network = JGR_J2O()
    gen = get_generator_for_dataset_prediction(dataset, network, batch_size=4, augment=True)

    for images, y_true in gen:
        plot_estimators_predictions(y_true, gen)


if __name__ == "__main__":
    # weights = LOGS_DIR.joinpath("20210330-024055/train_ckpts/weights.31.h5")  # bighand
    # weights = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')  # msra, first time, 31 MJE
    # weights = LOGS_DIR.joinpath('20210402-112810/train_ckpts/weights.14.h5')  # msra, then bighand
    # weights = LOGS_DIR.joinpath('20210403-183544/train_ckpts/weights.01.h5')  # overtrained on single image
    # weights = LOGS_DIR.joinpath('20210407-172551/train_ckpts/weights.13.h5')  # msra, second time, 18 MJE
    # weights = LOGS_DIR.joinpath("20210418-200105/train_ckpts/weights.12.h5")  # bighand, 196 features
    # weights = LOGS_DIR.joinpath('20210421-221853/train_ckpts/weights.22.h5')  # msra, third time, 14.87 MJE
    # weights = LOGS_DIR.joinpath("20210423-220702/train_ckpts/weights.13.h5")  # bighand
    # weights = LOGS_DIR.joinpath("20210424-122810/train_ckpts/weights.26.h5")  # bighand
    # weights = LOGS_DIR.joinpath("20210425-122826/train_ckpts/weights.25.h5")  # bighand
    weights = LOGS_DIR.joinpath("20210426-125059/train_ckpts/weights.25.h5")  # bighand

    infer_on_dataset('bighand', weights, model_features=196)
    # try_dataset_pipeline('bighand')
