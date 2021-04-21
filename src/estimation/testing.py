from src.estimation.common import get_dataset_generator
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.estimation.jgrp2o import JGR_J2O
from src.utils.plots import plot_joints_2d


def plot_estimators_predictions(y, generator: DatasetPreprocessor):
    uvz_pred = generator.postprocess(y)
    joints2d = generator.convert_coords_to_local(uvz_pred)

    for image, joints in zip(generator.cropped_imgs, joints2d):
        plot_joints_2d(image.to_tensor(), joints, figsize=(3, 3))


def inference_on_dataset(dataset: str, weights_path: str, model_features=128):
    network = JGR_J2O(n_features=model_features)
    model = network.graph()
    model.load_weights(weights_path)
    gen = get_dataset_generator(dataset, network, batch_size=4, augment=False)

    for batch_images, y_true in gen:
        y_pred = model.predict(batch_images)
        plot_estimators_predictions(y_pred, gen)


def try_dataset_pipeline(dataset: str):
    network = JGR_J2O()
    gen = get_dataset_generator(dataset, network, batch_size=4, augment=True)

    for images, y_true in gen:
        plot_estimators_predictions(y_true, gen)


if __name__ == "__main__":
    # weights = LOGS_DIR.joinpath("20210330-024055/train_ckpts/weights.31.h5")  # bighand
    # weights = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')  # msra, first time, 31 MJE
    # weights = LOGS_DIR.joinpath('20210402-112810/train_ckpts/weights.14.h5')  # msra, then bighand
    # weights = LOGS_DIR.joinpath('20210403-183544/train_ckpts/weights.01.h5')  # overtrained on single image
    # weights = LOGS_DIR.joinpath('20210407-172551/train_ckpts/weights.13.h5')  # msra, second time, 18 MJE
    # weights = LOGS_DIR.joinpath("20210418-200105/train_ckpts/weights.12.h5")  # bighand, 196 features
    # inference_on_dataset('bighand', weights)
    try_dataset_pipeline('bighand')
