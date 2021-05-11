import src.estimation.configuration as configs
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.live import generate_live_images
from src.utils.paths import BIGHAND_DATASET_DIR, CUSTOM_DATASET_DIR, DOCS_DIR


def plot_estimation_on_custom_dataset():
    # Gesture 1
    # for j in range(3, 18):
    #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-230536/{j}0.png")))

    # Gesture 1, but fingers are not stretched
    # for j in range(1, 6):  # up to 21
    #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-231615/{j}0.png")))

    # Gesture 1, but different angle
    # for j in range(1, 21):
    #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-231510/{j}0.png")))

    # Gesture 1, but fingers are not far apart
    # for j in range(1, 20):
    #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-231929/{j}0.png")))

    # Gesture 2
    # for j in range(1, 10):
    #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232147/{j}0.png")))

    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-230536/85.png')))
    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-232147/42.png')))
    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-232147/75.png')))
    # # Fails in otsus method for empty array
    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-233307/22.png")))

    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232147/70.png")))
    # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-231436/70.png")))

    # Closed hand
    # for j in range(9):
    #   estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232751/50{j}.png")))

    # Alphabet
    for j in range(1, 11):
        estimator.estimate_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232751/{j}00.png")))
        estimator.estimate_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232751/{j}50.png")))


def save_estimations_on_bighand():
    files = get_bighand_image_files()
    fig_filenames = get_figures_filenames(files)
    for file, fig_filename in zip(files, fig_filenames):
        estimator.estimate_from_file(str(BIGHAND_DATASET_DIR.joinpath(file)),
                                     fig_location=DOCS_DIR.joinpath(F"figures/evaluation/bighand/{fig_filename}"))


def get_bighand_image_files():
    files = []
    subjects = [1, 2, 3]
    angles = ['1 75', '76 150', '151 225', '226 300',
              '301 375', '376 450', '451 496']
    for subject in subjects:
        for angle in angles:
            file = F"Subject_{subject}/{angle}/image_D00010000.png"
            files.append(file)
    return files


def get_figures_filenames(image_files):
    return [F"estimation_bighand_{i}.png" for i, file in enumerate(image_files)]


def plot_estimation_on_bighand():
    for i in range(3):
        for j in range(0, 9):
            estimator.estimate_from_file(str(BIGHAND_DATASET_DIR.joinpath(
                F"Subject_1/226 300/image_D000{i}{j}000.png")))


if __name__ == '__main__':
    dataset = 'custom'
    if dataset == 'bighand':
        config = configs.PredictBighandConfig()
    elif dataset == 'msra':
        config = configs.PredictMsraConfig()
    else:
        config = configs.PredictCustomDataset()
    estimator = HandPositionEstimator(Camera(dataset), config=config, plot_detection=False,
                                      plot_estimation=True, plot_skeleton=True)

    # if dataset == 'bighand':
    #     # save_estimations_on_bighand()
    #     plot_estimation_on_bighand()
    # else:
    #     plot_estimation_on_custom_dataset()
    live_image_generator = generate_live_images()
    estimator.estimate_from_source(live_image_generator)
