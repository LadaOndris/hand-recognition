import src.estimation.configuration as configs
from src.datasets.bighand.dataset import BighandDataset
from src.datasets.msra.dataset import MSRADataset
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR


def get_generator_for_dataset_prediction(dataset_name: str, network, batch_size: int,
                                         augment: bool) -> DatasetPreprocessor:
    cam = Camera(dataset_name)
    if dataset_name == 'bighand':
        config = configs.PredictBighandConfig()
        config.augment = augment
        ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), network.input_size,
                                  network.out_size, camera=cam, config=config)
    elif dataset_name == 'msra':
        config = configs.PredictMsraConfig()
        config.augment = augment
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), network.input_size, network.out_size,
                                  camera=cam, config=config)
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return gen


def get_train_and_test_generator(dataset_name: str, network, batch_size) -> (DatasetPreprocessor, DatasetPreprocessor):
    cam = Camera(dataset_name)

    if dataset_name == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), network.input_size, network.out_size,
                                           camera=cam, config=configs.TrainBighandConfig())
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), network.input_size, network.out_size,
                                          camera=cam, config=configs.TestBighandConfig())
    elif dataset_name == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), network.input_size, network.out_size,
                                           camera=cam, config=configs.TrainMsraConfig())
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), network.input_size, network.out_size,
                                          camera=cam, config=configs.TestMsraConfig())
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return ds, train_ds_gen, test_ds_gen
