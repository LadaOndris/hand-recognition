from src.datasets.bighand.dataset import BighandDataset
from src.datasets.msra.dataset import MSRADataset
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR


def get_dataset_generator(dataset_name: str, network, batch_size: int, augment: bool) -> DatasetPreprocessor:
    cam = Camera(dataset_name)
    if dataset_name == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size,
                                  network.out_size, camera=cam, augment=augment, cube_size=150,
                                  refine_iters=0)
    elif dataset_name == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                  camera=cam, dataset_includes_bboxes=True, augment=augment, cube_size=180)
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return gen


def get_evaluation_dataset_generator(dataset_name: str, network, batch_size: int) -> DatasetPreprocessor:
    cam = Camera(dataset_name)
    if dataset_name == 'bighand':
        dataset = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=batch_size, shuffle=False)
        test_generator = DatasetPreprocessor(iter(dataset.test_dataset), cam.image_size, network.input_size,
                                             network.out_size, camera=cam, augment=False, thresholding=False,
                                             cube_size=150, refine_iters=0)
    elif dataset_name == 'msra':
        dataset = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=False)
        test_generator = DatasetPreprocessor(iter(dataset.test_dataset), cam.image_size, network.input_size,
                                             network.out_size, camera=cam, dataset_includes_bboxes=True,
                                             augment=False, thresholding=False, refine_iters=0, cube_size=180)
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return dataset, test_generator


def get_train_and_test_generator(dataset_name: str, network, batch_size) -> (DatasetPreprocessor, DatasetPreprocessor):
    cam = Camera(dataset_name)

    if dataset_name == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                           camera=cam, augment=True, thresholding=False, cube_size=150, refine_iters=0)
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                          camera=cam, augment=False, thresholding=False, cube_size=150, refine_iters=0)
    elif dataset_name == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                           camera=cam, dataset_includes_bboxes=True, augment=True, cube_size=180,
                                           thresholding=False, refine_iters=0)
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                          camera=cam, dataset_includes_bboxes=True, cube_size=180,
                                          thresholding=False, refine_iters=0)
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return ds, train_ds_gen, test_ds_gen
