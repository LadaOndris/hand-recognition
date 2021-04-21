from src.datasets.bighand.dataset import BighandDataset
from src.datasets.msra.dataset import MSRADataset
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR


def get_dataset_generator(dataset: str, network, batch_size: int, augment: bool) -> DatasetPreprocessor:
    cam = Camera(dataset)
    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size,
                                  network.out_size, camera=cam, augment=augment, cube_size=150,
                                  refine_iters=0)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                  camera=cam, dataset_includes_bboxes=True, augment=augment, cube_size=180)
    return gen


def get_train_and_test_generator(dataset: str, network, batch_size) -> (DatasetPreprocessor, DatasetPreprocessor):
    cam = Camera(dataset)
    bighand_test_size = 1.0

    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=bighand_test_size, batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                           camera=cam, augment=True, cube_size=150, refine_iters=0)
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                          camera=cam, augment=False, cube_size=150, refine_iters=0)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=batch_size, shuffle=True)
        train_ds_gen = DatasetPreprocessor(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                           camera=cam, dataset_includes_bboxes=True, augment=True, cube_size=180,
                                           thresholding=False, refine_iters=0)
        test_ds_gen = DatasetPreprocessor(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                          camera=cam, dataset_includes_bboxes=True, cube_size=180,
                                          thresholding=False, refine_iters=0)
    return ds, train_ds_gen, test_ds_gen
