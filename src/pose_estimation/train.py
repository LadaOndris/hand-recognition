import tensorflow as tf
from src.pose_estimation.loss import CoordinateAndOffsetLoss, CoordinateLoss, OffsetLoss
from src.datasets.bighand.dataset import BighandDataset
from src.datasets.MSRAHandGesture.dataset import MSRADataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR
from src.utils.camera import Camera
import matplotlib.pyplot as plt


def train(dataset: str):
    network = JGR_J2O()
    cam = Camera(dataset)

    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
        gen = DatasetGenerator(iter(ds.train_dataset), network.input_size, network.out_size, camera=cam,
                               dataset_includes_bboxes=False)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=4, shuffle=True)
        train_ds_gen = DatasetGenerator(iter(ds.train_dataset), network.input_size, network.out_size, camera=cam,
                                        dataset_includes_bboxes=True)
        test_ds_gen = DatasetGenerator(iter(ds.test_dataset), network.input_size, network.out_size, camera=cam,
                                       dataset_includes_bboxes=True)

    model = network.graph()
    print(model.summary(line_length=100))

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.96)
    model.compile(optimizer=adam, loss=[CoordinateLoss(), OffsetLoss()])
    model.fit(train_ds_gen, verbose=1, steps_per_epoch=ds.num_train_batches,
              validation_data=test_ds_gen, validation_steps=ds.num_test_batches)


def try_dataset_pipeline():
    network = JGR_J2O()
    cam = Camera('bighand')
    bighand_ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
    gen = DatasetGenerator(iter(bighand_ds.test_dataset), network.input_size, network.out_size, camera=cam)
    for images, y_true in gen:
        offsets = y_true[1]
        joints = y_true[0]
        plt.imshow(images[0])


if __name__ == "__main__":
    train('msra')
