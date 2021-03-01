import tensorflow as tf
from src.pose_estimation.loss import CoordinateAndOffsetLoss, CoordinateLoss, OffsetLoss
from src.datasets.bighand.dataset import BighandDataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.utils.paths import BIGHAND_DATASET_DIR
from src.utils.camera import Camera
import matplotlib.pyplot as plt


def train():
    network = JGR_J2O()
    cam = Camera('bighand')
    bighand_ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
    gen = DatasetGenerator(iter(bighand_ds.test_dataset), network.input_size, network.out_size, camera=cam)

    model = network.graph()
    print(model.summary(line_length=100))

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.96)
    model.compile(optimizer=adam, loss=[CoordinateLoss(), OffsetLoss()])
    model.fit(gen, batch_size=2)


def try_dataset_pipeline():
    network = JGR_J2O()
    cam = Camera('bighand')
    bighand_ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
    gen = DatasetGenerator(iter(bighand_ds.test_dataset), network.input_size, network.out_size, camera=cam)
    for images, y_true in gen:
        offsets = y_true['offsets']
        joints = y_true['coords']
        plt.imshow(images[0].numpy())


if __name__ == "__main__":
    train()
