import tensorflow as tf
from src.pose_estimation.loss import CoordinateAndOffsetLoss, CoordinateLoss, OffsetLoss
from src.datasets.bighand.dataset import BighandDataset
from src.datasets.MSRAHandGesture.dataset import MSRADataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.pose_estimation.evaluate import evaluate
from src.pose_estimation.metrics import MeanJointErrorMetric, DistancesBelowThreshold
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR, SAVED_MODELS_DIR, LOGS_DIR
from src.utils.camera import Camera
from src.utils.config import JGRJ2O_LEARNING_RATE, JGRJ2O_LR_DECAY, JGRJ2O_TRAIN_BATCH_SIZE
from src.utils.plots import plot_joints_2d
import matplotlib.pyplot as plt
import src.utils.logs as logs_utils
import os
import glob
import argparse


def test(dataset: str, weights_path: str):
    network = JGR_J2O()
    cam = Camera(dataset)

    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
        test_ds_gen = DatasetGenerator(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                       camera=cam,
                                       dataset_includes_bboxes=False)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=1, shuffle=True)
        test_ds_gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                       camera=cam, dataset_includes_bboxes=True)

    model = network.graph()
    model.load_weights(weights_path)
    # model.load_weights(SAVED_MODELS_DIR.joinpath('jgrp2o_msra_20210305-220222.h5'))
    for batch_images, y_true in test_ds_gen:
        y_pred = model.predict(batch_images)

        for image, true, pred in zip(batch_images, y_true[0], y_pred[0]):
            plot_joints_2d(image, true[..., :2] * 96)
            plot_joints_2d(image, pred[..., :2] * 96)

        """
        y_pred_joints = test_ds_gen.postprocess(y_pred)
        y_true_joints = test_ds_gen.postprocess(y_true)

        y_pred_joints = y_pred_joints[..., :2] - tf.cast(test_ds_gen.bboxes[:, tf.newaxis, :2], dtype=tf.float32)
        y_true_joints = y_true_joints[..., :2] - tf.cast(test_ds_gen.bboxes[:, tf.newaxis, :2], dtype=tf.float32)

        for image, local_joints_2d, true_joints_2d in zip(test_ds_gen.cropped_images, y_pred_joints, y_true_joints):
            plot_joints_2d(image.to_tensor(), local_joints_2d)
            plot_joints_2d(image.to_tensor(), true_joints_2d)
        """


def train(dataset: str):
    network = JGR_J2O()
    cam = Camera(dataset)

    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=JGRJ2O_TRAIN_BATCH_SIZE, shuffle=True)
        train_ds_gen = DatasetGenerator(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                        camera=cam, dataset_includes_bboxes=False, augment=True, cube_size=250)
        test_ds_gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                       camera=cam, dataset_includes_bboxes=False, cube_size=250)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=JGRJ2O_TRAIN_BATCH_SIZE, shuffle=True)
        train_ds_gen = DatasetGenerator(iter(ds.train_dataset), cam.image_size, network.input_size, network.out_size,
                                        camera=cam, dataset_includes_bboxes=True, augment=True, cube_size=180)
        test_ds_gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                                       camera=cam, dataset_includes_bboxes=True, cube_size=180)

    model = network.graph()
    print(model.summary(line_length=120))

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=JGRJ2O_LEARNING_RATE,
        decay_steps=ds.num_train_batches,
        decay_rate=JGRJ2O_LR_DECAY,
        staircase=True)
    adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.98)
    model.compile(optimizer=adam, loss=[CoordinateLoss(), OffsetLoss()])
    model.fit(train_ds_gen, epochs=70, verbose=0, callbacks=callbacks, steps_per_epoch=ds.num_train_batches,
              validation_data=test_ds_gen, validation_steps=ds.num_test_batches)

    # probably won't come to this, but just to be sure.
    # (the best checkpoint is being saved after each epoch)
    model_filepath = logs_utils.compose_model_path(prefix=F"jgrp2o_{dataset}_")
    model.save_weights(model_filepath)
    # checkpoints are located in the log_dir
    # the saved model is located in the model_filepath
    return log_dir, str(model_filepath)


def try_dataset_pipeline(dataset: str):
    network = JGR_J2O()
    cam = Camera(dataset)

    if dataset == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=8, shuffle=True)
        gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size,
                               network.out_size, camera=cam, augment=True, cube_size=250)
    elif dataset == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=8, shuffle=True)
        gen = DatasetGenerator(iter(ds.test_dataset), cam.image_size, network.input_size, network.out_size,
                               camera=cam, dataset_includes_bboxes=True, augment=True, cube_size=180)
    for images, y_true in gen:
        joints = y_true[0]
        # y_true_joints = gen.postprocess(y_true)
        # true_joints_2d = y_true_joints[..., :2] - tf.cast(gen.bboxes[:, tf.newaxis, :2], dtype=tf.float32)

        for image, true_joints in zip(images, joints):
            plot_joints_2d(image, true_joints[..., :2] * 96)


if __name__ == "__main__":
    # try_dataset_pipeline('bighand')
    # test('msra', weights)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, action='store', default=None)
    parser.add_argument('--evaluate', type=str, action='store', default=None)
    args = parser.parse_args()

    if args.train is not None:
        log_dir, model_filepath = train(args.train)

    if args.evaluate is not None and args.train is not None:
        if model_filepath is not None and os.path.isfile(model_filepath):
            path = model_filepath
        else:
            ckpts_pattern = os.path.join(str(log_dir), 'train_ckpts/*')
            ckpts = glob.glob(ckpts_pattern)
            path = max(ckpts, key=os.path.getctime)
        # path = LOGS_DIR.joinpath('20210306-000815/train_ckpts/weights.07.h5')
        if path is not None:
            mje = evaluate(args.evaluate, path)
            tf.print("MJE:", mje)
        else:
            raise ValueError("No checkpoints available")
