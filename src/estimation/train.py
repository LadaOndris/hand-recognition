import tensorflow as tf

import src.estimation.configuration as configs
import src.utils.logs as logs_utils
from src.datasets.bighand.dataset import BighandDataset
from src.datasets.msra.dataset import MSRADataset
from src.estimation.architecture.jgrp2o import JGR_J2O
from src.estimation.architecture.losses import CoordinateLoss, OffsetLoss
from src.estimation.configuration import Config
from src.estimation.preprocessing import DatasetPreprocessor
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR


def get_train_and_test_generator(dataset_name: str, network, train_config: configs.Config,
                                 test_config: configs.Config) -> (DatasetPreprocessor, DatasetPreprocessor):
    cam = Camera(dataset_name)

    def get_preprocessor(dataset, config):
        return DatasetPreprocessor(iter(dataset), network.input_size, network.out_size,
                                   camera=cam, config=config)

    if dataset_name == 'bighand':
        ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=train_config.batch_size,
                            shuffle=True)
    elif dataset_name == 'msra':
        ds = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=train_config.batch_size, shuffle=True)
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")

    train_ds_gen = get_preprocessor(ds.train_dataset, train_config)
    test_ds_gen = get_preprocessor(ds.test_dataset, test_config)
    return ds, train_ds_gen, test_ds_gen


def train(dataset_name: str, weights_path: str, config: Config, model_features=128):
    network = JGR_J2O(n_features=model_features)
    model = network.graph()
    print(model.summary(line_length=120))
    if weights_path is not None:
        model.load_weights(weights_path)

    dataset, train_ds_gen, test_ds_gen = get_train_and_test_generator(dataset_name, network, config)
    monitor_loss = 'val_loss'
    if dataset.num_test_batches == 0:
        test_ds_gen = None
        monitor_loss = 'loss'

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor=monitor_loss, patience=10, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    steps_per_epoch = dataset.num_train_batches
    if dataset_name == 'bighand':
        steps_per_epoch = 1024

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=config.learning_decay_rate,
        staircase=True)
    adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.98)
    model.compile(optimizer=adam, loss=[CoordinateLoss(), OffsetLoss()])

    if dataset_name == "bighand":
        model.fit(train_ds_gen, epochs=1000, verbose=0, callbacks=callbacks, steps_per_epoch=steps_per_epoch,
                  validation_data=test_ds_gen, validation_steps=dataset.num_test_batches)
    else:
        model.fit(train_ds_gen, epochs=70, verbose=0, callbacks=callbacks, steps_per_epoch=steps_per_epoch,
                  validation_data=test_ds_gen, validation_steps=dataset.num_test_batches)

    # probably won't come to this, but just to be sure.
    # (the best checkpoint is being saved after each epoch)
    model_filepath = logs_utils.compose_model_path(prefix=F"jgrp2o_{dataset_name}_")
    model.save_weights(model_filepath)
    # checkpoints are located in the log_dir
    # the saved model is located in the model_filepath
    return log_dir, str(model_filepath)
