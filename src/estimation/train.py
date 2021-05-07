import argparse
import glob
import os

import tensorflow as tf

import src.utils.logs as logs_utils
from src.estimation.common import get_train_and_test_generator
from src.estimation.evaluation import evaluate
from src.estimation.jgrp2o import JGR_J2O
from src.estimation.losses import CoordinateLoss, OffsetLoss
from src.utils.config import JGRJ2O_LEARNING_RATE, JGRJ2O_LR_DECAY, JGRJ2O_TRAIN_BATCH_SIZE


def train(dataset_name: str, weights_path: str, model_features=128):
    network = JGR_J2O(n_features=model_features)
    model = network.graph()
    print(model.summary(line_length=120))
    if weights_path is not None:
        model.load_weights(weights_path)

    dataset, train_ds_gen, test_ds_gen = get_train_and_test_generator(dataset_name, network, JGRJ2O_TRAIN_BATCH_SIZE)
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
        initial_learning_rate=JGRJ2O_LEARNING_RATE,
        decay_steps=steps_per_epoch,
        decay_rate=JGRJ2O_LR_DECAY,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, action='store', default=None)
    parser.add_argument('--evaluate', type=str, action='store', default=None)
    parser.add_argument('--model', type=str, action='store', default=None)
    parser.add_argument('--features', type=int, action='store', default=196)
    args = parser.parse_args()

    if args.train is not None:
        log_dir, model_filepath = train(args.train, args.model, model_features=args.features)

    if (args.evaluate is not None) and (args.train is not None):
        if model_filepath is not None and os.path.isfile(model_filepath):
            path = model_filepath
        else:
            ckpts_pattern = os.path.join(str(log_dir), 'train_ckpts/*')
            ckpts = glob.glob(ckpts_pattern)
            path = max(ckpts, key=os.path.getctime)
        if path is not None:
            thresholds, mje = evaluate(args.evaluate, path, args.features)
            tf.print("MJE:", mje)
        else:
            raise ValueError("No checkpoints available")
