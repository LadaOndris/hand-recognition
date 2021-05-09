import tensorflow as tf

import src.utils.logs as logs_utils
from src.datasets.handseg.dataset_bboxes import HandsegDatasetBboxes
from src.detection.yolov3.architecture.loader import YoloLoader
from src.detection.yolov3.architecture.loss import YoloLoss
from src.detection.yolov3.metrics import YoloBoxesIoU, YoloConfPrecisionMetric, YoloConfRecallMetric
from src.detection.yolov3.preprocessing import DatasetPreprocessor
from src.utils.config import TRAIN_YOLO_IOU_IGNORE_THRES
from src.utils.paths import HANDSEG_DATASET_DIR, YOLO_CONFIG_FILE


def train(batch_size, learning_rate, train_size=0.8, decay_rate=0.94):
    model = YoloLoader.parse_cfg(YOLO_CONFIG_FILE)
    yolo_out_shapes = model.yolo_output_shapes

    handseg_dataset = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=train_size,
                                           model_input_shape=model.input_shape,
                                           batch_size=batch_size)
    train_dataset_generator = DatasetPreprocessor(handseg_dataset.train_batch_iterator,
                                                  model.input_shape, yolo_out_shapes, model.anchors)
    test_dataset_generator = DatasetPreprocessor(handseg_dataset.test_batch_iterator,
                                                 model.input_shape, yolo_out_shapes, model.anchors)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=handseg_dataset.num_train_batches,
        decay_rate=decay_rate,
        staircase=True)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
    metrics = [YoloConfPrecisionMetric(), YoloConfRecallMetric(), YoloBoxesIoU()]
    loss = YoloLoss(model.input_shape, ignore_thresh=TRAIN_YOLO_IOU_IGNORE_THRES)

    model.tf_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    model.tf_model.fit(train_dataset_generator, epochs=40, verbose=0, callbacks=callbacks,
                       steps_per_epoch=handseg_dataset.num_train_batches,
                       validation_data=test_dataset_generator,
                       validation_steps=handseg_dataset.num_test_batches)

    model.tf_model.save_weights(logs_utils.compose_model_path(prefix='handseg_'))
