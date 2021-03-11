import tensorflow as tf

from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.detection.yolov3.dataset_generator import DatasetGenerator
from src.detection.yolov3.metrics import YoloConfPrecisionMetric, YoloConfRecallMetric, YoloBoxesIoU
from src.detection.yolov3.yolo_loss import YoloLoss
from src.core.cfg.cfg_parser import Model
from src.utils.config import YOLO_CONFIG_FILE
from src.utils.paths import HANDSEG_DATASET_DIR
import src.utils.logs as logs_utils


def train():
    model = Model.from_cfg(YOLO_CONFIG_FILE)
    yolo_out_shapes = model.yolo_output_shapes

    handseg_dataset = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=0.8, batch_size=model.batch_size)
    train_dataset_generator = DatasetGenerator(handseg_dataset.train_batch_iterator,
                                               model.input_shape, yolo_out_shapes, model.anchors)
    test_dataset_generator = DatasetGenerator(handseg_dataset.test_batch_iterator,
                                              model.input_shape, yolo_out_shapes, model.anchors)

    optimizer = tf.optimizers.Adam(learning_rate=model.learning_rate)
    metrics = [YoloConfPrecisionMetric(), YoloConfRecallMetric(), YoloBoxesIoU()]
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)

    model.tf_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    model.tf_model.fit(train_dataset_generator, epochs=40, verbose=0, callbacks=callbacks,
                       steps_per_epoch=handseg_dataset.num_train_batches,
                       validation_data=test_dataset_generator,
                       validation_steps=handseg_dataset.num_test_batches)

    model.tf_model.save_weights(logs_utils.compose_model_path(prefix='handseg_'))


if __name__ == '__main__':
    train()
