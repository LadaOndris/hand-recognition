import tensorflow as tf
import os
import sys
from datetime import datetime

from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset
from src.detection.yolov3.dataset_generator import DatasetGenerator
from src.detection.yolov3.metrics import YoloConfPrecisionMetric, YoloConfRecallMetric, YoloBoxesIoU
from src.detection.yolov3.yolo_loss import YoloLoss
from src.detection.yolov3.model import Model
from src.detection.yolov3 import utils


# disable CUDA, run on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(base_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_path, "logs/", timestamp)
    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model

    dataset_path = os.path.join(base_path, "datasets/handseg150k")
    train_dataset_bboxes = HandsegDatasetBboxes(dataset_path, type='train', train_size=0.8,
                                                batch_size=model.batch_size)
    train_dataset_generator = DatasetGenerator(train_dataset_bboxes.batch_iterator,
                                               model.input_shape, yolo_out_shapes, model.anchors)

    test_dataset_bboxes = HandsegDatasetBboxes(dataset_path, type='test', train_size=0.8,
                                               batch_size=model.batch_size)
    test_dataset_generator = DatasetGenerator(test_dataset_bboxes.batch_iterator,
                                              model.input_shape, yolo_out_shapes, model.anchors)

    optimizer = tf.optimizers.Adam(learning_rate=model.learning_rate)
    metrics = [YoloConfPrecisionMetric(), YoloConfRecallMetric(), YoloBoxesIoU()]
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)

    tf_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    checkpoint_prefix = os.path.join(log_dir, 'train_ckpts', "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    ]

    tf_model.fit(train_dataset_generator, epochs=10, verbose=1, callbacks=callbacks,
                 steps_per_epoch=train_dataset_bboxes.num_batches,
                 validation_data=test_dataset_generator,
                 validation_steps=test_dataset_bboxes.num_batches)

    model_name = os.path.join(base_path, F"saved_models/handseg_{timestamp}.h5")
    tf_model.save_weights(model_name)

    return


def predict(base_path):
    conf_threshold = .5

    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    loaded_model = model.tf_model
    # loaded_model.load_weights(os.path.join(base_path, "saved_models/simple_boxes8.h5"))
    loaded_model.load_weights(os.path.join(base_path, "logs/20201016-125612/train_ckpts/ckpt_10"))
    batch_size = model.batch_size
    # loaded_model = tf.keras.models.load_model("overfitted_model_conf_only", custom_objects={'YoloLoss':YoloLoss}, compile=False)

    dataset_path = os.path.join(base_path, "datasets/handseg150k")
    dataset_bboxes = HandsegDatasetBboxes(dataset_path, type='test', train_size=0.8, batch_size=batch_size,
                                          shuffle=False)
    # yolo_outputs = loaded_model.predict(dataset_bboxes, batch_size=16, steps=1, verbose=1)

    for batch_images, batch_bboxes in dataset_bboxes.batch_iterator:
        yolo_outputs = loaded_model.predict(batch_images)

        # utils.draw_grid(batch_images, yolo_outputs, [416, 416, 1])
        # utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        fig_location = os.path.join(base_path, "documentation/images/yolo_detection_handseg_{}.pdf")
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1],
                                    conf_threshold, draw_cells=False,
                                    fig_location=fig_location)
        break


if __name__ == '__main__':
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
    else:
        base_path = "../../../"

    predict(base_path)
