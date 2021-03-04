import os
from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.core.cfg.cfg_parser import Model
from src.detection.yolov3 import utils
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.paths import ROOT_DIR, LOGS_DIR, HANDSEG_DATASET_DIR


def predict():
    model = Model.from_cfg(YOLO_CONFIG_FILE)

    model.tf_model.load_weights(LOGS_DIR.joinpath("20201016-125612/train_ckpts/ckpt_10"))
    # loaded_model = tf.keras.models.load_model("overfitted_model_conf_only", custom_objects={'YoloLoss':YoloLoss},
    # compile=False)

    handseg_dataset = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=0.8, batch_size=model.batch_size,
                                           shuffle=False)
    # yolo_outputs = loaded_model.predict(dataset_bboxes, batch_size=16, steps=1, verbose=1)

    for batch_images, batch_bboxes in handseg_dataset.test_batch_iterator:
        yolo_outputs = model.tf_model.predict(batch_images)

        # utils.draw_grid(batch_images, yolo_outputs, [416, 416, 1])
        # utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        fig_location = ROOT_DIR.joinpath("documentation/images/yolo_detection_handseg_{}.pdf")
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1],
                                    TEST_YOLO_CONF_THRESHOLD, draw_cells=False,
                                    fig_location=fig_location)
        break


if __name__ == '__main__':
    # disable CUDA, run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    predict()
