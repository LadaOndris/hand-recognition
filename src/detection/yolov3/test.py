import src.utils.plots_detection
from src.datasets.handseg.dataset_bboxes import HandsegDatasetBboxes
from src.detection.yolov3 import utils
from src.detection.yolov3.cfg.cfg_parser import YoloLoader
from src.detection.yolov3.dataset_preprocessing import DatasetPreprocessor
from src.utils.config import RESIZE_MODE, TEST_YOLO_CONF_THRESHOLD
from src.utils.paths import DOCS_DIR, HANDSEG_DATASET_DIR


def predict_on_handseg(plot_prediction=False, plot_prediction_with_grid=False, plot_image_with_grid=False,
                       plot_cells=False, fig_pattern=None):
    model = YoloLoader.load_from_weights(RESIZE_MODE)
    handseg = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=0.99, batch_size=model.batch_size,
                                   shuffle=False, model_input_shape=model.input_shape)
    preprocessor = DatasetPreprocessor(handseg.test_batch_iterator,
                                       model.input_shape, model.yolo_output_shapes, model.anchors)

    for batch_images, batch_bboxes in preprocessor:
        yolo_outputs = model.tf_model.predict(batch_images)

        if plot_image_with_grid:
            fig_location = format_or_none(fig_pattern, 'img_grid')
            src.utils.plots_detection.plot_grid(batch_images, yolo_outputs, [416, 416, 1], fig_location=fig_location)
        if plot_prediction_with_grid:
            fig_location = format_or_none(fig_pattern, 'preds_grid')
            src.utils.plots_detection.plot_grid_detection(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD,
                                                          fig_location=fig_location)
        if plot_prediction:
            fig_location = format_or_none(fig_pattern, 'preds')
            src.utils.plots_detection.plot_detected_objects(batch_images, yolo_outputs, [416, 416, 1],
                                                            TEST_YOLO_CONF_THRESHOLD, draw_cells=plot_cells,
                                                            fig_location=fig_location)
        break


def format_or_none(pattern, str):
    if pattern is None:
        return None
    return pattern.format(str)


if __name__ == '__main__':
    fig_location_pattern = str(DOCS_DIR.joinpath('figures/architecture/yolo_{}.pdf'))
    predict_on_handseg(plot_prediction=True, plot_image_with_grid=True, plot_prediction_with_grid=True,
                       plot_cells=True, fig_pattern=fig_location_pattern)
