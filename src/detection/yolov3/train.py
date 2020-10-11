
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(base_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_path, "logs/", timestamp)
    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model
    
    #config = Config()
    dataset_path = os.path.join(base_path, "datasets/handseg150k")
    dataset_bboxes = HandsegDatasetBboxes(dataset_path, type='train', train_size=0.8, 
                                          batch_size=model.batch_size)
    dataset_generator = DatasetGenerator(dataset_bboxes.batch_iterator, 
                                         model.input_shape, yolo_out_shapes, model.anchors)
    
    
    # compile model
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)
    tf_model.compile(optimizer=tf.optimizers.Adam(learning_rate=model.learning_rate), 
                     loss=loss, metrics=[YoloConfPrecisionMetric(), YoloConfRecallMetric(),
                                         YoloBoxesIoU()])
   
    checkpoint_prefix = os.path.join(log_dir, 'train_ckpts', "ckpt_{epoch}")
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    ]
    
    tf_model.fit(dataset_generator, epochs=10, verbose=0, steps_per_epoch=3125, callbacks=callbacks)
    
    #train_summary_writer = tf.summary.create_file_writer(log_dir)
    #with train_summary_writer.as_default():
    #    for epoch in range(len(history.history['loss'])):
    #        tf.summary.scalar('history_loss', history.history['loss'][epoch], step=epoch)
            
    model_name = os.path.join(base_path, F"saved_models/handseg_{timestamp}.h5")
    tf_model.save_weights(model_name)
    
    
    """
    for images, bboxes in dataset_bboxes.batch_iterator:
        # images.shape is (batch_size, 416, 416, 1)
        # bboxes.shape is (batch_size, 2, 4)
    """
    return

def predict(base_path):
    conf_threshold = .5
    
    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    loaded_model = model.tf_model
    #loaded_model.load_weights(os.path.join(base_path, "saved_models/simple_boxes8.h5"))
    loaded_model.load_weights(os.path.join(base_path, "logs/20201009-181239/train_ckpts/ckpt_3"))
    batch_size = model.batch_size
    #loaded_model = tf.keras.models.load_model("overfitted_model_conf_only", custom_objects={'YoloLoss':YoloLoss}, compile=False)
    
    dataset_path = os.path.join(base_path, "datasets/handseg150k")
    dataset_bboxes = HandsegDatasetBboxes(dataset_path, type='test', train_size=0.8, batch_size=batch_size)
    #yolo_outputs = loaded_model.predict(dataset_bboxes, batch_size=16, steps=1, verbose=1)
    
    for batch_images, batch_bboxes in dataset_bboxes.batch_iterator:
        
        yolo_outputs = loaded_model.predict(batch_images)
        scale1_outputs = tf.reshape(yolo_outputs[0], [batch_size, -1, 6])
        scale2_outputs = tf.reshape(yolo_outputs[1], [batch_size, -1, 6])
        outputs = tf.concat([scale1_outputs, scale2_outputs], axis=1) # outputs for the whole batch
        
        
        utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        utils.draw_detected_objects(batch_images, outputs, [416, 416, 1], conf_threshold)
        
        break


if __name__ == '__main__':
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
    else:
        base_path = "../../../"
        
    predict(base_path)


