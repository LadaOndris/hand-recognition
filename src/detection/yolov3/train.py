
import tensorflow as tf
import os
import sys
from datetime import datetime

from utils import draw_detected_objects, draw_grid_detection
from model import Model
from datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset
from dataset_generator import DatasetGenerator
from metrics import YoloConfPrecisionMetric, YoloConfRecallMetric, YoloBoxesIoU
from yolo_loss import YoloLoss

# disable CUDA, run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(base_path):
    log_dir = os.path.join(base_path, "logs/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model
    
    #config = Config()
    dataset_bboxes = SimpleBoxesDataset(type='train', train_size=0.8, batch_size=model.batch_size)
    dataset_generator = DatasetGenerator(dataset_bboxes.batch_iterator, 
                                         model.input_shape, yolo_out_shapes, model.anchors)
    
    
    # compile model
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)
    tf_model.compile(optimizer=tf.optimizers.Adam(learning_rate=model.learning_rate), 
                     loss=loss, metrics=[YoloConfPrecisionMetric(), YoloConfRecallMetric(),
                                         YoloBoxesIoU()])
   
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    tf_model.fit(dataset_generator, epochs=50, verbose=1, steps_per_epoch=10,
                 callbacks=[tensorboard_callback])
    
    #train_summary_writer = tf.summary.create_file_writer(log_dir)
    #with train_summary_writer.as_default():
    #    for epoch in range(len(history.history['loss'])):
    #        tf.summary.scalar('history_loss', history.history['loss'][epoch], step=epoch)
            
    model_name = os.path.join(base_path, "saved_models/simple_boxes12.h5")
    tf_model.save_weights(model_name)
    
    
    """
    for images, bboxes in dataset_bboxes.batch_iterator:
        # images.shape is (batch_size, 416, 416, 1)
        # bboxes.shape is (batch_size, 2, 4)
    """
    return

def predict():
    conf_threshold = .5
    
    model = Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    loaded_model = model.tf_model
    loaded_model.load_weights(os.path.join(base_path, "saved_models/simple_boxes8.h5"))
    batch_size = model.batch_size
    #loaded_model = tf.keras.models.load_model("overfitted_model_conf_only", custom_objects={'YoloLoss':YoloLoss}, compile=False)
    
    dataset_bboxes = SimpleBoxesDataset(type='test', train_size=0.8, batch_size=batch_size)
    #yolo_outputs = loaded_model.predict(dataset_bboxes, batch_size=16, steps=1, verbose=1)
    
    for batch_images, batch_bboxes in dataset_bboxes.batch_iterator:
        
        yolo_outputs = loaded_model.predict(batch_images)
        scale1_outputs = tf.reshape(yolo_outputs[0], [batch_size, -1, 6])
        scale2_outputs = tf.reshape(yolo_outputs[1], [batch_size, -1, 6])
        outputs = tf.concat([scale1_outputs, scale2_outputs], axis=1) # outputs for the whole batch
        
        
        draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        draw_detected_objects(batch_images, outputs, [416, 416, 1], conf_threshold)
        
        break


if __name__ == '__main__':
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
    else:
        base_path = "../../../"
        
    train(base_path)


