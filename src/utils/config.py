from src.utils.paths import SRC_DIR

YOLO_CONFIG_FILE = SRC_DIR.joinpath("core/cfg/yolov3-tiny.cfg")
TRAIN_YOLO_CONF_THRESHOLD = .5
TEST_YOLO_CONF_THRESHOLD = .5

class Config:
    
    def __init__(self):
        
        self.IMAGE_SIZE = (480, 640)
        
        self.DETECTION_BATCH_SIZE = 16
    