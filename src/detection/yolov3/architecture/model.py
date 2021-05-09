class YoloModel:

    def __init__(self):
        self.yolo_output_shapes = None
        self.tf_model = None
        self.anchors = []
        self.yolo_out_preds = []
        self.yolo_output_shapes = []
        self.input_shape = None
        self.batch_size = None
