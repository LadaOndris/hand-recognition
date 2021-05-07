class YoloModel:

    def __init__(self):
        self.yolo_output_shapes = None
        self.tf_model = None
        self.learning_rate = 1e-3
        self.anchors = []
        self.yolo_out_preds = []
        self.yolo_output_shapes = []
        self.input_shape = None
