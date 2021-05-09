class Config:

    def __init__(self):
        self.return_xyz = False
        self.augment = False
        self.cube_size = 200,
        self.refine_iters = 4
        self.thresholding = True
        self.use_center_of_image = True
        self.ignore_threshold_otsus = 0.01

        self.learning_rate = None
        self.learning_decay_rate = None


class MsraConfig(Config):

    def __init__(self):
        Config.__init__(self)
        self.cube_size = 180


class BighandConfig(Config):

    def __init__(self):
        Config.__init__(self)
        self.cube_size = 150


"""
Train configurations
"""


class TrainMsraConfig(MsraConfig):

    def __init__(self):
        MsraConfig.__init__(self)
        self.augment = True
        self.thresholding = False
        self.refine_iters = 0
        self.use_center_of_image = False


class TrainBighandConfig(BighandConfig):

    def __init__(self):
        BighandConfig.__init__(self)
        self.augment = True
        self.thresholding = False
        self.refine_iters = 0
        self.use_center_of_image = True


"""
Test configurations
"""


class TestMsraConfig(TrainMsraConfig):

    def __init__(self):
        TrainMsraConfig.__init__(self)
        self.augment = False


class TestBighandConfig(TrainBighandConfig):

    def __init__(self):
        TrainBighandConfig.__init__(self)
        self.augment = False


"""
Evaluation configurations
"""


class EvaluateMsraConfig(MsraConfig):

    def __init__(self):
        MsraConfig.__init__(self)
        self.augment = False
        self.thresholding = False
        self.refine_iters = 0


class EvaluateBighandConfig(BighandConfig):

    def __init__(self):
        BighandConfig.__init__(self)
        self.augment = False
        self.thresholding = False
        self.refine_iters = 0
        self.cube_size = 160
        self.use_center_of_image = True


"""
Predictions on datasets configurations
"""


class PredictMsraConfig(TestMsraConfig):

    def __init__(self):
        TestMsraConfig.__init__(self)
        self.use_center_of_image = False


class PredictBighandConfig(TestBighandConfig):

    def __init__(self):
        TestBighandConfig.__init__(self)
        self.cube_size = 160
        self.use_center_of_image = False


class PredictCustomDataset(Config):

    def __init__(self):
        Config.__init__(self)
        self.cube_size = 160
        self.use_center_of_image = False
        self.return_xyz = False
        self.augment = False
        self.refine_iters = 0
        self.thresholding = True
