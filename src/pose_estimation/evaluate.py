from src.datasets.bighand.dataset import BighandDataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.utils.paths import BIGHAND_DATASET_DIR


def evaluate():
    # load model!
    network = JGR_J2O()
    model = network.graph()

    # initialize dataset
    im_out_size = 24
    bighand_ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
    gen = DatasetGenerator(iter(bighand_ds.test_dataset), im_out_size)

    for batch in gen:
        images, y_true = batch
        model.predict()
