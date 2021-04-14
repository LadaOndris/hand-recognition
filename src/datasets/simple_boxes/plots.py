from dataset_bboxes import SimpleBoxesDataset
from src.datasets.handseg.analysis import draw_bounding_box, show_images_with_bboxes


def show_images(num):
    dataset = SimpleBoxesDataset(batch_size=num)
    show_images_with_bboxes(dataset)


show_images(5)
