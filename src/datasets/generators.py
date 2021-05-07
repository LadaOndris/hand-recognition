from src.datasets.custom.dataset import CustomDataset, CustomDatasetGenerator
from src.utils.live import generate_live_images
from src.utils.paths import CUSTOM_DATASET_DIR


def get_source_generator(source_type: str):
    source_type = source_type.lower()
    if source_type == 'live':
        return generate_live_images()
    elif source_type == 'dataset':
        return get_custom_dataset_generator()
    else:
        raise ValueError(f'Invalid source type: {source_type}')


def get_custom_dataset_generator():
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=1)
    generator = CustomDatasetGenerator(ds)
    return generator
