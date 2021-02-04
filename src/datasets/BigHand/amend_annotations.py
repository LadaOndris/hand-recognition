"""
The annotations files of the BigHand dataset contain
only relative indexing of images. This script adds a folder
to which the index refers to so it can be easily loaded
in the tf.data pipeline.
"""

from src.utils.paths import BIGHAND_DATASET_DIR
import glob
import os
import pandas as pd

annotation_files_pattern = os.path.join(BIGHAND_DATASET_DIR, 'full_annotation/*/[!README]*.txt')
filenames = glob.glob(annotation_files_pattern)


def get_dir_from_filename(filename: str) -> str:
    """
    FOR FILENAME: full_annotation/Subject_1/301\ 375_loc_shift_made_by_qi_20180112_v2.txt
    RETURNS DIR: Subject_1/301\ 375
    """
    parts = filename.split(os.sep)
    name = parts[-1]
    parent = parts[-2]  # Subject_1
    angle = name.split('_')[0]  # 301\ 375
    return os.path.join(parent, angle)


def load_df(filename: str) -> pd.DataFrame:
    return pd.read_table(filename, header=None, dtype=str)


def amend_df(df: pd.DataFrame, directory: str) -> pd.DataFrame:
    df[0] = directory + "/image_D" + df[0] + ".png"
    return df


def save_df(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, sep='\t', index=False, header=False)


# read each annotation file and replace the first column
for filename in filenames:
    directory = get_dir_from_filename(filename)
    df = load_df(filename)
    if len(df.loc[0, 0]) <= 10:
        print(F"Amending annotations for: {directory}")
        df = amend_df(df, directory)
        save_df(df, filename)
