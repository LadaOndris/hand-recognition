"""
It was found that the BigHand annotation files contain annotations for
files that do not exist. This script checks that the length of annotation files
is equal to the number of files in corresponding directories.
"""


from src.utils.paths import BIGHAND_DATASET_DIR
import os
import glob
from pathlib import Path


def file_lines(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def check_annotation_file(annot_filepath, filesdir):
    annot_lines = file_lines(annot_filepath)
    files_count = len(os.listdir(filesdir))
    if annot_lines != files_count:
        print(F"Annots: {annot_lines}, files: {files_count}, in: {annot_filepath}")


def check_files():
    print("Any inconsistencies are printed below:")
    subject_dirs_paths = [f for f in BIGHAND_DATASET_DIR.iterdir() if f.is_dir()]
    subject_dirs = [f.stem for f in subject_dirs_paths]
    for subject_dir, subject_dir_path in zip(subject_dirs, subject_dirs_paths):
        pattern = F"full_annotation/{subject_dir}/[!README]*.txt"
        full_pattern = os.path.join(BIGHAND_DATASET_DIR, pattern)
        annotation_files = glob.glob(full_pattern)

        for annot_file in annotation_files:
            annot_filename = Path(annot_file).stem
            annot_name = annot_filename.split('_')[0]
            annot_files_dir_path = subject_dir_path.joinpath(annot_name)
            check_annotation_file(annot_file, annot_files_dir_path)


if __name__ == '__main__':
    check_files()
