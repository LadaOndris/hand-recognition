from src.utils.paths import USECASE_DATASET_DIR
import numpy as np


class UsecaseDatabaseReader:

    def __init__(self):
        self.hand_poses = None
        self.labels = None

    def load_from_subdir(self, subdir):
        annotation_files = self._find_annotation_files(subdir)
        annotation_labels = self._extract_annotation_labels(annotation_files)
        self.hand_poses, self.labels = self._read_hand_poses(annotation_files, annotation_labels)

    def _find_annotation_files(self, subdir):
        subdir_path = USECASE_DATASET_DIR.joinpath(subdir)
        return subdir_path.iterdir()

    def _extract_annotation_labels(self, annotation_files):
        labels = []
        for file in annotation_files:
            file_name = file.stem
            name_parts = file_name.split('_')
            if len(name_parts) != 2:
                raise Exception(
                    F"Unexpected file name: {file_name}. Expected name given by "
                    "two components: label_timestamp")
            label, timestamp = name_parts
            labels.append(label)
        return labels

    def _read_hand_poses(self, files, labels):
        hand_poses_arrs = []
        labels_arrs = []

        for file, label in zip(files, labels):
            joints_flattened = np.genfromtxt(file, dtype=np.float, delimiter=' ')  # (lines, 63)
            joints_arr = np.reshape(joints_flattened, [joints_flattened.shape[0], -1, 3])
            labels_arr = np.full(shape=[joints_flattened.shape[0]], fill_value=label)
            hand_poses_arrs.append(joints_arr)
            labels_arrs.append(labels_arr)

            hand_poses = np.concatenate(hand_poses_arrs)
            labels = np.concatenate(labels_arrs)
        return hand_poses, labels

    def get_label_by_index(self, hand_pose_index):
        if self.labels is None:
            raise Exception("No hand poses are loaded")
        if hand_pose_index < 0 or hand_pose_index >= np.shape(self.labels)[0]:
            raise Exception("Index out of bounds")
        return self.labels[hand_pose_index]
