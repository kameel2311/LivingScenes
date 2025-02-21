import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append("../")
from utils.pointcloud_helper import path_generator
from utils.idealworks_assests_helper import idealworks_path_generator


class Dataloader:
    def __init__(self, dataset, object_idx_limit):
        self.dataset = dataset
        self.object_idx_limit = object_idx_limit
        self.rotation_alignment_matrix = np.eye(3)
        self.set_metadata()
        self.object_classes = os.listdir(self.data_dir)

    def set_metadata(self):
        if self.dataset == "ModelNet10":
            self.data_dir = "/Datasets/ModelNet10/ModelNet10"
            self.folder = "train"
            self.noise = 0.5

        elif self.dataset == "Idealworks":
            self.data_dir = "/Datasets/Idealworks_assests"
            self.folder = ""
            self.object_idx_limit = 1  # Because there is only one object per class
            self.noise = 0.05
            self.rotation_alignment_matrix = R.from_euler(
                "x", 90, degrees=True
            ).as_matrix()
        else:
            raise ValueError("Invalid dataset")

    def get_path(self, object_class, idx):
        if self.dataset == "ModelNet10":
            return path_generator(self.data_dir, object_class, self.folder, idx)
        elif self.dataset == "Idealworks":
            return idealworks_path_generator(self.data_dir, object_class)
        else:
            raise ValueError("Invalid dataset")

    def get_metadata(self):
        return (
            self.object_classes,
            self.object_idx_limit,
            self.noise,
            self.rotation_alignment_matrix,
        )
