import sys
import os

sys.path.append("../")
from utils.metrics_helper import compute_pointcloud_overlap
import numpy as np
import pyrender
import trimesh
import random
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    draw_point_cloud,
    draw_point_cloud_with_cameras,
    scale_point_cloud,
    rotate_pointcloud,
)


def idealworks_path_generator(data_dir, class_name, extention=".obj"):
    """
    Generate the path for the Idealworks dataset, assumes a single instance per class
    """
    files_in_class = [
        file
        for file in os.listdir(os.path.join(data_dir, class_name))
        if file.endswith(extention)
    ]
    assert len(files_in_class) == 1, f"Expected 1 object in class {class_name}"
    return os.path.join(data_dir, class_name, files_in_class[0])


if __name__ == "__main__":
    # Define Paths
    DATASET_DIR = "/Datasets/Idealworks_assests"
    object_classes = os.listdir(DATASET_DIR)
    print(f"Object Classes: {object_classes}")

    # Rotate the point cloud 90 degrees around X axis
    r = R.from_euler("x", 90, degrees=True).as_matrix()

    # Load the point clouds
    for object_class in object_classes:
        print(f"Object Class: {object_class}")
        path_to_file = idealworks_path_generator(DATASET_DIR, object_class)
        v, f = pcu.load_mesh_vf(path_to_file)
        pointcloud = sample_mesh_random(v, f, num_samples=1000)
        pointcloud, _ = rotate_pointcloud(pointcloud, r)
        draw_point_cloud(pointcloud, title=object_class)
        plt.show()
