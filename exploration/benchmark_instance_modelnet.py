"""
Comparing Object Instances from the same class and different classes
"""

import sys
from utils.metrics_helper import (
    matrix_fitness_metric,
    plot_data,
    plot_dataset,
    matrix_angular_similarity,
)

from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    draw_point_cloud,
    add_gaussian_noise,
    rotate_pointcloud_randomly,
)

sys.path.append("../")

import torch
import random
import numpy as np
import pandas as pd
from lib_math import torch_se3
import point_cloud_utils as pcu
from pytorch3d.ops import sample_farthest_points as fps
from tqdm import tqdm
from lib_more.more_solver import More_Solver
from lib_more.pose_estimation import kabsch_transformation_estimation, rotation_error

from lib_more.pose_estimation import *
from pycg import vis
from lib_more.utils import (
    read_list_from_txt,
    load_json,
    load_yaml,
    visualize_shape_matching,
)
from evaluate import (
    compute_chamfer_distance,
    chamfer_distance_torch,
    compute_sdf_recall,
    compute_volumetric_iou,
)


def create_instance_combinations(num_scenes, num_objects_per_scene, object_index_limit):
    return [
        random.sample(range(1, object_index_limit), num_objects_per_scene)
        for _ in range(num_scenes)
    ]


# Test Parameters
DATA_DIR = "/Datasets/ModelNet10/ModelNet10"
FOLDER = "train"
NUM_SCENES = 10
NUM_ITERATIONS = 4
NUM_OBJECTS_PER_SCENE = 4
OBJECT_INDEX_LIMIT = 100
PC_COUNT = 600

VISUALIZE = False

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Benchmark Iterations
    # object_classes = ["chair", "table", "monitor", "sofa"]
    object_classes = [
        "bathtub",
        "bed",
        "chair",
        "desk",
        "dresser",
        "monitor",
        "night_stand",
        "sofa",
        "table",
        "toilet",
    ]
    # object_classes = ["chair"]

    # Variable Declarations
    dataset_diagonal_mean = {}
    dataset_off_diagonal_mean = {}
    dataset_off_diagonal_std = {}
    # rotational_errors = {}

    # Scene ID Combinations
    scene_combinations = create_instance_combinations(
        NUM_SCENES, NUM_OBJECTS_PER_SCENE, OBJECT_INDEX_LIMIT
    )

    # Iterate over all the object classes
    for object_class in object_classes:
        # Variable Declarations
        class_diagonal_mean = []
        class_off_diagonal_mean = []
        class_off_diagonal_std = []
        # class_rotational_errors = []

        # Iterate over all the scenes
        for scene in scene_combinations:
            # Repreat each scene NUM_ITERATIONS times
            print(f"Object Class: {object_class}, Scene: {scene}")
            object_meshes = []
            for idx in scene:  # Load Object Meshes
                path_to_file = path_generator(DATA_DIR, object_class, FOLDER, idx)
                v, f = pcu.load_mesh_vf(path_to_file)
                object_meshes.append((v, f))
            for _ in range(NUM_ITERATIONS):
                ref_object_pointclouds = []
                rescan_object_pointclouds = []
                gt_rotation = []
                for i, (v, f) in enumerate(object_meshes):
                    pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
                    pointcloud = add_gaussian_noise(pointcloud, sigma=0.5)
                    ref_object_pointclouds.append(pointcloud)

                    if VISUALIZE:
                        draw_point_cloud(pointcloud)

                    pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
                    pointcloud, rot_matrix = rotate_pointcloud_randomly(
                        pointcloud, pure_z_rotation=True
                    )
                    pointcloud = add_gaussian_noise(pointcloud, sigma=0.2)
                    # draw_point_cloud(pointcloud)
                    rescan_object_pointclouds.append(pointcloud)
                    # gt_rotation.append(torch.tensor(rot_matrix))

                # gt_rotation = torch.stack(gt_rotation)
                ref_object_pointclouds = (
                    torch.tensor(np.array(ref_object_pointclouds))
                    .cuda()
                    .transpose(-1, -2)
                )
                rescan_object_pointclouds = (
                    torch.tensor(np.array(rescan_object_pointclouds))
                    .cuda()
                    .transpose(-1, -2)
                )

                # print(ref_object_pointclouds.shape)
                # print(rescan_object_pointclouds.shape)

                with torch.no_grad():
                    ref_code = model.encode(ref_object_pointclouds)
                    rescan_code = model.encode(rescan_object_pointclouds)

                ref_code_invariant = ref_code["z_inv"]
                rescan_code_invariant = rescan_code["z_inv"]
                ref_code_se3 = ref_code["z_so3"] + ref_code["t"]
                rescan_code_se3 = rescan_code["z_so3"] + rescan_code["t"]

                # compute the similarity matrix
                score_mat = matrix_angular_similarity(
                    ref_code_invariant, rescan_code_invariant
                )

                diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(
                    score_mat
                )
                class_diagonal_mean.append(diag_mean)
                class_off_diagonal_mean.append(off_diag_mean)
                class_off_diagonal_std.append(off_diag_std)

        # Save the data at class level
        dataset_diagonal_mean[object_class] = class_diagonal_mean
        dataset_off_diagonal_mean[object_class] = class_off_diagonal_mean
        dataset_off_diagonal_std[object_class] = class_off_diagonal_std

    # Plot the data
    for object_class in object_classes:
        print(f"Object Class: {object_class}")
        plot_data(
            dataset_diagonal_mean[object_class],
            dataset_off_diagonal_mean[object_class],
            dataset_off_diagonal_std[object_class],
            class_title=object_class,
        )

    # Plot Whole dataset togther
    plot_dataset(
        dataset_diagonal_mean,
        dataset_off_diagonal_mean,
        dataset_off_diagonal_std,
        same_color=True,
    )
