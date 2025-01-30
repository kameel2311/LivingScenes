"""
Comparing Object Instances from the same class and different classes
"""

import sys
import logging
from utils.metrics_helper import (
    matrix_fitness_metric,
    plot_data,
    plot_rre,
    matrix_angular_similarity,
)
from utils.dataloader import Dataloader
from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    draw_point_cloud,
    add_gaussian_noise,
    rotate_pointcloud_randomly,
    rotate_pointcloud,
)

sys.path.append("../")

import torch
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

NUM_ITERATIONS = 2
MAX_IDX = 50
PC_COUNT = 600
VISUALIZE = False

logging.basicConfig(
    filename="benchmark_classes_rotation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Benchmark Iterations
    dataloader = Dataloader("ModelNet10", MAX_IDX)  # "ModelNet10" or "Idealworks"
    object_classes, object_index_limit, noise_std, alignment_matrix = (
        dataloader.get_metadata()
    )

    # object_classes = [
    #     "pallet_1",
    #     "pallet_2",
    #     "cone_2",
    #     "bin",
    #     "cone_1",
    #     "klt",
    #     "rack_boxes",
    #     "wooden_crate",
    #     "purple_container",
    #     "table",
    #     "Ru2_Dolly",
    #     "glt",
    #     "wet_floor_sign",
    #     "wet_floor_sign_2",
    # ]

    # Console Output
    print(f"Object Classes: {object_classes}")
    print(f"Object Index Limit: {object_index_limit}")
    logger.info(f"Object Classes: {object_classes}")

    # Variable Declarations
    dataset_diagonal_mean = []
    dataset_off_diagonal_mean = []
    dataset_off_diagonal_std = []
    rotational_errors = []
    for idx in range(0, object_index_limit):
        # Load Object Instance
        object_meshes = []
        for object_class in object_classes:
            path_to_file = dataloader.get_path(object_class, idx)
            v, f = pcu.load_mesh_vf(path_to_file)
            object_meshes.append((v, f))

        for _ in range(NUM_ITERATIONS):
            ref_object_pointclouds = []
            rescan_object_pointclouds = []
            gt_rotation = []
            for i, (v, f) in enumerate(object_meshes):
                print(f"Object: ", {object_classes[i]}, " index: ", idx)
                pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
                pointcloud, _ = rotate_pointcloud(pointcloud, alignment_matrix)
                pointcloud = add_gaussian_noise(pointcloud, sigma=noise_std)
                ref_object_pointclouds.append(pointcloud)

                if VISUALIZE:
                    draw_point_cloud(pointcloud)

                pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
                pointcloud, _ = rotate_pointcloud(pointcloud, alignment_matrix)
                pointcloud, rot_matrix = rotate_pointcloud_randomly(
                    pointcloud, pure_z_rotation=True
                )
                pointcloud = add_gaussian_noise(pointcloud, sigma=noise_std)
                # draw_point_cloud(pointcloud)
                rescan_object_pointclouds.append(pointcloud)
                gt_rotation.append(torch.tensor(rot_matrix))
            gt_rotation = torch.stack(gt_rotation)

            ref_object_pointclouds = (
                torch.tensor(np.array(ref_object_pointclouds)).cuda().transpose(-1, -2)
            )
            rescan_object_pointclouds = (
                torch.tensor(np.array(rescan_object_pointclouds))
                .cuda()
                .transpose(-1, -2)
            )

            print(ref_object_pointclouds.shape)
            print(rescan_object_pointclouds.shape)

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

            diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(score_mat)
            dataset_diagonal_mean.append(diag_mean)
            dataset_off_diagonal_mean.append(off_diag_mean)
            dataset_off_diagonal_std.append(off_diag_std)

            # Compute the relative transformation matrix
            R, t, _, _ = kabsch_transformation_estimation(ref_code_se3, rescan_code_se3)
            rres = rotation_error(R, gt_rotation.cuda())
            rres = rres.cpu().numpy()
            for rre in rres:
                rotational_errors.append(rre[0])

            # Logging friendly format
            logging_rotations = {}
            for object_class, rre in zip(object_classes, rres):
                logging_rotations[object_class] = rre[0]
            logger.info(f"Object Index: {idx}, Rotational Errors: {logging_rotations}")

    plot_data(
        dataset_diagonal_mean, dataset_off_diagonal_mean, dataset_off_diagonal_std
    )
    plot_rre(rotational_errors, labels=object_classes)
