"""
Comparing Object Instances from the same class and different classes under different viewpoints
"""

import sys
from utils.metrics_helper import (
    matrix_fitness_metric,
    plot_data,
    plot_rre,
    matrix_angular_similarity,
)

from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    scale_point_cloud,
    draw_point_cloud,
    add_gaussian_noise,
    rotate_pointcloud_randomly,
)
from utils.rendering_helper import (
    Camera,
    get_circle_poses,
    render_point_cloud_from_viewpoint,
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

# Constants
DATA_DIR = "/Datasets/ModelNet10/ModelNet10"
FOLDER = "train"
NUM_VIEWPOINTS = 4
VISUALIZE = False

# Sampling Options
MIN_ANGLE = 0
MAX_ANGLE = 360
PC_COUNT = 600
SEQUENTIAL = False

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Benchmark Iterations
    object_classes = ["chair", "table", "monitor", "sofa"]

    object_index_limit = 100

    # Variable Declarations
    dataset_diagonal_mean = []
    dataset_off_diagonal_mean = []
    dataset_off_diagonal_std = []
    rotational_errors = []

    # Viewpoint Sampling Camera
    image_height = 500
    image_width = 500
    camera = Camera(
        scale=1, image_height=image_height, image_width=image_width, fx=400, fy=400
    )
    k = camera.get_intrinsics()
    camera_py = camera.get_pyrender_camera()

    for idx in range(3, object_index_limit):
        # for idx in [1]:
        object_meshes = []
        object_rendering_info = []

        # Load Object Instance and Extract Pointcloud & Rendering Poses
        for object_class in object_classes:
            path_to_file = path_generator(DATA_DIR, object_class, FOLDER, idx)
            v, f = pcu.load_mesh_vf(path_to_file)
            object_meshes.append((v, f))
            pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
            pointcloud, pointcloud_centered, center, scaling_factor = scale_point_cloud(
                pointcloud
            )
            radius = np.max(np.linalg.norm(pointcloud_centered, axis=1)) * 1.0
            world_pose, pyrender_pose = get_circle_poses(
                NUM_VIEWPOINTS,
                MIN_ANGLE,
                MAX_ANGLE,
                radius,
                center,
                sequential=SEQUENTIAL,
            )
            object_rendering_info.append((world_pose, pyrender_pose, scaling_factor))

        for view_idx in range(NUM_VIEWPOINTS - 1):
            ref_object_pointclouds = []
            rescan_object_pointclouds = []
            gt_rotation = []
            for ref_obj_idx, (v, f) in enumerate(object_meshes):
                print(f"Object: ", {object_classes[ref_obj_idx]}, " index: ", idx)

                # Render Viewpoints for Reference Object
                world_poses, pyrender_poses, scaling_factor = object_rendering_info[
                    ref_obj_idx
                ]
                v, f = object_meshes[ref_obj_idx]
                pointcloud = render_point_cloud_from_viewpoint(
                    v,
                    f,
                    camera_py,
                    k,
                    image_width,
                    image_height,
                    PC_COUNT,
                    world_poses[view_idx],
                    pyrender_poses[view_idx],
                    mesh_scale=scaling_factor,
                    visualize=VISUALIZE,
                )

                # pointcloud = add_gaussian_noise(pointcloud, sigma=0.5)
                ref_object_pointclouds.append(pointcloud)
                # draw_point_cloud(pointcloud)

                # Render Viewpoints for Rescan Object
                pointcloud = render_point_cloud_from_viewpoint(
                    v,
                    f,
                    camera_py,
                    k,
                    image_width,
                    image_height,
                    PC_COUNT,
                    world_poses[view_idx + 1],
                    pyrender_poses[view_idx + 1],
                    mesh_scale=scaling_factor,
                    visualize=VISUALIZE,
                )

                # pointcloud, rot_matrix = rotate_pointcloud_randomly(
                #     pointcloud, pure_z_rotation=True
                # )
                # pointcloud = add_gaussian_noise(pointcloud, sigma=0.2)
                rescan_object_pointclouds.append(pointcloud)
                # gt_rotation.append(torch.tensor(rot_matrix))
            # gt_rotation = torch.stack(gt_rotation)

            ref_object_pointclouds = (
                torch.tensor(np.array(ref_object_pointclouds)).cuda().transpose(-1, -2)
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
            # ref_code_se3 = ref_code["z_so3"] + ref_code["t"]
            # rescan_code_se3 = rescan_code["z_so3"] + rescan_code["t"]

            # compute the similarity matrix
            score_mat = matrix_angular_similarity(
                ref_code_invariant, rescan_code_invariant
            )

            diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(score_mat)
            dataset_diagonal_mean.append(diag_mean)
            dataset_off_diagonal_mean.append(off_diag_mean)
            dataset_off_diagonal_std.append(off_diag_std)

            # # Compute the relative transformation matrix
            # R, t, _, _ = kabsch_transformation_estimation(ref_code_se3, rescan_code_se3)
            # rres = rotation_error(R, gt_rotation.cuda())
            # rres = rres.cpu().numpy()
            # for rre in rres:
            #     rotational_errors.append(rre[0])

    plot_data(
        dataset_diagonal_mean, dataset_off_diagonal_mean, dataset_off_diagonal_std
    )
    # plot_rre(rotational_errors, labels=object_classes)
