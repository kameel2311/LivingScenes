"""
Comparing Object Instances from the same class and different classes
"""

import sys

from utils.metrics_helper import (
    angular_similarity,
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
    draw_point_cloud,
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
from pytorch3d.ops.points_alignment import iterative_closest_point, SimilarityTransform


NUM_ITERATIONS = 10
MAX_IDX = 100
PC_COUNT = 600
VISUALIZE = True

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Benchmark Iterations
    dataloader = Dataloader("Idealworks", MAX_IDX)  # "ModelNet10" or "Idealworks"
    object_classes, object_index_limit, noise_std, alignment_matrix = (
        dataloader.get_metadata()
    )

    object_class = "bed"
    idx = 23
    for object_class in object_classes:
        # Sample object twice
        v, f = pcu.load_mesh_vf(dataloader.get_path(object_class, idx))
        pointcloud_1 = sample_mesh_random(v, f, num_samples=PC_COUNT)
        pointcloud_1, _ = rotate_pointcloud(pointcloud_1, alignment_matrix)
        pointcloud_1 = add_gaussian_noise(pointcloud_1, sigma=noise_std)

        pointcloud_2 = sample_mesh_random(v, f, num_samples=PC_COUNT)
        pointcloud_2, _ = rotate_pointcloud(pointcloud_2, alignment_matrix)
        pointcloud_2, rotation = rotate_pointcloud_randomly(
            pointcloud_2, pure_z_rotation=True, about_center=True, identity=True
        )
        pointcloud_2 = add_gaussian_noise(pointcloud_2, sigma=noise_std)
        gt_rotation = torch.tensor(rotation)

        object_pointclouds = (
            torch.tensor(np.array([pointcloud_1, pointcloud_2]))
            .cuda()
            .transpose(-1, -2)
        )

        with torch.no_grad():
            object_code = model.encode(object_pointclouds)

        for key, value in object_code.items():
            print(key, value.shape)

        pc_1_invariant = object_code["z_inv"][0]
        pc_2_invariant = object_code["z_inv"][1]
        pc_1_se3 = (object_code["z_so3"][0] + object_code["t"][0]).unsqueeze(0)
        pc_2_se3 = (object_code["z_so3"][1] + object_code["t"][1]).unsqueeze(0)

        # compute the similarity matrix
        score = angular_similarity(pc_1_invariant, pc_2_invariant)

        # Compute the relative transformation matrix
        print(pc_1_se3.shape, pc_2_se3.shape)
        R, t, _, _ = kabsch_transformation_estimation(pc_1_se3, pc_2_se3)
        # s0 = torch.tensor([1]).float().cuda()
        # icp_solution = iterative_closest_point(
        #     object_pointclouds[0].unsqueeze(0),
        #     object_pointclouds[1].unsqueeze(0),
        #     init_transform=SimilarityTransform(R.transpose(-1, -2), t.squeeze(2), s0),
        # )
        # R, t, _ = icp_solution[3]
        print(R, t)
        print(gt_rotation)
        rres = rotation_error(R, gt_rotation.cuda())
        rres = rres.cpu().numpy()
        estimated_rotation = R.cpu().numpy().squeeze()
        print("Estimated Rotation: ", estimated_rotation)

        print("Angular Similarity: ", score)
        print("Relative Rotation Error: ", rres)
        draw_point_cloud(
            pointcloud_1, overlay_pointcloud=pointcloud_2, title="Actual Pointclouds"
        )
        draw_point_cloud(
            pointcloud_1,
            overlay_pointcloud=rotate_pointcloud(pointcloud_2, rotation.T)[0],
            title="Aligned with GT Rotation",
        )
        draw_point_cloud(
            pointcloud_1,
            overlay_pointcloud=rotate_pointcloud(pointcloud_2, estimated_rotation.T)[0],
            title=f"Aligned with Estimated Rotation Class {object_class} (Error: {rres})",
        )
