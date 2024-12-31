"""
Comparing Object Instances from the same class and different classes
"""

import sys

sys.path.append("../")

import os, sys, yaml, shutil
import glob
import torch
import numpy as np
import os.path as osp
from lib_math import torch_se3
import trimesh
import point_cloud_utils as pcu
from pytorch3d.ops import sample_farthest_points as fps
from tqdm import tqdm
from lib_more.more_solver import More_Solver

from lib_more.pose_estimation import *
from pycg import vis
import logging, coloredlogs
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

from point_cloud import *


def set_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level="INFO", logger=logger)
    file_handler = logging.FileHandler(log_path, mode="w")
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


def to_tensor(v, device):
    v = torch.from_numpy(v)
    v = torch.reshape(v, (1, 3, v.shape[0])).to(device)
    return v


def angular_similarity(a, b):
    """
    Compute the angular similarity between two tensors
    """
    # a = torch.squeeze(a)
    # b = torch.squeeze(b)
    a = torch.flatten(a)
    b = torch.flatten(b)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


DATA_DIR = "/Datasets/ModelNet10/ModelNet10"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Load the model
    solver_cfg = load_yaml("/workspace/configs/more_3rscan.yaml")
    solver = More_Solver(solver_cfg)
    model_encoder = solver.model.encoder
    print(type(model_encoder))

    # Prepare the model
    device = torch.device("cuda")
    model_encoder = model_encoder.to(device)
    model_encoder.eval()

    # Benchmark Iterations
    object_class_1 = "chair"
    object_class_2 = "table"
    file = "train"
    num_samples = [600, 400, 200]
    instances = [40]
    overlap = 0.5
    sample_1_lower_limit = 310
    sample_1_upper_limit = 100
    sample_2_lower_limit = 200
    sample_2_upper_limit = 30

    for instance in instances:
        # Load Object Instance
        path_to_file = path_generator(DATA_DIR, object_class_1, file, instance)
        v_1, f_1 = pcu.load_mesh_vf(path_to_file)
        print(f"Vertices: {v_1.shape}, Faces: {f_1.shape}")
        path_to_file = path_generator(DATA_DIR, object_class_2, file, instance)
        v_2, f_2 = pcu.load_mesh_vf(path_to_file)
        print(f"Vertices: {v_2.shape}, Faces: {f_2.shape}")

        # Sample points from the mesh
        for num_sample in num_samples:
            print("======================= START ========================")
            print(
                f"{object_class_1}, {object_class_2}: instance {instance} with {num_sample} samples"
            )
            pointcloud_o1 = sample_mesh_random(v_1, f_1, num_samples=num_sample)
            pointcloud_o1 = add_gaussian_noise(pointcloud_o1, sigma=0.5)
            pointcloud_o2 = sample_mesh_random(v_2, f_2, num_samples=num_sample)
            pointcloud_o2 = add_gaussian_noise(pointcloud_o2, sigma=0.5)

            # Point Clouds Random Subsampling
            r1_subsample_o1, r2_subsample_o1 = random_subsample_pointcloud(
                pointcloud_o1,
                num_per_sample=int(0.75 * num_sample),
                overlap=overlap,
                center=True,
            )
            r1_subsample_o2, r2_subsample_o2 = random_subsample_pointcloud(
                pointcloud_o2,
                num_per_sample=int(0.75 * num_sample),
                overlap=overlap,
                center=True,
            )

            # Point Clouds Viewpoint Subsampling
            v1_subsample_o1 = viewpoint_subsample_pointcloud(
                pointcloud_o1, sample_1_lower_limit, sample_1_upper_limit, center=True
            )
            v2_subsample_o1 = viewpoint_subsample_pointcloud(
                pointcloud_o1, sample_2_lower_limit, sample_2_upper_limit, center=True
            )
            v1_subsample_o2 = viewpoint_subsample_pointcloud(
                pointcloud_o2, sample_1_lower_limit, sample_1_upper_limit, center=True
            )
            v2_subsample_o2 = viewpoint_subsample_pointcloud(
                pointcloud_o2, sample_2_lower_limit, sample_2_upper_limit, center=True
            )

            # Draw the point cloud
            draw_point_cloud(
                pointcloud_o1,
                title=f"Noisy {object_class_1} and {object_class_2} Point Clouds",
                overlay_pointcloud=pointcloud_o2,
            )
            draw_point_cloud(
                r1_subsample_o1,
                title="Random SubSamping",
                overlay_pointcloud=r2_subsample_o2,
            )
            draw_point_cloud(
                v1_subsample_o1,
                title="Viewpoint Subsampling",
                overlay_pointcloud=v2_subsample_o2,
            )

            # Convert the point cloud to tensor
            pointcloud_o1 = to_tensor(pointcloud_o1, device)
            pointcloud_o2 = to_tensor(pointcloud_o2, device)
            r1_subsample_o1 = to_tensor(r1_subsample_o1, device)
            r1_subsample_o2 = to_tensor(r1_subsample_o2, device)
            v1_subsample_o1 = to_tensor(v1_subsample_o1, device)
            v1_subsample_o2 = to_tensor(v1_subsample_o2, device)
            v2_subsample_o1 = to_tensor(v2_subsample_o1, device)
            v2_subsample_o2 = to_tensor(v2_subsample_o2, device)
            r2_subsample_o1 = to_tensor(r2_subsample_o1, device)
            r2_subsample_o2 = to_tensor(r2_subsample_o2, device)

            with torch.no_grad():
                _, _, _, inv_feat_o1 = model_encoder(pointcloud_o1)
                _, _, _, inv_feat_o2 = model_encoder(pointcloud_o2)
                _, _, _, inv_feat_r1_o1 = model_encoder(r1_subsample_o1)
                _, _, _, inv_feat_r2_o1 = model_encoder(r2_subsample_o1)
                _, _, _, inv_feat_r1_o2 = model_encoder(r1_subsample_o2)
                _, _, _, inv_feat_r2_o2 = model_encoder(r2_subsample_o2)
                _, _, _, inv_feat_v1_o1 = model_encoder(v1_subsample_o1)
                _, _, _, inv_feat_v2_o1 = model_encoder(v2_subsample_o1)
                _, _, _, inv_feat_v1_o2 = model_encoder(v1_subsample_o2)
                _, _, _, inv_feat_v2_o2 = model_encoder(v2_subsample_o2)

                print(
                    "Same-Object Full Point Cloud 1 and Random Subsampled Point Cloud 1",
                    angular_similarity(inv_feat_o1, inv_feat_r1_o1),
                )
                print(
                    "Same-Object Full Point Cloud 1 and Random Subsampled Point Cloud 2",
                    angular_similarity(inv_feat_o1, inv_feat_r2_o1),
                )
                print(
                    "Same-Object Full Point Cloud 1 and Viewpoint Subsampled Point Cloud 1",
                    angular_similarity(inv_feat_o1, inv_feat_v1_o1),
                )
                print(
                    "Same-Object Full Point Cloud 1 and Viewpoint Subsampled Point Cloud 2",
                    angular_similarity(inv_feat_o1, inv_feat_v2_o1),
                )
                print(
                    "Same-Object Full Point Cloud 2 and Random Subsampled Point Cloud 1",
                    angular_similarity(inv_feat_o2, inv_feat_r1_o2),
                )
                print(
                    "Same-Object Full Point Cloud 2 and Random Subsampled Point Cloud 2",
                    angular_similarity(inv_feat_o2, inv_feat_r2_o2),
                )
                print(
                    "Same-Object Full Point Cloud 2 and Viewpoint Subsampled Point Cloud 1",
                    angular_similarity(inv_feat_o2, inv_feat_v1_o2),
                )
                print(
                    "Same-Object Full Point Cloud 2 and Viewpoint Subsampled Point Cloud 2",
                    angular_similarity(inv_feat_o2, inv_feat_v2_o2),
                )
                print(
                    "Inter-Object Full Point Clouds",
                    angular_similarity(inv_feat_o1, inv_feat_o2),
                )
                print(
                    "Inter-Object Random Subsampled Point Clouds 1",
                    angular_similarity(inv_feat_r1_o1, inv_feat_r1_o2),
                )
                print(
                    "Inter-Object Random Subsampled Point Clouds 2",
                    angular_similarity(inv_feat_r2_o1, inv_feat_r2_o2),
                )
                print(
                    "Inter-Object Viewpoint Subsampled Point Clouds 1",
                    angular_similarity(inv_feat_v1_o1, inv_feat_v1_o2),
                )
                print(
                    "Inter-Object Viewpoint Subsampled Point Clouds 2",
                    angular_similarity(inv_feat_v2_o1, inv_feat_v2_o2),
                )
                print(
                    "Inter-Object Viewpoint Subsampled Point Clouds 1-2",
                    angular_similarity(inv_feat_v1_o1, inv_feat_v2_o2),
                )
                print(
                    "Inter-Object Viewpoint Subsampled Point Clouds 2-1",
                    angular_similarity(inv_feat_v2_o1, inv_feat_v1_o2),
                )
            print("======================= END ========================")
