"""
Comparing Object Instances from the same instance
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
    object_class = "chair"
    file = "train"
    num_samples = [600, 400, 200]
    instances = [100, 200, 300]
    overlap = 0.5
    sample_1_lower_limit = 0
    sample_1_upper_limit = 179
    sample_2_lower_limit = 180
    sample_2_upper_limit = 359

    for instance in instances:
        # Load Object Instance
        path_to_file = path_generator(DATA_DIR, object_class, file, instance)
        v, f = pcu.load_mesh_vf(path_to_file)
        print(f"Vertices: {v.shape}, Faces: {f.shape}")

        # Sample points from the mesh
        for num_sample in num_samples:
            print("======================= START ========================")
            print(f"{object_class}: instance {instance} with {num_sample} samples")
            v_sampled = sample_mesh_random(v, f, num_samples=num_sample)
            v_sampled = add_gaussian_noise(v_sampled, sigma=0.5)
            # v_noisy_2 = add_gaussian_noise(v_sampled, sigma=0.3)

            # Point Clouds with Overlapping Points - Random Subsampling
            r_subsample_1, r_subsample_2 = random_subsample_pointcloud(
                v_sampled, num_per_sample=int(num_sample * 0.75), overlap=overlap
            )

            # Point Clouds with Overlapping Points - Viewpoint Subsampling
            v_subsample_1 = viewpoint_subsample_pointcloud(
                v_sampled, sample_1_lower_limit, sample_1_upper_limit
            )
            v_subsample_2 = viewpoint_subsample_pointcloud(
                v_sampled, sample_2_lower_limit, sample_2_upper_limit
            )

            # Draw the point cloud
            draw_point_cloud(v_sampled, title=f"Noisy {object_class} Point Cloud")
            draw_point_cloud(
                r_subsample_1,
                title="Random SubSamping",
                overlay_pointcloud=r_subsample_2,
            )
            draw_point_cloud(
                v_subsample_1,
                title="Viewpoint Subsampling",
                overlay_pointcloud=v_subsample_2,
            )

            # Convert the point cloud to tensor
            v_sampled = to_tensor(v_sampled, device)
            r_subsample_1 = to_tensor(r_subsample_1, device)
            r_subsample_2 = to_tensor(r_subsample_2, device)
            v_subsample_1 = to_tensor(v_subsample_1, device)
            v_subsample_2 = to_tensor(v_subsample_2, device)

            with torch.no_grad():
                centre, scale, so3_feat, inv_feat = model_encoder(v_sampled)
                centre_r_1, scale_r_1, so3_feat_r_1, inv_feat_r_1 = model_encoder(
                    r_subsample_1
                )
                centre_r_2, scale_r_2, so3_feat_r_2, inv_feat_r_2 = model_encoder(
                    r_subsample_2
                )
                centre_v_1, scale_v_1, so3_feat_v_1, inv_feat_v_1 = model_encoder(
                    v_subsample_1
                )
                centre_v_2, scale_v_2, so3_feat_v_2, inv_feat_v_2 = model_encoder(
                    v_subsample_2
                )

                print(
                    "Noisy Point Cloud with Random Subsampled Point Cloud 1",
                    angular_similarity(inv_feat, inv_feat_r_1),
                )
                print(
                    "Noisy Point Cloud with Random Subsampled Point Cloud 2",
                    angular_similarity(inv_feat, inv_feat_r_2),
                )
                print(
                    f"Random Samples Overlapping {overlap}",
                    angular_similarity(inv_feat_r_1, inv_feat_r_2),
                )

                print(
                    "Noisy Point Cloud with Viewpoint Subsampled Point Cloud 1",
                    angular_similarity(inv_feat, inv_feat_v_1),
                )
                print(
                    "Noisy Point Cloud with Viewpoint Subsampled Point Cloud 2",
                    angular_similarity(inv_feat, inv_feat_v_2),
                )
                print(
                    "Viewpoint Subsampled Point Clouds",
                    angular_similarity(inv_feat_v_1, inv_feat_v_2),
                )
            print("======================= END ========================")
