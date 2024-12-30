"""
Testing Code to see whats happening under the hood
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

from point_cloud import draw_point_cloud, path_generator, sample_mesh_random


def set_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level="INFO", logger=logger)
    file_handler = logging.FileHandler(log_path, mode="w")
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


DATA_DIR = "/Datasets/ModelNet10/ModelNet10"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    solver_cfg = load_yaml("/workspace/configs/more_3rscan.yaml")
    # print("Solver Configuration: ", solver_cfg)

    logging.info(f"--------Loading the Solver--------")
    solver = More_Solver(solver_cfg)
    logging.info(f"------------------------------------")

    model_encoder = solver.model.encoder
    print(type(model_encoder))

    # print("Decoder Type: ", type(solver.model.decoder))

    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 100
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)
    v, f = pcu.load_mesh_vf(path_to_file)
    print(f"Vertices: {v.shape}, Faces: {f.shape}")

    # Sample points from the mesh
    v_sampled = sample_mesh_random(v, f, num_samples=512)

    # Draw the point cloud
    draw_point_cloud(v_sampled, title="Chair Point Cloud", overlay_pointcloud=v)

    # Run the model
    device = torch.device("cuda")
    model_encoder = model_encoder.to(device)
    model_encoder.eval()
    v_sampled_cuda = torch.from_numpy(v_sampled)
    v_sampled_cuda = torch.reshape(v_sampled_cuda, (1, 3, 512)).to(device)
    print(v_sampled_cuda.shape)
    print(v_sampled_cuda.type())

    with torch.no_grad():
        centre, scale, so3_feat, inv_feat = model_encoder(v_sampled_cuda)
        centre_2, scale_2, so3_feat_2, inv_feat_2 = model_encoder(v_sampled_cuda)

        print("Difference Centre: ", centre - centre_2)
        print("Difference Scale: ", scale - scale_2)
        print("Difference SO3: ", so3_feat - so3_feat_2)
        print("Difference INV: ", inv_feat - inv_feat_2)
