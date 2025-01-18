import point_cloud_utils as pcu
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.pointcloud_helper import *
import os.path as osp
import torch
import glob
import sys

sys.path.append("../")
from lib_more.utils import (
    read_list_from_txt,
    load_json,
    load_yaml,
    visualize_shape_matching,
)
from lib_more.more_solver import More_Solver
from tqdm import tqdm
from utils.metrics_helper import (
    matrix_angular_similarity,
    matrix_fitness_metric,
    plot_data,
)


class FlyingShape(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        # load data
        # [int(shape_n.split("_")[-1]) for shape_n in sorted(os.listdir(path))]
        self.n_shape_lst = sorted(os.listdir(path))
        self.scene_lst = []
        for n_shape in self.n_shape_lst:
            n = int(n_shape.split("_")[-1])
            scene_lst = sorted(os.listdir(osp.join(path, n_shape)))
            self.scene_lst += [
                osp.join(path, n_shape, scene_i) for scene_i in scene_lst
            ]

    def _load_scene(self, scene_path):
        # get scene dict
        scene_list = sorted(glob.glob(osp.join(scene_path, "*.npz")))
        scene_list = [np.load(scene_path) for scene_path in scene_list]
        return scene_list

    def __len__(self):
        return len(self.scene_lst)

    def __getitem__(self, idx):
        scene_path = self.scene_lst[idx]
        return self._load_scene(scene_path)


if __name__ == "__main__":
    DATA_PATH = "/Datasets/FlyingShapes"
    dataset = FlyingShape(DATA_PATH)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Variable Declarations
    dataset_diagonal_mean = []
    dataset_off_diagonal_mean = []
    dataset_off_diagonal_std = []

    # Loop through the dataset
    for data in tqdm(dataset):
        ref_pc = torch.from_numpy(data[0]["pc"]).cuda().float().transpose(-1, -2)
        ref_pc_obj_id = data[0]["obj_id"]
        rescan_lst = [
            torch.from_numpy(scene["pc"]).cuda().float().transpose(-1, -2)
            for scene in data[1:]
        ]
        rescan_obj_id_lst = [scene["obj_id"] for scene in data[1:]]
        with torch.no_grad():
            ref_code = model.encode(ref_pc)
            rescan_code_lst = [model.encode(rescan_pc) for rescan_pc in rescan_lst]

        ref_code_invariant = ref_code["z_inv"]
        # print("Reference IDs: ", ref_pc_obj_id)
        # print(ref_code_invariant.shape)

        for rescan_code in rescan_code_lst:
            # print("Rescan IDs: ", rescan_obj_id_lst[0])
            assert (
                ref_pc_obj_id == rescan_obj_id_lst[0]
            ).all(), "Object ID/Order mismatch"
            rescan_code_invariant = rescan_code["z_inv"]
            # compute the similarity matrix
            score_mat = matrix_angular_similarity(
                ref_code_invariant, rescan_code_invariant
            )

            diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(score_mat)
            dataset_diagonal_mean.append(diag_mean)
            dataset_off_diagonal_mean.append(off_diag_mean)
            dataset_off_diagonal_std.append(off_diag_std)

    plot_data(
        dataset_diagonal_mean, dataset_off_diagonal_mean, dataset_off_diagonal_std
    )
