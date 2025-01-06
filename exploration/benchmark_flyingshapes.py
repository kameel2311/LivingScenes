import point_cloud_utils as pcu
import os
import matplotlib.pyplot as plt
import numpy as np
from point_cloud import *
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
import seaborn as sns
from tqdm import tqdm


def angular_similarity(a, b):
    """
    Compute the angular similarity between two tensors
    """
    a = torch.flatten(a)
    b = torch.flatten(b)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def matrix_angular_similarity(a, b):
    """
    Compute the cross similarity Matrix between two tensors
    """
    similarity = a @ b.T
    # Compute norms for each row
    norm_a = torch.norm(a, dim=1, keepdim=True)  # Shape: (N, 1)
    norm_b = torch.norm(b, dim=1, keepdim=True)  # Shape: (M, 1)

    # Outer product of norms
    norm_matrix = norm_a @ norm_b.T  # Shape: (N, M)
    return similarity / norm_matrix


def matrix_fitness_metric(similarity_matrix):
    # Ensure the input is a tensor
    if isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = torch.tensor(similarity_matrix)
    elif not isinstance(similarity_matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

    num_elements = similarity_matrix.shape[0]
    off_diag_means = []
    off_diag_stds = []

    for i in range(num_elements):
        # Extract row and column
        extracted_row = similarity_matrix[i, :]
        extracted_col = similarity_matrix[:, i]

        # Ensure the diagonal element consistency
        diagonal_element = extracted_row[i]
        assert diagonal_element == extracted_col[i]

        # Remove diagonal element
        extracted_row = torch.cat((extracted_row[:i], extracted_row[i + 1 :]))
        extracted_col = torch.cat((extracted_col[:i], extracted_col[i + 1 :]))

        # Combine row and column values and take absolute values
        extracted_values = torch.cat((extracted_row, extracted_col)).abs()

        # Compute mean and std of extracted values
        mean_extracted_values = extracted_values.mean().item()
        std_extracted_values = (extracted_values - diagonal_element).abs().std().item()

        off_diag_means.append(mean_extracted_values)
        off_diag_stds.append(std_extracted_values)

    # Compute final metrics
    off_diag_mean = np.mean(off_diag_means)
    off_diag_std = np.mean(off_diag_stds)
    diag_mean = similarity_matrix.diag().mean().item()

    return diag_mean, off_diag_mean, off_diag_std


def plot_data(
    dataset_diagonal_mean, dataset_off_diagonal_mean, dataset_off_diagonal_std
):
    """
    Plots histograms with density curves for diagonal mean, off-diagonal mean, and std.

    Args:
        dataset_diagonal_mean (list): List of diagonal means.
        dataset_off_diagonal_mean (list): List of off-diagonal means.
        dataset_off_diagonal_std (list): List of off-diagonal standard deviations.
    """
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Histogram with density curve for off-diagonal means
    sns.histplot(
        dataset_off_diagonal_mean,
        bins=10,
        kde=True,
        color="orange",
        label="Off-Diagonal Means",
        alpha=0.6,
        stat="density",
    )

    # Histogram with density curve for off-diagonal std
    sns.histplot(
        dataset_off_diagonal_std,
        bins=10,
        kde=True,
        color="green",
        label="Off-Diagonal Stds",
        alpha=0.6,
        stat="density",
    )

    # Histogram with density curve for diagonal means
    sns.histplot(
        dataset_diagonal_mean,
        bins=10,
        kde=True,
        color="blue",
        label="Diagonal Means",
        alpha=0.6,
        stat="density",
    )

    # Add labels and title
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Histogram and Density of Matrix Fitness Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()


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
