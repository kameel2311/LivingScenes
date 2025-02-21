import sys
import math
import numpy as np
import point_cloud_utils as pcu
import torch
from collections import defaultdict  # Such a cool find btw
from matplotlib import pyplot as plt

sys.path.append("../")
from utils.pointcloud_helper import (
    draw_point_cloud,
    translate_pointcloud_center,
    sample_mesh_random,
    rotate_pointcloud,
    scale_point_cloud,
    center_pointcloud,
    center_pointcloud_v2,
    round_to_1,
)
from utils.rendering_helper import (
    Camera,
    get_circle_poses,
    render_point_cloud_from_viewpoint,
)
from utils.simulation_helper import Object
from utils.metrics_helper import (
    matrix_fitness_metric,
    plot_data,
    plot_dataset,
    plot_rre,
    matrix_angular_similarity,
    plot_correlation,
    compute_pointcloud_overlap,
)


class ObjectTracked(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_tracked_pointcloud = None

    def get_active_tracked_pointcloud(self):
        return self._active_tracked_pointcloud

    def add_active_tracked_pointcloud(self, view_idx):
        if self._active_tracked_pointcloud is None:
            self._active_tracked_pointcloud = self.get_rendered_view(view_idx)
        else:
            new_tracked_pointcloud = [self._active_tracked_pointcloud]
            new_tracked_pointcloud.append(self.get_rendered_view(view_idx))
            self._active_tracked_pointcloud = np.concatenate(
                new_tracked_pointcloud, axis=0
            )

    def visualize_active_tracked_pointcloud(self):
        if self._active_tracked_pointcloud is None:
            raise ValueError("No active tracked pointcloud available")
        draw_point_cloud(self._active_tracked_pointcloud)


class ObjectCollection:
    def __init__(self, objects: list[ObjectTracked], object_collection_config: dict):
        self.objects = objects
        self.config = object_collection_config
        self.lhs_mode = self.config["lhs_mode"]
        self.rhs_mode = self.config["rhs_mode"]
        self.num_views = self.config["number_views"]

        assert (
            self.config["object_scaling_mode"] == "rendering"
        ), "Only Equal Sampling is supported for now"

    def get_view_pointclouds(self, idx):
        lhs_pointclouds = []
        rhs_pointclouds = []
        if idx >= self.num_views and self.lhs_mode == "scan":
            raise ValueError(f"Index out of bounds for LHS in mode: {self.lhs_mode}")

        if idx + 1 == self.num_views and self.rhs_mode == "next_scan":
            raise ValueError(
                f"Index + 1 out of bounds for RHS in mode: {self.rhs_mode}"
            )

        for object in self.objects:
            if self.lhs_mode == "scan":
                lhs_pointclouds.append(object.get_rendered_view(idx))
            elif self.lhs_mode == "tracked":
                object.add_active_tracked_pointcloud(idx)
                lhs_pointclouds.append(object.get_active_tracked_pointcloud())
            else:
                raise ValueError(f"Invalid mode: {self.lhs_mode}")

            if self.rhs_mode == "next_scan":
                rhs_pointclouds.append(object.get_rendered_view(idx + 1))
            elif self.rhs_mode == "mesh":
                rhs_pointclouds.append(object.get_pointcloud())
            else:
                raise ValueError(f"Invalid mode: {self.rhs_mode}")
        return lhs_pointclouds, rhs_pointclouds


class VNBenchmark:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float64)

    def infer_collection(self, object_collection: ObjectCollection):
        collection_similarity_per_view_metrics = defaultdict(list)
        for idx in range(object_collection.num_views):
            lhs_pointclouds, rhs_pointclouds = object_collection.get_view_pointclouds(
                idx
            )
            lhs_pointclouds = (
                torch.tensor(np.array(lhs_pointclouds))
                .to(self.device)
                .float()
                .transpose(-1, -2)
            )
            rhs_pointclouds = (
                torch.tensor(np.array(rhs_pointclouds))
                .to(self.device)
                .float()
                .transpose(-1, -2)
            )
            print(idx)
            print(lhs_pointclouds.shape)
            print(rhs_pointclouds.shape)

            with torch.no_grad():
                lhs_code = self.model.encode(lhs_pointclouds)
                rhs_code = self.model.encode(rhs_pointclouds)

            lhs_code_invariant = lhs_code["z_inv"]
            rhs_code_invariant = rhs_code["z_inv"]
            # lhs_code_se3 = lhs_code["z_so3"] + lhs_code["t"]
            # rhs_code_se3 = rhs_code["z_so3"] + rhs_code["t"]

            # compute the similarity matrix
            score_mat = matrix_angular_similarity(
                lhs_code_invariant, rhs_code_invariant
            )

            diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(
                score_mat, average_along_matrix=False
            )
            collection_similarity_per_view_metrics["diag_mean"].append(diag_mean)
            collection_similarity_per_view_metrics["off_diag_mean"].append(
                off_diag_mean
            )
            collection_similarity_per_view_metrics["off_diag_std"].append(off_diag_std)
        return collection_similarity_per_view_metrics
