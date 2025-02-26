import sys
import math
import random
import numpy as np
import point_cloud_utils as pcu
import torch
from collections import defaultdict  # Such a cool find btw
from matplotlib import pyplot as plt
from itertools import chain

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
    plot_views_subplots,
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

        if self.lhs_mode != "mesh":
            if self.rhs_mode == "mesh":
                self.num_views = self.config["number_views"]
            elif self.rhs_mode == "next_scan":
                self.num_views = self.config["number_views"] - 1
            else:
                raise ValueError(f"Invalid mode: RHS: {self.rhs_mode}")
        elif self.lhs_mode == "mesh" and self.rhs_mode == "mesh":
            self.num_views = 1
        else:
            raise ValueError(
                f"Invalid mode combination: LHS: {self.lhs_mode}, RHS: {self.rhs_mode}"
            )

        assert (
            self.config["object_scaling_mode"] == "rendering"
        ), "Only Equal Sampling is supported for now"

    def get_view_pointclouds(self, idx):
        lhs_pointclouds = []
        rhs_pointclouds = []
        if idx == self.num_views:
            raise ValueError("Index out of bounds")

        for object in self.objects:
            if self.lhs_mode == "scan":
                lhs_pointclouds.append(object.get_rendered_view(idx))
            elif self.lhs_mode == "tracked":
                object.add_active_tracked_pointcloud(idx)
                lhs_pointclouds.append(object.get_active_tracked_pointcloud())
            elif self.lhs_mode == "mesh":  # IMPLEMENT THIS for 1 round
                lhs_pointclouds.append(object.get_pointcloud())
            else:
                raise ValueError(f"Invalid mode: {self.lhs_mode}")

            if self.rhs_mode == "next_scan":
                rhs_pointclouds.append(object.get_rendered_view(idx + 1))
            elif self.rhs_mode == "mesh":
                rhs_pointclouds.append(object.get_pointcloud())
            else:
                raise ValueError(f"Invalid mode: {self.rhs_mode}")
        return lhs_pointclouds, rhs_pointclouds


# TODO: Add intraclass
class CollectionGenerator:
    def __init__(self, dataloader, config, camera):
        self.dataloader = dataloader
        self.config = config
        self.camera = camera
        self.prepare_collections()

    def parse_scene_object(self, semantic_class, object_idx):
        # Define Objects
        object_settings = self.config["object_settings"]
        path_to_file = self.dataloader.get_path(semantic_class, object_idx)
        object = ObjectTracked(
            semantic_class=semantic_class,
            semantic_idx=object_idx,
            path=path_to_file,
            num_points=object_settings["number_points"],
            num_rend_points=object_settings["number_rendered_points"],
            num_views=object_settings["number_views"],
            min_angle=object_settings["min_angle"],
            max_angle=object_settings["max_angle"],
            camera=self.camera,
            center=None,
            scaling_Mode=object_settings["scaling_mode"],
            adapt_num_points=object_settings["adapt_number_points"],
            verbose=object_settings["verbose"],
        )

        return object

    def prepare_collections(self):
        self.number_collections = self.config["object_collection"]["number_collections"]
        excluded_classes = self.config["object_collection"]["excluded_classes"]

        # Setting the Data Loader
        self.object_collection_settings = self.config["object_collection"][
            "object_collection_settings"
        ]
        self.object_collection_sampler = self.config["object_collection"][
            "object_collection_sampler"
        ]

        # Filter out the excluded classes
        if excluded_classes is None:
            self._allowed_classes = self.dataloader.object_classes
        else:
            self._allowed_classes = [
                semantic_class
                for semantic_class in self.dataloader.object_classes
                if semantic_class not in excluded_classes
            ]

    def get_allowed_classes(self):
        return self._allowed_classes

    def generate_collections(self):
        # Benchmark Type
        if self.object_collection_sampler["type"] == "interclass":
            semantic_classes = self._allowed_classes.copy()
            collection_count = 0

            # Generate Object Collections
            while collection_count < self.number_collections:
                if self.object_collection_sampler["index_selection"] == "sequential":
                    semantic_idxs = np.ones(len(semantic_classes)) * collection_count
                elif self.object_collection_sampler["index_selection"] == "random":
                    semantic_idxs = random.choices(
                        list(range(self.dataloader.object_idx_limit)),
                        k=len(semantic_classes),
                    )
                else:
                    raise ValueError("Invalid index selection mode")
                objects = []
                for semantic_class, object_idx in zip(semantic_classes, semantic_idxs):
                    objects.append(self.parse_scene_object(semantic_class, object_idx))
                object_collection = ObjectCollection(
                    objects, self.object_collection_settings
                )
                yield object_collection

                # Increment the Collection Count
                collection_count += 1
        elif self.object_collection_sampler["type"] == "intraclass":
            raise NotImplementedError("Intraclass Sampler is not implemented yet")
        else:
            raise ValueError("Invalid Object Collection Sampler Type")


class VNBenchmark:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float64)

        self.dataset_similarity_per_view_metrics = defaultdict(list)

    def infer_collection(self, object_collection: ObjectCollection):
        collection_similarity_per_view_metrics = defaultdict(list)
        self.num_views = object_collection.num_views
        for idx in range(self.num_views):
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

    def collect_metrics(
        self, similarity_metric, classes
    ):  # TODO: Adapt for Intraclass too
        # NOW ONLY FOR INTERCLASS
        diagonal_means = similarity_metric["diag_mean"]
        off_diagonal_means = similarity_metric["off_diag_mean"]
        off_diagonal_stds = similarity_metric["off_diag_std"]

        # Convert GPU Tensor to CPU List
        diagonal_means = torch.stack(diagonal_means).cpu().numpy().tolist()
        assert (
            len(diagonal_means)
            == len(off_diagonal_means)
            == len(off_diagonal_stds)
            == self.num_views
        )

        for i in range(self.num_views):
            self.dataset_similarity_per_view_metrics[f"view_{i}_diag_mean"].append(
                diagonal_means[i]
            )

            self.dataset_similarity_per_view_metrics[f"view_{i}_off_diag_mean"].append(
                off_diagonal_means[i]
            )

            self.dataset_similarity_per_view_metrics[f"view_{i}_off_diag_std"].append(
                off_diagonal_stds[i]
            )

    def plot_metrics(self):
        plot_views_subplots(self.dataset_similarity_per_view_metrics, self.num_views)
