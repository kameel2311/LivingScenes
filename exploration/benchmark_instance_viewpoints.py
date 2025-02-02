"""
Comparing Object Instances from the same class and different classes under different viewpoints
"""

import sys
from utils.metrics_helper import (
    matrix_fitness_metric,
    plot_data,
    plot_dataset,
    plot_rre,
    matrix_angular_similarity,
    plot_correlation,
    compute_pointcloud_overlap,
)
from utils.dataloader import Dataloader

from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    scale_point_cloud,
    draw_point_cloud,
    add_gaussian_noise,
    rotate_pointcloud_randomly,
    rotate_pointcloud,
)
from utils.rendering_helper import (
    Camera,
    get_circle_poses,
    render_point_cloud_from_viewpoint,
)

sys.path.append("../")

import torch
import random
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
NUM_VIEWPOINTS = 4
VISUALIZE = False
OBJECT_INDEX_LIMIT = 100
NUM_SCENES = 10
NUM_OBJECTS_PER_SCENE = 4

# Sampling Options
MIN_ANGLE = -120
MAX_ANGLE = 120
PC_COUNT = 600
SEQUENTIAL = True


def create_instance_combinations(num_scenes, num_objects_per_scene, object_index_limit):
    return [
        random.sample(range(1, object_index_limit), num_objects_per_scene)
        for _ in range(num_scenes)
    ]


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Loading the Model
    ckpt = "../weights"
    solver_cfg = load_yaml("../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model

    # Benchmark Iterations, Idealworks test doesnt make sense as only one instance is available
    dataloader = Dataloader(
        "ModelNet10", OBJECT_INDEX_LIMIT
    )  # "ModelNet10" or "Idealworks"
    object_classes, object_index_limit, noise_std, alignment_matrix = (
        dataloader.get_metadata()
    )
    # object_classes = ["chair", "table", "monitor", "sofa"]
    # object_classes = [
    #     "bathtub",
    #     "bed",
    #     "chair",
    #     "desk",
    #     "dresser",
    #     "monitor",
    #     "night_stand",
    #     "sofa",
    #     "table",
    #     "toilet",
    # ]

    # Variable Declarations
    dataset_diagonal_mean = {}
    dataset_off_diagonal_mean = {}
    dataset_off_diagonal_std = {}
    rotational_errors = []

    # Point Cloud Distance Metric
    dataset_champfer_distance = []
    dataset_overlap_ratio = []
    dataset_object_labels = []

    # Viewpoint Sampling Camera
    image_height = 500
    image_width = 500
    camera = Camera(
        scale=1, image_height=image_height, image_width=image_width, fx=250, fy=250
    )
    k = camera.get_intrinsics()
    camera_py = camera.get_pyrender_camera()

    # Scene ID Combinations
    scene_combinations = create_instance_combinations(
        NUM_SCENES, NUM_OBJECTS_PER_SCENE, OBJECT_INDEX_LIMIT
    )

    # Iterate over all the object classes
    for object_class in object_classes:
        print("Object Class: ", object_class)
        # Variable Declarations
        class_diagonal_mean = []
        class_off_diagonal_mean = []
        class_off_diagonal_std = []
        # class_rotational_errors = []
        class_champfer_distance = []
        class_overlap_ratio = []

        # Iterate over the scenes
        for scene in scene_combinations:
            print("Scene: ", scene)
            object_meshes = []
            object_rendering_info = []
            for object_idx in scene:
                path_to_file = dataloader.get_path(object_class, object_idx)
                v, f = pcu.load_mesh_vf(path_to_file)
                object_meshes.append((v, f))
                pointcloud = sample_mesh_random(v, f, num_samples=PC_COUNT)
                # pointcloud, _ = rotate_pointcloud(pointcloud, alignment_matrix)
                pointcloud, pointcloud_centered, center, scaling_factor = (
                    scale_point_cloud(
                        pointcloud, inference_method=False, desired_max_dim=10
                    )
                )
                radius = np.max(np.linalg.norm(pointcloud_centered, axis=1)) * 1.5
                world_pose, pyrender_pose = get_circle_poses(
                    NUM_VIEWPOINTS,
                    MIN_ANGLE,
                    MAX_ANGLE,
                    radius,
                    center,
                    sequential=SEQUENTIAL,
                )
                object_rendering_info.append(
                    (world_pose, pyrender_pose, scaling_factor)
                )

            for view_idx in range(NUM_VIEWPOINTS - 1):
                ref_object_pointclouds = []
                rescan_object_pointclouds = []
                gt_rotation = []
                for ref_obj_idx, (v, f) in enumerate(object_meshes):
                    # Render Viewpoints for Reference Object
                    world_poses, pyrender_poses, scaling_factor = object_rendering_info[
                        ref_obj_idx
                    ]
                    v, f = object_meshes[ref_obj_idx]
                    ref_pointcloud = render_point_cloud_from_viewpoint(
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

                    ref_pointcloud = add_gaussian_noise(ref_pointcloud, sigma=0.2)
                    ref_object_pointclouds.append(ref_pointcloud)
                    # draw_point_cloud(pointcloud)

                    # Render Viewpoints for Rescan Object
                    rescan_pointcloud = render_point_cloud_from_viewpoint(
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
                    # Log PointCloud Distance Metric
                    champfer_distance = pcu.chamfer_distance(
                        np.array(ref_pointcloud, order="C"),
                        np.array(rescan_pointcloud, order="C"),
                    )
                    overlap = compute_pointcloud_overlap(
                        ref_pointcloud,
                        rescan_pointcloud,
                        epsilon=min(champfer_distance, 1.0),
                    )
                    dataset_overlap_ratio.append(overlap)
                    dataset_champfer_distance.append(champfer_distance)
                    dataset_object_labels.append(object_classes[ref_obj_idx])

                    # Add Noise and Rotation
                    rescan_pointcloud, rot_matrix = rotate_pointcloud_randomly(
                        rescan_pointcloud, pure_z_rotation=True, identity=True
                    )
                    rescan_pointcloud = add_gaussian_noise(rescan_pointcloud, sigma=0.2)
                    rescan_object_pointclouds.append(rescan_pointcloud)
                    gt_rotation.append(torch.tensor(rot_matrix))

                    # Debugging
                    if VISUALIZE:
                        print("Champfer Distance: ", champfer_distance)
                        print("Overlap Ratio: ", overlap)
                        draw_point_cloud(
                            ref_pointcloud, overlay_pointcloud=rescan_pointcloud
                        )
                gt_rotation = torch.stack(gt_rotation)

                ref_object_pointclouds = (
                    torch.tensor(np.array(ref_object_pointclouds))
                    .cuda()
                    .transpose(-1, -2)
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
                ref_code_se3 = ref_code["z_so3"] + ref_code["t"]
                rescan_code_se3 = rescan_code["z_so3"] + rescan_code["t"]

                # compute the similarity matrix
                score_mat = matrix_angular_similarity(
                    ref_code_invariant, rescan_code_invariant
                )

                diag_mean, off_diag_mean, off_diag_std = matrix_fitness_metric(
                    score_mat, average_along_matrix=False
                )
                class_diagonal_mean.append(diag_mean)
                class_off_diagonal_mean.append(off_diag_mean)
                class_off_diagonal_std.append(off_diag_std.cpu())

                # # Compute the relative transformation matrix
                # R, t, _, _ = kabsch_transformation_estimation(
                #     ref_code_se3, rescan_code_se3
                # )
                # rres = rotation_error(R, gt_rotation.cuda())
                # rres = rres.cpu().numpy()
                # # print(rres)
                # for rre in rres:
                #     rotational_errors.append(rre[0])
        # Log class level data
        dataset_diagonal_mean[object_class] = np.array(class_diagonal_mean).flatten()
        dataset_off_diagonal_mean[object_class] = np.array(
            class_off_diagonal_mean
        ).flatten()
        dataset_off_diagonal_std[object_class] = np.array(
            class_off_diagonal_std
        ).flatten()

        print(
            len(dataset_diagonal_mean[object_class])
        )  # (NUM_VIEWPOINTS-1 * NUM_SCENES * NUM_OBJECTS_PER_SCENE)

    # Plot per class data
    # Plot the data
    for object_class in object_classes:
        print(f"Object Class: {object_class}")
        plot_data(
            dataset_diagonal_mean[object_class],
            dataset_off_diagonal_mean[object_class],
            dataset_off_diagonal_std[object_class],
            class_title=object_class,
        )

    # Plot Whole dataset togther
    plot_dataset(
        dataset_diagonal_mean,
        dataset_off_diagonal_mean,
        dataset_off_diagonal_std,
        same_color=True,
    )
    # plot_rre(rotational_errors, labels=object_classes)

    # plot_correlation(
    #     dataset_champfer_distance,
    #     dataset_diagonal_mean,
    #     "Chamfer",
    #     "Diagonal Mean",
    #     labels=dataset_object_labels,
    # )
    # plot_correlation(
    #     dataset_champfer_distance,
    #     rotational_errors,
    #     "Chamfer",
    #     "Rotation Error",
    #     labels=dataset_object_labels,
    # )
    # plot_correlation(
    #     dataset_diagonal_mean,
    #     rotational_errors,
    #     "Diagonal Values",
    #     "Rotation Error",
    #     labels=dataset_object_labels,
    # )

    # plot_correlation(
    #     dataset_overlap_ratio,
    #     dataset_diagonal_mean,
    #     "Overlap",
    #     "Diagonal Mean",
    #     labels=dataset_object_labels,
    # )
    # plot_correlation(
    #     dataset_overlap_ratio,
    #     rotational_errors,
    #     "Overlap",
    #     "Rotation Error",
    #     labels=dataset_object_labels,
    # )
