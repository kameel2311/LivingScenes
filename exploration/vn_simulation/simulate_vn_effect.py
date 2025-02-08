"""This script is targeted to simulate the effect of the usage of VN Encoders with Panoptic TSDFs for reconstruction enhancement."""

import os
import sys
import numpy as np
import point_cloud_utils as pcu

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.simulation_helper import Object, Scene
from utils.rendering_helper import Camera

np.random.seed(0)


def plot_gradual_metrics(metrics, x_values=None, x_label="Added View"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    if x_values is None:
        x_values = range(len(metrics))
    ax[0].plot(x_values, [metric[0] for metric in metrics], label="MAD")
    ax[0].set_title("Mean Absolute Distance")
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel("MAD")
    ax[0].grid()
    ax[1].plot(x_values, [metric[1] for metric in metrics], label="Coverage")
    ax[1].set_title("Pointcloud Coverage")
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel("Coverage")
    ax[1].grid()
    plt.show()


# TODO: 1) Major edit, make the scaling reflect on the number of
#       pointclouds rather than uniformly having the objects to same scale
#       2) Implement the subsampling of the object pointclouds
#       3) TSDF Integration and Sampling per object ?

if __name__ == "__main__":
    # Define the dataset to work with
    dataloader = Dataloader("ModelNet10", None)
    dataset_metadata = dataloader.get_metadata()
    scene_objects = ["chair", "table", "sofa"]
    scene_object_idx = [0, 1, 2]
    object_centers = [(40, 35, 0), (50, -20, 0), (-20, 0, 0)]
    object_num_samples = [400] * 3  # [600, 600, 1200]
    num_views = 4

    # Scene's Camera
    camera = Camera(scale=1, image_height=500, image_width=500, fx=250, fy=250)

    # Define Objects
    objects = []
    for object_class, object_idx in zip(scene_objects, scene_object_idx):
        path_to_file = dataloader.get_path(object_class, object_idx)
        v, f = pcu.load_mesh_vf(path_to_file)
        objects.append(
            Object(
                path=path_to_file,
                num_points=object_num_samples[object_idx],
                num_rend_points=int(object_num_samples[object_idx] / num_views),
                num_views=num_views,
                min_angle=0,
                max_angle=360,
                camera=camera,
                translation=object_centers[object_idx],
                fix_scaling=False,
            )
        )

    # # Metrics as Views are gradually added
    # gradual_scene = Scene(objects)
    # gradual_metrics = []
    # for view_idx in range(num_views):
    #     for obj_idx in range(len(objects)):
    #         gradual_scene.add_to_scene(obj_idx, view_idx)
    #         gradual_metrics.append(
    #             (gradual_scene.get_MAD(), gradual_scene.get_scene_coverage(epsilon=1.0))
    #         )
    #         gradual_scene.visualize()

    # plot_gradual_metrics(gradual_metrics)
    # # gradual_scene.visualize()

    # # Metrics as Views are gradually added
    # rotations = np.linspace(0, 180, 19)
    # gradual_metrics = []
    # for rotation in rotations:
    #     gradual_scene = Scene(objects)
    #     for view_idx in range(num_views):
    #         for obj_idx in range(len(objects)):
    #             gradual_scene.add_to_scene(obj_idx, view_idx, yaw_angle=rotation)
    #     gradual_metrics.append(
    #         (
    #             gradual_scene.get_MAD(),
    #             gradual_scene.get_scene_coverage(epsilon=1.0),
    #         )
    #     )

    # plot_gradual_metrics(gradual_metrics, x_label="Iterations", x_values=rotations)
    # # gradual_scene.visualize()

    # Scene as if more data retained
    scene = Scene(objects)
    for view_idx in range(num_views):
        for obj_idx in range(1, len(objects)):
            scene.add_to_scene(obj_idx, view_idx)
    scene.visualize()
    adding_object_metrics = []
    adding_object_metrics.append(
        (scene.get_MAD(), scene.get_scene_coverage(epsilon=1.0))
    )
    for i in range(4):
        yaw_angle = 0
        if i >= 1:
            yaw_angle = 20
        scene.add_to_scene(0, i, yaw_angle=yaw_angle)
        adding_object_metrics.append(
            (scene.get_MAD(), scene.get_scene_coverage(epsilon=0.5))
        )
        scene.visualize()
    plot_gradual_metrics(adding_object_metrics, x_label="Added Object")
