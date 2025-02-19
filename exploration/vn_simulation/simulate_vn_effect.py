"""This script is targeted to simulate the effect of the usage of VN Encoders with Panoptic TSDFs for reconstruction enhancement.
    It tries to emulate having already seen objects that get moved to regions of partial visibility and the effect of the VN Encoder on the reconstruction quality. """

import os
import sys
import yaml
import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.simulation_helper import Object, Scene
from utils.rendering_helper import Camera

np.random.seed(0)


def parse_scene_camera(config):
    return Camera(
        scale=config["camera"]["scale"],
        image_height=config["camera"]["image_height"],
        image_width=config["camera"]["image_width"],
        fx=config["camera"]["fx"],
        fy=config["camera"]["fy"],
    )


def parse_scene_objects(config, dataloader, camera):
    # Define Objects
    objects = []
    object_settings = config["object_settings"]
    for object_metadata in config["objects"]:
        path_to_file = dataloader.get_path(
            object_metadata["class"], object_metadata["object_idx"]
        )
        objects.append(
            Object(
                semantic_class=object_metadata["class"],
                semantic_idx=object_metadata["object_idx"],
                path=path_to_file,
                num_points=object_settings["number_points"],
                num_rend_points=object_settings["number_rendered_points"],
                num_views=object_settings["number_views"],
                min_angle=object_settings["min_angle"],
                max_angle=object_settings["max_angle"],
                camera=camera,
                center=None,
                scaling_Mode=object_settings["scaling_mode"],
                adapt_num_points=object_settings["adapt_number_points"],
                verbose=object_settings["verbose"],
            )
        )
    return objects


def plot_gradual_metrics(**metrics_kargs):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    colors = [
        hsv_to_rgb((i / len(metrics_kargs), 0.8, 0.8))
        for i in range(len(metrics_kargs))
    ]
    for i, (key, metrics) in enumerate(metrics_kargs.items()):
        x_values = np.arange(len(metrics))
        ax[0].plot(
            x_values, [metric[0] for metric in metrics], color=colors[i], label=key
        )
        ax[0].set_title("Mean Absolute Distance")
        ax[0].set_xlabel("Added View")
        ax[0].set_ylabel("MAD")
        ax[1].plot(
            x_values, [metric[1] for metric in metrics], color=colors[i], label=key
        )
        ax[1].set_title("Pointcloud Coverage")
        ax[1].set_xlabel("Added View")
        ax[1].set_ylabel("Coverage")

    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    plt.suptitle("Gradual Scene Metrics")
    plt.show()


def plot_object_metrics(**data_dicts):

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = [
        hsv_to_rgb((i / len(data_dicts), 0.8, 0.8)) for i in range(len(data_dicts))
    ]

    # Extract data
    for i, (key, data_dict) in enumerate(data_dicts.items()):
        objects = list(data_dict.keys())
        mad_errors = [data[0] for data in data_dict.values()]
        coverages = [data[1] for data in data_dict.values()]

        # Common settings for both plots
        y_pos = np.arange(len(objects))

        # Plot MAD errors
        ax1.barh(
            y_pos, mad_errors, align="center", color=colors[i], alpha=0.7, label=key
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(objects)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel("MAD Error")
        ax1.set_title("Per Object MAD Error")
        ax1.grid(True, axis="x")

        # Plot coverage
        ax2.barh(
            y_pos, coverages, align="center", color=colors[i], alpha=0.7, label=key
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(objects)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel("Coverage")
        ax2.set_title("Per Object Coverage")
        ax2.grid(True, axis="x")

    # Adjust layout and display
    ax1.legend()
    ax2.legend()
    plt.suptitle("Per Object Metrics")
    plt.tight_layout()
    plt.show()


# TODO: 1) Major edit, make the scaling reflect on the number of
#       pointclouds rather than uniformly having the objects to same scale -> DONE ?
#       2) Implement the subsampling of the object pointclouds
#       3) TSDF Integration and Sampling per object ?

if __name__ == "__main__":
    # Load the configuration file
    experiment_config_name = "partial_visibility.yaml"
    with open(os.path.join("scenarios", experiment_config_name), "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Define the dataset to work with
    dataloader = Dataloader(config["dataset"], None)
    dataset_metadata = dataloader.get_metadata()

    # Parse the Scene
    camera = parse_scene_camera(config)
    objects = parse_scene_objects(config, dataloader, camera)

    # Metrics as Views are gradually added
    panoptic_scene = Scene(objects)

    # Add Objects and Map the Changes
    panoptic_scene_metrics, panoptic_object_metrics = (
        panoptic_scene.inflict_scene_changes(config["reconstruction"]["panoptic"])
    )
    panoptic_scene.visualize()

    # VN Scene, clear panoptic scene (NOT to recreate scene for RAM Usage)
    panoptic_scene.clear_simulated_scene()
    vn_enhanced_scene_metrics, vn_enhanced_object_metrics = (
        panoptic_scene.inflict_scene_changes(config["reconstruction"]["vn_enhanced"])
    )
    panoptic_scene.visualize()

    # Plot Metrics
    plot_gradual_metrics(
        panoptic=panoptic_scene_metrics, vn_enchanced=vn_enhanced_scene_metrics
    )

    plot_object_metrics(
        panoptic=panoptic_object_metrics, vn_enchanced=vn_enhanced_object_metrics
    )
