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

    # Get all unique objects across all runs
    all_objects = sorted(
        set().union(*[data_dict.keys() for data_dict in data_dicts.values()])
    )

    # Common settings for both plots
    y_pos = np.arange(len(all_objects))
    bar_height = 0.8 / len(data_dicts)  # Divide available space by number of runs

    # Extract data
    for i, (key, data_dict) in enumerate(data_dicts.items()):
        # Create arrays with proper ordering and handle missing values
        mad_errors = []
        coverages = []
        for obj in all_objects:
            if obj in data_dict:
                mad_errors.append(data_dict[obj][0])
                coverages.append(data_dict[obj][1])
            else:
                mad_errors.append(0)  # or np.nan if you prefer gaps
                coverages.append(0)  # or np.nan if you prefer gaps

        # Calculate offset for this run's bars
        offset = bar_height * (i - (len(data_dicts) - 1) / 2)

        # Plot MAD errors
        ax1.barh(
            y_pos + offset,  # Offset each run's bars
            mad_errors,
            height=bar_height,
            color=colors[i],
            alpha=0.7,
            label=key,
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(all_objects)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel("MAD Error")
        ax1.set_title("Per Object MAD Error")
        ax1.grid(True)

        # Plot coverage
        ax2.barh(
            y_pos + offset,  # Offset each run's bars
            coverages,
            height=bar_height,
            color=colors[i],
            alpha=0.7,
            label=key,
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_objects)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel("Coverage")
        ax2.set_title("Per Object Coverage")
        ax2.grid(True)

    # Adjust layout and display
    ax1.legend()
    ax2.legend()
    plt.suptitle("Per Object Metrics")
    plt.tight_layout()
    plt.show()


def main():
    # Load the configuration file
    experiment_config_name = "missed_reobservation.yaml"
    with open(os.path.join("scenarios", experiment_config_name), "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setting the Random Seed
    np.random.seed(config["random_seed"])

    # Define the dataset to work with
    dataloader = Dataloader(config["dataset"], None)
    dataset_metadata = dataloader.get_metadata()

    # Parse the Scene
    camera = parse_scene_camera(config)
    objects = parse_scene_objects(config, dataloader, camera)

    # Metrics as Views are gradually added
    panoptic_scene = Scene(objects)
    panoptic_scene.set_gt_scene(config["reconstruction"]["panoptic"])

    # Add Objects and Map the Changes
    panoptic_scene_metrics, panoptic_object_metrics = (
        panoptic_scene.inflict_scene_changes(config["reconstruction"]["panoptic"])
    )
    panoptic_scene.visualize()

    # VN Scene, clear panoptic scene (NOT to recreate scene for RAM Usage)
    panoptic_scene.clear_scene()
    panoptic_scene.set_gt_scene(config["reconstruction"]["vn_enhanced"])
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

    print(panoptic_object_metrics)
    print(vn_enhanced_object_metrics)


if __name__ == "__main__":

    # TODO: 1) Major edit, make the scaling reflect on the number of
    #       pointclouds rather than uniformly having the objects to same scale -> DONE ?
    #       2) Implement the subsampling of the object pointclouds
    #       3) TSDF Integration and Sampling per object
    main()
