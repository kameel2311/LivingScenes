"""This script is targeted to simulate the effect of the usage of VN Encoders with Panoptic TSDFs for reconstruction enhancement.
    It tries to emulate having already seen objects that get moved to regions of partial visibility and the effect of the VN Encoder on the reconstruction quality. """

import os
import sys
import yaml
import numpy as np
import point_cloud_utils as pcu

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
    # Load the configuration file
    experiment_config_name = "partial_visibility.yaml"
    with open(os.path.join("scenarios", experiment_config_name), "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Define the dataset to work with
    dataloader = Dataloader(config["dataset"], None)
    dataset_metadata = dataloader.get_metadata()

    # Scene's Camera
    camera = parse_scene_camera(config)

    # Scene's Objects
    objects = parse_scene_objects(config, dataloader, camera)

    # Metrics as Views are gradually added
    gradual_scene = Scene(objects)
    gradual_metrics = []
    # for view_idx in range(num_views):
    #     for obj_idx in range(len(objects)):
    #         gradual_scene.add_to_scene(obj_idx, view_idx)
    #         gradual_metrics.append(
    #             (gradual_scene.get_MAD(), gradual_scene.get_scene_coverage(epsilon=1.0))
    #         )
    #         gradual_scene.visualize()

    # plot_gradual_metrics(gradual_metrics)
    gradual_scene.visualize()
