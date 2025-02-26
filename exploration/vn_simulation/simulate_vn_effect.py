"""This script is targeted to simulate the effect of the usage of VN Encoders with Panoptic TSDFs for reconstruction enhancement.
    It tries to emulate having already seen objects that get moved to regions of partial visibility and the effect of the VN Encoder on the reconstruction quality. """

import os
import sys
import yaml
import argparse
import random
import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.simulation_helper import Object, Scene, BenchmarkRunner
from utils.rendering_helper import Camera
from utils.plotting_utils import plot_gradual_metrics, plot_object_metrics


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
    object_loader = config["objects_loader"]
    classes, idx_limit, _, _ = dataloader.get_metadata()

    if object_loader["limit_to_classes"] is not None:
        classes = object_loader["limit_to_classes"]

    # Randomly Sample Objects
    semantic_classes = random.choices(classes, k=object_loader["number_objects"])
    semantic_idxs = random.choices(range(idx_limit), k=object_loader["number_objects"])
    print(semantic_classes)
    for semantic_class, semantic_idx in zip(semantic_classes, semantic_idxs):
        path_to_file = dataloader.get_path(semantic_class, semantic_idx)
        objects.append(
            Object(
                semantic_class=semantic_class,
                semantic_idx=semantic_idx,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VN Simulation on defined Scenario"
    )
    parser.add_argument("--scenario", help="Scneario Config File", required=True)
    return parser.parse_args()


def main(args):
    # Load the configuration file
    # experiment_config_name = "missed_reobservation.yaml"
    with open(os.path.join("scenarios", args.scenario), "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setting the Random Seed
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # Define the dataset to work with
    dataloader = Dataloader(config["dataset"]["name"], config["dataset"]["idx_limit"])

    # Parse the Scene
    camera = parse_scene_camera(config)
    objects = parse_scene_objects(config, dataloader, camera)

    # Benchmark Runner
    runner = BenchmarkRunner(objects, config["reconstruction"])
    runner.run_benchmark()
    runner.plot_metrics()


if __name__ == "__main__":

    # TODO: 1) Major edit, make the scaling reflect on the number of
    #       pointclouds rather than uniformly having the objects to same scale -> DONE ?
    #       2) Implement the subsampling of the object pointclouds
    #       3) TSDF Integration and Sampling per object

    args = parse_args()
    assert args.scenario != "missed_reobservation.yaml", "Scenario Deprecated for now"
    main(args)
