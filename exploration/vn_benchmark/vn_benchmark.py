import os
import sys
import yaml
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import manual_seed as torch_manual_seed

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.rendering_helper import Camera
from utils.benchmark_helper import VNBenchmark, CollectionGenerator

sys.path.append("../../")
from lib_more.more_solver import More_Solver
from lib_more.pose_estimation import kabsch_transformation_estimation, rotation_error
from lib_more.utils import load_yaml


def parse_scene_camera(config):
    return Camera(
        scale=config["camera"]["scale"],
        image_height=config["camera"]["image_height"],
        image_width=config["camera"]["image_width"],
        fx=config["camera"]["fx"],
        fy=config["camera"]["fy"],
    )


def load_vn_model():
    # Load the Model
    ckpt = "../../weights"
    solver_cfg = load_yaml("../../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model
    return model


if __name__ == "__main__":
    # Load the Config File
    CONFIG_FILE_NAME = "config.yaml"
    with open(os.path.join("configs", CONFIG_FILE_NAME), "r") as file:
        config = yaml.safe_load(file)

    # Saving Directory
    if config["save_plots"]:
        if not os.path.exists(config["save_dir"]):
            os.makedirs(config["save_dir"])
            os.makedirs(os.path.join(config["save_dir"], "per_class"))
        else:
            raise ValueError("Directory already exists")

    # Setting the Random Seed
    np.random.seed(config["random_seed"])
    torch_manual_seed(config["random_seed"])
    random.seed(config["random_seed"])

    # Setting the Data Loader
    dataloader = Dataloader(
        config["dataset"]["name"], config["dataset"]["object_idx_limit"]
    )

    # Parse the Scene
    camera = parse_scene_camera(config)
    collection_generator = CollectionGenerator(dataloader, config, camera)
    semantic_classes = collection_generator.get_allowed_classes()

    # Load the Model
    benchmark = VNBenchmark(
        model=load_vn_model(),
        pose_estimation=config["pose_estimation"],
        noise_std=config["noise_std"],
    )

    for i, object_collection in enumerate(collection_generator.generate_collections()):
        print(f"Collection {i+1}/{config['object_collection']['number_collections']}")
        similarity_metrics, pose_errors, pointcloud_metrics = (
            benchmark.infer_collection(
                object_collection, epsilon=config["epsilon"], fps=config["fps"]
            )
        )

        # Should work for now since no duplicates and no random class selection
        assert (
            semantic_classes == object_collection.get_collection_classes()
        ), "Mismatch in the classes"
        benchmark.collect_metrics(
            similarity_metrics, pose_errors, pointcloud_metrics, semantic_classes
        )

    # Plotting and Saving the Results
    benchmark.plot_metrics(
        config["plot_per_class"],
        config["save_plots"],
        config["save_dir"],
        cls_subfolder="per_class",
    )
