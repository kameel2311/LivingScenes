import os
import sys
import yaml
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

    # Setting the Random Seed
    np.random.seed(config["random_seed"])
    torch_manual_seed(config["random_seed"])

    # Setting the Data Loader
    dataloader = Dataloader(
        config["dataset"]["name"], config["dataset"]["object_idx_limit"]
    )

    # Parse the Scene
    camera = parse_scene_camera(config)
    collection_generator = CollectionGenerator(dataloader, config, camera)
    semantic_classes = collection_generator.get_allowed_classes()

    # Load the Model
    benchmark = VNBenchmark(load_vn_model())

    for object_collection in collection_generator.generate_collections():
        similarity_metrics, pose_errors = benchmark.infer_collection(object_collection)
        benchmark.collect_metrics(similarity_metrics, pose_errors, semantic_classes)

    benchmark.plot_metrics()
