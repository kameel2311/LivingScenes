import os
import sys
import yaml
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import manual_seed as torch_manual_seed

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.rendering_helper import Camera
from utils.benchmark_helper import ObjectTracked, ObjectCollection, VNBenchmark

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


def parse_scene_object(config, dataloader, camera, semantic_class, object_idx):
    # Define Objects
    object_settings = config["object_settings"]
    path_to_file = dataloader.get_path(semantic_class, object_idx)
    object = ObjectTracked(
        semantic_class=semantic_class,
        semantic_idx=object_idx,
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

    return object


def load_vn_model():
    # Load the Model
    ckpt = "../../weights"
    solver_cfg = load_yaml("../../configs/more_3rscan.yaml")
    solver_cfg["shape_priors"]["ckpt_dir"] = ckpt
    solver = More_Solver(solver_cfg)
    model = solver.model
    return model


def collection_generator(dataloader, config):
    collection_count = 0

    # Setting the Data Loader
    object_collection_settings = config["object_collection"][
        "object_collection_settings"
    ]
    object_collection_sampler = config["object_collection"]["object_collection_sampler"]

    # Filter out the excluded classes
    allowed_classes = [
        semantic_class
        for semantic_class in dataloader.object_classes
        if semantic_class not in object_collection_sampler["excluded_classes"]
    ]

    # Generate Object Collections
    while collection_count < object_collection_sampler["number_collections"]:
        if object_collection_sampler["allow_same_class"]:
            semantic_classes = random.choices(
                allowed_classes,
                k=object_collection_sampler["number_objects_per_collection"],
            )
        else:
            assert object_collection_sampler["number_objects_per_collection"] <= len(
                allowed_classes
            ), "Number of objects per collection is greater than the number of allowed classes"
            semantic_classes = random.sample(
                allowed_classes,
                k=object_collection_sampler["number_objects_per_collection"],
            )

        semantic_idxs = random.choices(
            list(range(dataloader.object_idx_limit)),
            k=object_collection_sampler["number_objects_per_collection"],
        )
        objects = []
        for semantic_class, object_idx in zip(semantic_classes, semantic_idxs):
            objects.append(
                parse_scene_object(
                    config, dataloader, camera, semantic_class, object_idx
                )
            )
        object_collection = ObjectCollection(objects, object_collection_settings)
        yield object_collection

        # Increment the Collection Count
        collection_count += 1


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
    collection_generator = collection_generator(dataloader, config)

    # Load the Model
    benchmark = VNBenchmark(load_vn_model())

    for object_collection in collection_generator:
        metrics = benchmark.infer_collection(object_collection)
        print(metrics)
