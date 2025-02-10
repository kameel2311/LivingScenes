"""Genetares simulated panoptic data for single object."""

import cv2
import sys
import os
import numpy as np
import point_cloud_utils as pcu

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.simulation_helper import Object, Scene
from utils.rendering_helper import Camera, transformation_matrix
from PIL import Image as PILImage
import time

np.random.seed(0)

# Note: only works for single object as no real camera pose in scene used


def depth_to_segmentation(depth_image, threshold=0.0):
    """Converts depth image to segmentation mask."""
    segmentation = np.zeros_like(depth_image)
    segmentation[depth_image > threshold] = 1
    return segmentation


def depth_to_rgd(depth_image, threshold=0.0):
    """Converts depth image to rgb image."""
    rgb_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    rgb_image = np.array(rgb_image, dtype=np.uint8)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    # rgb_image = cv2.applyColorMap(rgb_image, cv2.COLORMAP_JET)

    return rgb_image


def save_timestamps_as_csv(image_to_timestamps, run_dir, file_name="timestamps.csv"):
    with open(os.path.join(run_dir, file_name), "w") as f:
        f.write("ImageID,TimeStamp\n")
        for image_id, timestamp in image_to_timestamps.items():
            f.write(f"{image_id}, {timestamp}\n")


def depth_inflict_pattern(depth_image, radius, in_value, out_value=0):
    center = (depth_image.shape[0] // 2, depth_image.shape[1] // 2)
    xes = np.hstack(
        [np.array(range(depth_image.shape[0]))] * depth_image.shape[1]
    ).reshape(depth_image.shape)
    yes = (
        np.vstack([np.array(range(depth_image.shape[1]))] * depth_image.shape[0])
        .reshape(depth_image.shape)
        .T
    )
    dists = np.sqrt((xes - center[0]) ** 2 + (yes - center[1]) ** 2)
    dists[dists < radius] = in_value
    dists[dists >= radius] = out_value
    return dists.reshape(depth_image.shape)


output_dir = "/Datasets/simulated_data/"
run_name = "run1"
run_dir = os.path.join(output_dir, run_name)

FIXED_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
FIXED_TRANS = transformation_matrix(FIXED_ROT, np.array([0, 0, 0]))

if __name__ == "__main__":
    # Check if the output directory exists
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    else:
        pass
        # raise ValueError("Output directory already exists")

    # The camera poses
    # camera_poses = object.get_pyrender_poses()
    number_frames = 100
    img_no_and_ts = {}
    start_time = time.time()
    first_pose = None

    # Loop and Save the data
    for i in range(number_frames):
        # Simulate the object
        depth_image = np.ones((500, 500)) * 5

        depth_image = depth_inflict_pattern(depth_image, 50, 5, 0)

        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rgb_image = depth_to_rgd(depth_image)
        segmentation_image = depth_to_segmentation(depth_image)

        # Image Name
        image_id = "%06d" % i
        img_no_and_ts[image_id] = start_time + i * 1

        # Save depth image
        depth_image = PILImage.fromarray(depth_image)
        depth_image.save(
            os.path.join(
                run_dir,
                image_id + "_depth.tiff",
            )
        )

        # Save colour image
        cv2.imwrite(
            os.path.join(
                run_dir,
                image_id + "_color.png",
            ),
            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
        )

        # Save segmentation image
        cv2.imwrite(
            os.path.join(
                run_dir,
                image_id + "_segmentation.png",
            ),
            segmentation_image,
        )

        # Save camera pose
        with open(os.path.join(run_dir, image_id + "_pose.txt"), "w") as f:
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")

        # Save the timestamps
        save_timestamps_as_csv(img_no_and_ts, run_dir, file_name="timestamps.csv")
