"""Genetares simulated panoptic data for single object."""

import cv2
import sys
import os
import numpy as np
import point_cloud_utils as pcu

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.simulation_helper import Object, Scene
from utils.rendering_helper import Camera
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
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    rgb_image = np.array(rgb_image, dtype=np.uint8)
    # rgb_image = cv2.applyColorMap(rgb_image, cv2.COLORMAP_JET)

    return rgb_image


def save_timestamps_as_csv(image_to_timestamps, run_dir, file_name="timestamps.csv"):
    with open(os.path.join(run_dir, file_name), "w") as f:
        f.write("ImageID,TimeStamp\n")
        for image_id, timestamp in image_to_timestamps.items():
            f.write(f"{image_id}, {timestamp}\n")


def plot_image(depth_image, segementation_image, rgb, title="Image"):
    """Plots the image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].imshow(depth_image)
    ax[0].set_title("Depth Image")
    ax[1].imshow(segementation_image)
    ax[1].set_title("Segmentation Image")
    ax[2].imshow(rgb)
    ax[2].set_title("RGB Image")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


output_dir = "/Datasets/simulated_data/"
run_name = "run1"
run_dir = os.path.join(output_dir, run_name)

if __name__ == "__main__":
    # Check if the output directory exists
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    else:
        pass
        # raise ValueError("Output directory already exists")

    # Define the dataset to work with
    dataloader = Dataloader("ModelNet10", None)
    dataset_metadata = dataloader.get_metadata()
    object_class = "chair"
    object_num_samples = 600
    num_views = 36
    object_idx = 0
    object_center = (0, 0, 0)

    # Scene's Camera
    camera = Camera(scale=1, image_height=500, image_width=500, fx=250, fy=250)

    path_to_file = dataloader.get_path(object_class, object_idx)
    v, f = pcu.load_mesh_vf(path_to_file)
    object = Object(
        path=path_to_file,
        num_points=object_num_samples,
        num_rend_points=int(object_num_samples / num_views),
        num_views=num_views,
        min_angle=0,
        max_angle=360,
        camera=camera,
        translation=object_center,
        fix_scaling=False,
        save_depth=True,
    )

    # object.visualize(depth=True)

    # The camera poses
    camera_poses = object.get_pyrender_poses()
    depth_images = object.get_depth_images()
    segmentation_images = [
        depth_to_segmentation(depth_image) for depth_image in depth_images
    ]
    rgb_images = [depth_to_rgd(depth_image) for depth_image in depth_images]
    img_no_and_ts = {}
    start_time = time.time()

    # Save the intrensics
    with open(os.path.join(output_dir, "intrinsics.txt"), "w") as f:
        f.write(" ".join(map(str, camera.get_intrinsics())) + "\n")

    # Loop and Save the data
    for i, (depth_image, segmentation_image, rgb_image, pose) in enumerate(
        zip(depth_images, segmentation_images, rgb_images, camera_poses)
    ):
        # print(f"View {i}: ", pose)
        # plot_image(depth_image, segmentation_image, rgb_image, title=f"View {i}")

        # Image Name
        image_id = "%06d" % i
        img_no_and_ts[image_id] = start_time + i * 10

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
