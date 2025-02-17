import sys
import os
import numpy as np
import point_cloud_utils as pcu
from matplotlib import pyplot as plt

sys.path.append("../")
from utils.pointcloud_helper import (
    draw_point_cloud,
    translate_pointcloud_center,
    sample_mesh_random,
    rotate_pointcloud,
    scale_point_cloud,
    center_pointcloud,
    center_pointcloud_v2,
    round_to_1,
)
from utils.rendering_helper import (
    Camera,
    get_circle_poses,
    render_point_cloud_from_viewpoint,
)
from utils.metrics_helper import mean_absolute_distance, pointcloud_coverage

# Consider /2 as dist from center is taken
CLASS_MAX_DIM_SIZE = {
    "bathtub": 3,
    "bed": 3,
    "chair": 1,
    "desk": 2,
    "dresser": 4,
    "monitor": 0.5,
    "night_stand": 1,
    "sofa": 3,
    "table": 4,
    "toilet": 1.5,
}


# TODO: Change way of generating the pointclouds if it takes too much RAM
class Object:
    def __init__(
        self,
        semantic_class,
        path,
        num_points,  # MINIMUM NUMBER OF POINTS
        num_rend_points,  # NUMBER OF POINTS TO RENDER
        num_views,
        min_angle,
        max_angle,
        camera,
        max_dim=10,
        translation=(0, 0, 0),
        random_rotation=False,
        scaling_Mode="rendering",
        adapt_num_points=False,
        save_depth=False,
        w_T_delta_pose_change=None,
        verbose=False,
    ):
        self.semantic_class = semantic_class
        self.vertices, self.faces = pcu.load_mesh_vf(path)
        self.num_points = num_points
        self.num_rend_points = num_rend_points
        self.num_views = num_views
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.camera = camera
        self.max_dim = max_dim
        self.scaling_Mode = scaling_Mode
        self.adapt_num_points = adapt_num_points
        self.w_T_delta_pose_change = w_T_delta_pose_change
        self.save_depth = save_depth
        self._depth_images = []
        self.verbose = verbose

        (
            self._pointcloud,
            self._rendered_views,
            self._world_poses,
            self._pyrender_poses,
        ) = self.generate_pointclouds(camera)
        self.translate_object(translation)

        # Final Pointcloud Scale (As Rendered, Original Mesh or Class Dict Based)
        assert scaling_Mode in [
            "rendering",
            "original",
            "class_based",
        ], "Fix Scaling should be either 'rendering', 'original' or 'class_based'"

    def generate_pointclouds(self, camera):
        # Sample Pointcloud
        pointcloud = sample_mesh_random(
            self.vertices, self.faces, num_samples=self.num_points
        )

        # Preprocess Pointcloud
        pointcloud_scaled, pointcloud_scaled_centered, center, rend_scaling_factor = (
            scale_point_cloud(
                pointcloud, inference_method=False, desired_max_dim=self.max_dim
            )
        )

        # Define the scaling factor
        if self.scaling_Mode == "rendering":
            scaling_factor = 1
        elif self.scaling_Mode == "original":
            scaling_factor = 1 / rend_scaling_factor
        elif self.scaling_Mode == "class_based":
            scaling_factor = round_to_1(
                CLASS_MAX_DIM_SIZE.get(self.semantic_class)
                / np.max(np.linalg.norm(pointcloud - center, axis=1))
            )

        # If Sampling Adaptation is true, then need to resample
        if self.adapt_num_points and scaling_factor != 1:
            self.num_points = np.max(
                [int(self.num_points * scaling_factor), self.num_points]
            )
            self.num_rend_points = np.max(
                [int(self.num_rend_points * scaling_factor), self.num_rend_points]
            )

        # Camera Poses
        radius = np.max(np.linalg.norm(pointcloud_scaled_centered, axis=1)) * 1.5
        world_poses, pyrender_poses = get_circle_poses(
            self.num_views,
            self.min_angle,
            self.max_angle,
            radius,
            center,
            sequential=True,
        )

        # Render Pointclouds
        camera_pyrender = camera.get_pyrender_camera()
        k = camera.get_intrinsics()
        image_width, image_height = camera.get_resolution()
        rendered_output = [
            render_point_cloud_from_viewpoint(
                self.vertices,
                self.faces,
                camera_pyrender,
                k,
                image_width,
                image_height,
                self.num_rend_points,
                world_poses[i],
                pyrender_poses[i],
                mesh_scale=rend_scaling_factor,
                visualize=False,
                pointcloud=pointcloud_scaled,
                return_depth=self.save_depth,
                mesh_transform=self.w_T_delta_pose_change,
            )
            for i in range(self.num_views)
        ]
        # Save Depth Images
        if self.save_depth:
            rendered_views, depth_images = zip(*rendered_output)
            self._depth_images = depth_images
        else:
            rendered_views = rendered_output

        # Scale Pointcloud and Rendered Views
        if self.scaling_Mode != "rendering":
            # Resample original pointcloud
            if self.adapt_num_points:
                pointcloud = sample_mesh_random(
                    self.vertices, self.faces, num_samples=self.num_points
                )
            pointcloud = pointcloud_scaled * scaling_factor
            rendered_views = [
                rendered_view * scaling_factor for rendered_view in rendered_views
            ]

        # Set Print Statements
        if self.verbose:
            print(f"Rendering Scaling Factor: {rend_scaling_factor}")
            print(f"Scaling Factor: {scaling_factor}")
            print(f"Scaling Mode: {self.scaling_Mode}")
            print(f"Adapt Num Points: {self.adapt_num_points}")
            print(f"Num Points: {self.num_points}")
            print(f"Num Rend Points: {self.num_rend_points}")

        return pointcloud, rendered_views, world_poses, pyrender_poses

    def translate_object(self, translation, level_z=True):
        if not level_z:
            z_delta = 0
        else:
            z_delta = -np.min(self._pointcloud[:, -1])
        translation = translation + (z_delta)
        self._pointcloud, shift = translate_pointcloud_center(
            self._pointcloud, translation
        )
        self._rendered_views = [
            rendered_view + shift for rendered_view in self._rendered_views
        ]

    def get_pointcloud(self):
        return self._pointcloud

    def get_rendered_view(self, idx):
        assert idx < self.num_views, f"Index out of bounds as {idx} >= {self.num_views}"
        return self._rendered_views[idx]

    def get_pyrender_poses(self):
        return self._pyrender_poses

    def get_world_poses(self):
        return self._world_poses

    def get_depth_images(self):
        assert self.save_depth, "Depth Images not saved"
        assert len(self._depth_images), "No images have been saved"
        return self._depth_images

    # TODO: Implement this
    def add_to_submap(self, rendered_idx):
        pass

    def visualize(self, depth=False):
        for i in range(self.num_views):
            draw_point_cloud(
                self._pointcloud,
                overlay_pointcloud=self._rendered_views[i],
                title=f"Viewpoint {i}",
            )
            if depth:
                assert self.save_depth, "Depth Images not saved"
                plt.imshow(self._depth_images[i])
                plt.colorbar()
                plt.title(f"Depth Map {i}")
                plt.show()

    def sample_pointcloud(self, num_points: int):
        pointcloud = sample_mesh_random(
            self.vertices, self.faces, num_samples=num_points
        )
        _, center = center_pointcloud_v2(pointcloud)
        print(center)
        pointcloud[:, -1] += -3.743
        pointcloud *= 0.06
        return pointcloud


class Scene:
    def __init__(self, objects: list[Object]):
        self.objects = objects
        self._gt_scene_pointcloud = self.create_gt_scene()
        self._simulated_scene_pointcloud = []
        self._simulated_scene_history = []

    def create_gt_scene(self):
        scene_pointcloud = [object.get_pointcloud() for object in self.objects]
        scene_pointcloud = np.concatenate(scene_pointcloud, axis=0)
        return scene_pointcloud

    def get_gt_scene(self):
        return self._gt_scene_pointcloud

    # TODO: Implement subsampling from object to have better object pc distribution
    def add_to_scene(self, object_idx, rendered_idx, yaw_angle=None):
        if (object_idx, rendered_idx) in self._simulated_scene_history:
            print("Object already added to the scene")
        else:
            object = self.objects[object_idx]
            rendered_view = object.get_rendered_view(rendered_idx)
            if yaw_angle is not None:
                rendered_view, _ = rotate_pointcloud(
                    rendered_view,
                    rotation_matrix=None,
                    about_center=True,
                    z_angle=yaw_angle,
                )

            self._simulated_scene_pointcloud.append(rendered_view)
            self._simulated_scene_history.append((object_idx, rendered_idx))

    def get_simulated_scene(self):
        if len(self._simulated_scene_history):
            return np.concatenate(self._simulated_scene_pointcloud, axis=0)
        else:
            raise ValueError("No objects added to the scene")

    def visualize(self):
        print(self._gt_scene_pointcloud.shape)

        draw_point_cloud(
            self._gt_scene_pointcloud,
            overlay_pointcloud=self.get_simulated_scene(),
            title="Simulated Scene",
        )

    def get_MAD(self):
        if len(self._simulated_scene_history):
            return mean_absolute_distance(
                self._gt_scene_pointcloud, self.get_simulated_scene()
            )
        else:
            raise ValueError("No objects added to the scene")

    def get_scene_coverage(self, epsilon=0.1):
        if len(self._simulated_scene_history):
            return pointcloud_coverage(
                self._gt_scene_pointcloud, self.get_simulated_scene(), epsilon
            )
        else:
            raise ValueError("No objects added to the scene")


if __name__ == "__main__":
    from utils.dataloader import Dataloader

    object_class = "chair"
    object_idx = 2
    num_points = 2000
    num_views = 4

    print(f"Testing Object Class: {object_class} and Index: {object_idx}")

    # Loading the Object
    dataloader = Dataloader("ModelNet10", None)
    path_to_file = dataloader.get_path(object_class, object_idx)

    # Defining the Camera
    camera = Camera(scale=1, image_height=500, image_width=500, fx=250, fy=250)
    object = Object(
        semantic_class=object_class,
        path=path_to_file,
        num_points=num_points,
        num_rend_points=1000,
        num_views=num_views,
        min_angle=0,
        max_angle=360,
        camera=camera,
        scaling_Mode="class_based",
        adapt_num_points=True,
        save_depth=True,
        verbose=True,
    )
    object.visualize(depth=True)
