"""This script is targeted to simulate the effect of the usage of VN Encoders with Panoptic TSDFs for reconstruction enhancement."""

import os
import sys
import numpy as np
import point_cloud_utils as pcu

sys.path.append("../")
from utils.dataloader import Dataloader
from utils.pointcloud_helper import (
    draw_point_cloud,
    translate_pointcloud_center,
    sample_mesh_random,
    rotate_pointcloud,
    scale_point_cloud,
)
from utils.rendering_helper import (
    Camera,
    get_circle_poses,
    render_point_cloud_from_viewpoint,
)
from utils.metrics_helper import mean_absolute_distance, pointcloud_coverage

np.random.seed(0)


class Object:
    def __init__(
        self,
        path,
        num_points,
        num_rend_points,
        num_views,
        center,
        min_angle,
        max_angle,
        camera,
        translation=(0, 0, 0),
        random_rotation=False,
        fix_scaling=True,
    ):
        self.vertices, self.faces = pcu.load_mesh_vf(path)
        self.num_points = num_points
        self.num_rend_points = num_rend_points
        self.num_views = num_views
        self.center_pose = center
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.camera = camera
        self.fix_scaling = fix_scaling

        self._pointcloud, self._rendered_views = self.generate_pointclouds()
        self.translate_object(translation)

        # To Control the number of points in merged objects
        self._rendered_submap = None

    def generate_pointclouds(self):
        pointcloud = sample_mesh_random(
            self.vertices, self.faces, num_samples=self.num_points
        )
        # Preprocess Pointcloud
        pointcloud_scaled, pointcloud_scaled_centered, center, scaling_factor = (
            scale_point_cloud(pointcloud, inference_method=False, desired_max_dim=10)
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

        camera_pyrender = camera.get_pyrender_camera()
        k = camera.get_intrinsics()
        image_width, image_height = camera.get_resolution()
        rendered_views = [
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
                mesh_scale=scaling_factor,
                visualize=False,
                pointcloud=pointcloud_scaled,
            )
            for i in range(self.num_views)
        ]

        # Return to original scale
        if self.fix_scaling:
            pointcloud = pointcloud_scaled / scaling_factor
            rendered_views = rendered_views / scaling_factor
        return pointcloud, rendered_views

    def translate_object(self, translation):
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

    # TODO: Implement this
    def add_to_submap(self, rendered_idx):
        pass

    def visualize(self):
        for i in range(self.num_views):
            draw_point_cloud(
                self._pointcloud,
                overlay_pointcloud=self._rendered_views[i],
                title=f"Viewpoint {i}",
            )


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
    # Define the dataset to work with
    dataloader = Dataloader("ModelNet10", None)
    dataset_metadata = dataloader.get_metadata()
    scene_objects = ["chair", "table", "sofa"]
    scene_object_idx = [0, 1, 2]
    object_centers = [(40, 35, 0), (50, -20, 0), (-20, 0, 0)]
    object_num_samples = [600, 600, 1200]
    num_views = 4

    # Scene's Camera
    camera = Camera(scale=1, image_height=500, image_width=500, fx=250, fy=250)

    # Define Objects
    objects = []
    for object_class, object_idx in zip(scene_objects, scene_object_idx):
        path_to_file = dataloader.get_path(object_class, object_idx)
        v, f = pcu.load_mesh_vf(path_to_file)
        objects.append(
            Object(
                path=path_to_file,
                num_points=object_num_samples[object_idx],
                num_rend_points=int(object_num_samples[object_idx] / num_views),
                num_views=num_views,
                center=object_centers[object_idx],
                min_angle=0,
                max_angle=360,
                camera=camera,
                translation=object_centers[object_idx],
                fix_scaling=False,
            )
        )

    # Define the scene
    scene = Scene(objects)
    scene.add_to_scene(0, 0, yaw_angle=90)
    scene.add_to_scene(0, 1)
    scene.add_to_scene(1, 0)
    scene.add_to_scene(2, 1)
    scene.visualize()
    print("MAD VALUE: ", scene.get_MAD())
    print("Scene Coverage: ", scene.get_scene_coverage(epsilon=0.3))
