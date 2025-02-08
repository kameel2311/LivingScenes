import sys
import os
import numpy as np
import point_cloud_utils as pcu
from matplotlib import pyplot as plt

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


class Object:
    def __init__(
        self,
        path,
        num_points,
        num_rend_points,
        num_views,
        min_angle,
        max_angle,
        camera,
        translation=(0, 0, 0),
        random_rotation=False,
        fix_scaling=True,
        save_depth=False,
    ):
        self.vertices, self.faces = pcu.load_mesh_vf(path)
        self.num_points = num_points
        self.num_rend_points = num_rend_points
        self.num_views = num_views
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.camera = camera
        self.fix_scaling = fix_scaling
        self.save_depth = save_depth
        self._depth_images = []

        (
            self._pointcloud,
            self._rendered_views,
            _,
            self._pyrender_poses,
        ) = self.generate_pointclouds(camera)
        self.translate_object(translation)

        # To Control the number of points in merged objects
        self._rendered_submap = None

    def generate_pointclouds(self, camera):
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
                mesh_scale=scaling_factor,
                visualize=False,
                pointcloud=pointcloud_scaled,
                return_depth=self.save_depth,
            )
            for i in range(self.num_views)
        ]
        if self.save_depth:
            rendered_views, depth_images = zip(*rendered_output)
            self._depth_images = depth_images
        else:
            rendered_views = rendered_output

        # Return to original scale
        if self.fix_scaling:
            pointcloud = pointcloud_scaled / scaling_factor
            rendered_views = rendered_views / scaling_factor
        return pointcloud, rendered_views, world_poses, pyrender_poses

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

    def get_pyrender_poses(self):
        return self._pyrender_poses

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
