import sys
import math
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
from utils.plotting_utils import plot_gradual_metrics, plot_object_metrics

# Consider /2 as dist from center is taken
CLASS_MAX_DIM_SIZE = {
    "bathtub": 2,
    "bed": 2,
    "chair": 1,
    "desk": 2,
    "dresser": 2,
    "monitor": 0.5,
    "night_stand": 1,
    "sofa": 2,
    "table": 2,
    "toilet": 1.5,
}


# TODO: Change way of generating the pointclouds if it takes too much RAM
class Object:
    def __init__(
        self,
        semantic_class,
        semantic_idx,
        path,
        num_points,  # MINIMUM NUMBER OF POINTS
        num_rend_points,  # NUMBER OF POINTS TO RENDER
        num_views,
        min_angle,
        max_angle,
        camera,
        max_dim=10,
        center=None,
        random_rotation=False,
        scaling_Mode="rendering",
        adapt_num_points=False,
        save_depth=False,
        w_T_delta_pose_change=None,
        verbose=False,
    ):
        self.semantic_class = semantic_class
        self.semantic_idx = semantic_idx
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

        if center is not None:
            self.center_object(center)

        # Final Pointcloud Scale (As Rendered, Original Mesh or Class Dict Based)
        assert scaling_Mode in [
            "rendering",
            "original",
            "class_based",
        ], "Fix Scaling should be either 'rendering', 'original' or 'class_based'"

    def object_scale(self, pointcloud, center, rend_scaling_factor):
        # Define the scaling factor & adapt the number of points
        if self.scaling_Mode == "rendering":
            scaling_factor = 1
        elif self.scaling_Mode == "original":
            scaling_factor = 1 / rend_scaling_factor
            if self.adapt_num_points:
                self.num_points = int(self.num_points * scaling_factor)
                self.num_rend_points = int(self.num_rend_points * scaling_factor)
        elif self.scaling_Mode == "class_based":
            scaling_factor = round_to_1(
                CLASS_MAX_DIM_SIZE.get(self.semantic_class)
                / np.max(np.max(np.abs(pointcloud - center), axis=1) * 2)
            )
            if self.adapt_num_points:  # Since Scaling Factor is always less than 1
                self.num_points = int(
                    self.num_points * CLASS_MAX_DIM_SIZE.get(self.semantic_class)
                )
                self.num_rend_points = int(
                    self.num_rend_points * CLASS_MAX_DIM_SIZE.get(self.semantic_class)
                )
        return scaling_factor

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

        # Get Scaling Factor and Adapt Num Points
        scaling_factor = self.object_scale(
            pointcloud_scaled, center, rend_scaling_factor
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
            print(f"Semantic Class: {self.semantic_class}")
            print(f"Rendering Scaling Factor: {rend_scaling_factor}")
            print(f"Scaling Factor: {scaling_factor}")
            print(f"Scaling Mode: {self.scaling_Mode}")
            print(f"Adapt Num Points: {self.adapt_num_points}")
            print(f"Num Points: {self.num_points}")
            print(f"Num Rend Points: {self.num_rend_points}")

        return pointcloud, rendered_views, world_poses, pyrender_poses

    def center_object(self, translation: list[float, float], level_z=True):
        if not level_z:
            z_delta = 0
        else:
            z_delta = -np.min(self._pointcloud[:, -1])
        translation.append(z_delta)
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

    # TODO: FIX THIS BUG
    def sample_pointcloud(self, num_points: int):
        pointcloud = sample_mesh_random(
            self.vertices, self.faces, num_samples=num_points
        )
        _, center = center_pointcloud_v2(pointcloud)
        pointcloud[:, -1] += -3.743
        pointcloud *= 0.06
        return pointcloud

    def get_object_xy_radius(self):
        centered_pointcloud, _ = center_pointcloud(self._pointcloud)
        return np.max(np.linalg.norm(centered_pointcloud[:, :2], axis=1))

    def get_object_MAD(self, used_views: list):
        merged_rendered_pointclouds = np.concatenate(
            [self._rendered_views[i] for i in used_views], axis=0
        )
        return mean_absolute_distance(self._pointcloud, merged_rendered_pointclouds)

    def get_object_coverage(self, used_views: list, epsilon=0.1):
        merged_rendered_pointclouds = np.concatenate(
            [self._rendered_views[i] for i in used_views], axis=0
        )
        return pointcloud_coverage(
            self._pointcloud, merged_rendered_pointclouds, epsilon
        )


class Scene:
    def __init__(self, objects: list[Object], distribute: bool = True):
        self.objects = objects
        if distribute:
            self.distribute_objects()
        self._gt_scene_pointcloud = None
        self._simulated_scene_pointcloud = []
        self._simulated_scene_history = []

    def distribute_objects(self, extra_spacing=0.5):
        # Get Max Radius of all objects
        object_radii = [object.get_object_xy_radius() for object in self.objects]
        max_radius = np.max(object_radii)
        distance_between_objects = 2 * max_radius + extra_spacing

        # Grid Like Distribution
        grid_size = int(math.ceil(math.sqrt(len(self.objects))))
        x = np.linspace(
            -distance_between_objects * grid_size / 2,
            distance_between_objects * grid_size / 2,
            grid_size,
        )
        y = np.linspace(
            -distance_between_objects * grid_size / 2,
            distance_between_objects * grid_size / 2,
            grid_size,
        )
        x, y = np.meshgrid(x, y)
        object_centers = np.array([x.flatten(), y.flatten()]).T

        # Distribute Objects
        for object, object_center in zip(self.objects, object_centers):
            object.center_object(list(object_center))

    def create_gt_scene(self, config: dict):
        scene_pointcloud = []
        for idx, object in enumerate(self.objects):
            if idx not in config["skip_objects_from_gt"]:
                scene_pointcloud.append(object.get_pointcloud())
        scene_pointcloud = np.concatenate(scene_pointcloud, axis=0)
        return scene_pointcloud

    def set_gt_scene(self, config: dict):
        self._gt_scene_pointcloud = self.create_gt_scene(config)

    def get_gt_scene(self):
        return self._gt_scene_pointcloud

    def inflict_scene_changes(self, changes_dict: dict):
        # Add Objects wrt to Visiblity
        gradual_metrics = []
        per_object_gradual_metrics = {}
        if changes_dict["changed_objects"] is None:
            changed_idxs = []
        elif isinstance(changes_dict["changed_objects"], list):
            changed_idxs = changes_dict["changed_objects"]
        elif isinstance(changes_dict["changed_objects"], float):
            # Sequential Selection
            number_changed = int(changes_dict["changed_objects"] * len(self.objects))
            changed_idxs = list(range(len(self.objects)))[:number_changed]
        else:
            raise ValueError("Changed Objects not defined properly")

        print(f"Changed Idxs: {changed_idxs}")

        for idx in range(len(self.objects)):
            if (
                idx in changes_dict["skip_objects_from_gt"]
                and not changes_dict["map_skipped_objects"]
            ):
                continue
            if idx in changed_idxs:
                visibility = changes_dict["changed_objects_visibility"]
            else:
                visibility = changes_dict["unchanged_objects_visibility"]

            # Sequential Visibility Assumed
            max_view_id = int(visibility * self.objects[idx].num_views)
            view_ids = []  # If Sequential is no longer used in future

            # Check if Rotation Error is Inflicted
            yaw_angle = None
            delta_trans = None
            if idx in changed_idxs:
                if changes_dict["rotation"] is not None:
                    rotational_error = changes_dict["rotation"]
                    if rotational_error["sampling_distribution"] == "uniform":
                        yaw_angle = np.random.uniform(
                            rotational_error["min_angle_error"],
                            rotational_error["max_angle_error"],
                        )
                    else:
                        raise NotImplementedError(
                            "Sampling Distribution not implemented yet"
                        )
                if changes_dict["translation"] is not None:
                    translation_error = changes_dict["translation"]
                    if translation_error["sampling_distribution"] == "uniform":
                        delta_trans = np.append(
                            np.random.uniform(
                                translation_error["min_displacement_xy"],
                                translation_error["max_displacement_xy"],
                                size=(2),
                            ),
                            0,
                        )
                    else:
                        raise NotImplementedError(
                            "Sampling Distribution not implemented yet"
                        )

            for view_id in range(max_view_id):
                view_ids.append(view_id)
                self.add_to_scene(idx, view_id, yaw_angle, delta_trans)
            # Scene Metrics
            gradual_metrics.append(
                (
                    self.get_scene_MAD(),
                    self.get_scene_coverage(epsilon=changes_dict["coverage_epsilon"]),
                )
            )
            # Per Object Metrics
            base_object_key = (
                f"{self.objects[idx].semantic_class}_{self.objects[idx].semantic_idx}"
            )
            object_key = base_object_key
            counter = 1
            while object_key in per_object_gradual_metrics:
                object_key = f"{base_object_key}_{counter}"
                counter += 1
            per_object_gradual_metrics[object_key] = (
                self.objects[idx].get_object_MAD(view_ids),
                self.objects[idx].get_object_coverage(
                    view_ids, epsilon=changes_dict["coverage_epsilon"]
                ),
            )
        return gradual_metrics, per_object_gradual_metrics

    # TODO: Implement subsampling from object to have better object pc distribution
    def add_to_scene(self, object_idx, rendered_idx, yaw_angle=None, delta_trans=None):
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
            if delta_trans is not None:
                rendered_view += delta_trans

            self._simulated_scene_pointcloud.append(rendered_view)
            self._simulated_scene_history.append((object_idx, rendered_idx))

    def get_simulated_scene(self):
        if len(self._simulated_scene_history):
            return np.concatenate(self._simulated_scene_pointcloud, axis=0)
        else:
            raise ValueError("No objects added to the scene")

    def visualize(self, title=None):
        # print(f"Scene Shape: {self._gt_scene_pointcloud.shape}")

        try:
            overlay_pointcloud = self.get_simulated_scene()
        except ValueError:
            overlay_pointcloud = None
            print("NOTE: No objects added to the scene")

        draw_point_cloud(
            self._gt_scene_pointcloud,
            overlay_pointcloud=overlay_pointcloud,
            title=title,
        )

    def get_scene_MAD(self):
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

    def clear_scene(self):
        self._simulated_scene_pointcloud = []
        self._simulated_scene_history = []
        self._gt_scene_pointcloud = None


class BenchmarkRunner:
    def __init__(self, objects: list, reconstruction_config: dict):
        self.objects = objects
        self.config = reconstruction_config
        self.panoptic_scene_metrics, self.panoptic_object_metrics = [], []
        self.vn_enhanced_scene_metrics, self.vn_enhanced_object_metrics = [], []

        # For Benchmark Run
        self.x_axis = None
        self.x_label = None
        self.plot_title = None

    def generate_run_settings(self):
        if self.config["run_type"] == "single":
            panoptic_run_settings = self.config["single_settings"]["panoptic"]
            vn_enhanced_run_settings = self.config["single_settings"]["vn_enhanced"]
            self.plot_title = "Single Object Run"
            yield panoptic_run_settings, vn_enhanced_run_settings
        elif self.config["run_type"] == "benchmark":
            panoptic_run_settings = self.config["study"]["default_settings"]["panoptic"]
            vn_enhanced_run_settings = self.config["study"]["default_settings"][
                "vn_enhanced"
            ]
            study = self.config["study"]["name"]
            params = self.config["study"]["params"]

            # Parse the Settings
            if study == "percentage_changed_objects":
                values = np.arange(
                    params["min"], params["max"] + params["step"], params["step"]
                )
                # Setting Plot Params
                self.x_axis = values
                self.x_label = "Percentage of Changed Objects"
                self.plot_title = "Percentage of Changed Objects Benchmark"
                print(f"Values: {values}")
                for value in values:
                    panoptic_run_settings["changed_objects"] = value
                    vn_enhanced_run_settings["changed_objects"] = value
                    yield panoptic_run_settings, vn_enhanced_run_settings
        else:
            raise NotImplementedError("Benchmark Type not implemented yet")

    def run_benchmark(self):
        for (
            panoptic_settings,
            vn_enhanced_settings,
        ) in self.generate_run_settings():
            # Create Scene
            scene = Scene(self.objects)

            print(f"Panoptic Settings: {panoptic_settings}")
            print(f"VN Enhanced Settings: {vn_enhanced_settings}")

            # Gather Panoptic Metrics
            scene.set_gt_scene(panoptic_settings)
            panoptic_scene_metrics, panoptic_object_metrics = (
                scene.inflict_scene_changes(panoptic_settings)
            )
            scene.visualize(title="Panoptic Scene")

            # Gather Enhancement* Metrics
            scene.clear_scene()
            scene.set_gt_scene(vn_enhanced_settings)
            vn_enhanced_scene_metrics, vn_enhanced_object_metrics = (
                scene.inflict_scene_changes(vn_enhanced_settings)
            )
            scene.visualize(title="VN Enhanced Scene")

            if self.config["run_type"] == "single":
                self.panoptic_scene_metrics = panoptic_scene_metrics
                self.panoptic_object_metrics = panoptic_object_metrics
                self.vn_enhanced_scene_metrics = vn_enhanced_scene_metrics
                self.vn_enhanced_object_metrics = vn_enhanced_object_metrics
            elif self.config["run_type"] == "benchmark":
                self.panoptic_scene_metrics.append(panoptic_scene_metrics[-1])
                self.vn_enhanced_scene_metrics.append(vn_enhanced_scene_metrics[-1])

    def plot_metrics(self):
        if not (
            len(self.panoptic_scene_metrics) and len(self.vn_enhanced_scene_metrics)
        ):
            raise ValueError("Run the benchmark first")

        plot_gradual_metrics(
            title=self.plot_title,
            x_axis=self.x_axis,
            x_label=self.x_label,
            panoptic=self.panoptic_scene_metrics,
            vn_enhanced=self.vn_enhanced_scene_metrics,
        )
        if self.config["run_type"] == "single":

            plot_object_metrics(
                panoptic=self.panoptic_object_metrics,
                vn_enhancement=self.vn_enhanced_object_metrics,
            )


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
