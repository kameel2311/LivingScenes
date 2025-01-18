import sys

sys.path.append("../")
from utils.metrics_helper import compute_pointcloud_overlap
import numpy as np
import pyrender
import trimesh
import random
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.pointcloud_helper import (
    path_generator,
    sample_mesh_random,
    draw_point_cloud,
    draw_point_cloud_with_cameras,
    scale_point_cloud,
)


class Camera:
    def __init__(self, scale, image_height, image_width, fx, fy):
        self.scale = scale
        self.image_height = image_height  # * scale
        self.image_width = image_width  # * scale
        self.fx = fx / scale
        self.fy = fy / scale
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        self.k = np.array(
            [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1], dtype=np.float32
        ).reshape((3, 3))

    def get_pyrender_camera(self):
        return pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

    def get_intrinsics(self):
        return self.k


def transformation_matrix(rotation, translation):
    """
    Create a 4x4 transformation matrix from rotation and translation vectors.

    Parameters:
        rotation (numpy.ndarray): 3x3 rotation matrix.
        translation (numpy.ndarray): 3x1 translation vector.

    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
    # Create a 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = np.reshape(rotation, (3, 3))
    transformation[:3, 3] = np.reshape(translation, (3,))

    return transformation


def sample_points_in_circle(num_points, angle_lower, angle_upper, radius, center):
    """
    Sample points in a circle
    """
    # Generate points on a circle
    angle_lower = np.deg2rad(angle_lower)
    angle_upper = np.deg2rad(angle_upper)
    theta = np.linspace(angle_lower, angle_upper, num_points, endpoint=False)

    # Convert to cartesian coordinates
    x = radius * np.sin(theta)  # Hence first point is infront of Y axis
    y = -1 * radius * np.cos(theta)
    z = np.zeros_like(x)
    delta = np.stack((x, y, z), axis=1)
    camera_centers = center + delta

    return camera_centers, theta


def sample_points_in_circle_pySpace(
    num_points, angle_lower, angle_upper, radius, center
):
    """
    Sample points in a circle
    """
    # Generate points on a circle
    angle_lower = np.deg2rad(angle_lower)
    angle_upper = np.deg2rad(angle_upper)
    theta = np.linspace(angle_lower, angle_upper, num_points, endpoint=False)

    # Convert to cartesian coordinates
    x = radius * np.sin(theta)  # Hence first point is infront of Y axis
    z = radius * np.cos(theta)
    y = np.zeros_like(x)
    delta = np.stack((x, y, z), axis=1)
    camera_centers = (PY_R_W @ np.reshape(center, (3, 1))).T + delta

    return camera_centers, theta


W_R_C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
PY_R_W = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
PY_T_W = transformation_matrix(PY_R_W, np.zeros([1, 3]))
W_T_C = transformation_matrix(W_R_C, np.zeros([1, 3]))


def get_circle_poses(
    num_points,
    angle_lower,
    angle_upper,
    radius,
    center,
    pyrender_transform=PY_R_W,
    camera_transform=W_R_C,
    sequential=True,
):
    """
    Get camera poses in a circle in the world frame and pyrender frame
    """
    # Generate points on a circle
    angle_lower = np.deg2rad(angle_lower)
    angle_upper = np.deg2rad(angle_upper)
    if sequential:
        thetas = np.linspace(angle_lower, angle_upper, num_points, endpoint=False)
    else:
        thetas = np.random.uniform(angle_lower, angle_upper, num_points)

    print("Thetas: ", np.rad2deg(thetas))

    world_camera_poses = []
    pyrender_camera_poses = []

    # World Frame
    x = radius * np.sin(thetas)  # Hence first point is infront of Y axis
    y = -1 * radius * np.cos(thetas)
    z = np.zeros_like(x)
    delta = np.stack((x, y, z), axis=1)
    world_camera_centers = center + delta

    # Pyrender Frame
    z = radius * np.cos(thetas)
    y = np.zeros_like(x)
    delta = np.stack((x, y, z), axis=1)
    pyrender_camera_centers = (
        pyrender_transform @ np.reshape(center, (3, 1))
    ).T + delta

    for i in range(num_points):
        # World Frame
        rotation = (
            R.from_euler("z", thetas[i], degrees=False).as_matrix() @ camera_transform
        )
        world_camera_poses.append(
            transformation_matrix(rotation, world_camera_centers[i])
        )

        # Pyrender Frame
        rotation = R.from_euler("y", thetas[i], degrees=False).as_matrix()
        pyrender_camera_poses.append(
            transformation_matrix(rotation, pyrender_camera_centers[i])
        )

    return world_camera_poses, pyrender_camera_poses


def render_point_cloud_from_viewpoint(
    v,
    f,
    camera_py,
    k,
    image_width,
    image_height,
    num_points,
    world_pose,
    pyrender_pose,
    mesh_scale=1,
    pointcloud=None,
    visualize=False,
):

    # Rendering
    scene = pyrender.Scene()
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.apply_scale(mesh_scale)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # Define Scene and Renderer
    scene.add(mesh, pose=PY_T_W)
    scene.add(camera_py, pose=pyrender_pose)
    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    # Render Depth Image
    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    # Deproject & sample PC
    visible_points = deproject_depth_image(depth, k, y_multipler=-1)
    visible_points = visible_points[visible_points[:, 2] != 0]
    rows_id = random.sample(range(0, visible_points.shape[0] - 1), num_points)
    visible_points = visible_points[rows_id]

    # Transform point cloud to world coordinates
    visible_points = (world_pose[:3, :3] @ visible_points.T).T + world_pose[:3, 3]

    # Visualize
    if visualize:
        # pyrender.Viewer(
        #     scene, use_raymond_lighting=True, viewport_size=(image_width, image_height)
        # )
        plt.imshow(depth)
        plt.colorbar()
        plt.title("Depth Map")
        plt.show()
        if pointcloud is not None:
            draw_point_cloud_with_cameras(
                pointcloud,
                "Point Cloud with Circular camera and visible points",
                overlay_pointcloud=visible_points,
                cameras=[(k, [image_width, image_height], world_pose)],
            )
        else:
            draw_point_cloud_with_cameras(
                visible_points,
                "Point Cloud with Circular camera and visible points",
                cameras=[(k, [image_width, image_height], world_pose)],
            )

    return visible_points


def deproject_depth_image(depth, k, y_multipler=1):
    """
    Deproject depth image to point cloud

    Args:
        depth: HxW numpy array of depth values
        k: 3x3 camera intrinsic matrix

    Returns:
        point_cloud: Nx3 numpy array of 3D points
    """
    # Create grid of pixel coordinates
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()

    # Deproject depth image to 3D points
    z = depth.flatten()
    x = (u - k[0, 2]) * z / k[0, 0]
    y = y_multipler * (v - k[1, 2]) * z / k[1, 1]

    point_cloud = np.stack((x, y, z), axis=1)

    return point_cloud


# TODO: Implement Sampling as in Dep.pyrender_view_render (Spherical Sampling)

W_R_C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
PY_R_W = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
PY_T_W = transformation_matrix(PY_R_W, np.zeros([1, 3]))
W_T_C = transformation_matrix(W_R_C, np.zeros([1, 3]))

if __name__ == "__main__":
    DATA_DIR = "/Datasets/ModelNet10/ModelNet10"
    # Load Object Instance
    file = "train"

    # Define Camera
    image_height = 500
    image_width = 500
    camera = Camera(
        scale=1, image_height=image_height, image_width=image_width, fx=250, fy=250
    )
    k = camera.get_intrinsics()
    num_cameras = 5

    # Loop through the instances
    # for object_class in ["chair", "table", "monitor", "sofa"]:
    for object_class in ["sofa"]:
        # Champfer Distance
        mean_objects_champfer_distance = []
        for instance in range(1, 100):
            # for instance in [11]:
            path_to_file = path_generator(DATA_DIR, object_class, file, instance)

            # Load vertices and faces
            v, f = pcu.load_mesh_vf(path_to_file)

            # Sample points from mesh
            pointcloud = sample_mesh_random(v, f, num_samples=600)
            pointcloud, pointcloud_centered, center, scaling_factor = scale_point_cloud(
                pointcloud, inference_method=False, desired_max_dim=10
            )
            print("Scaling Factor: ", scaling_factor)

            # Camera Poses
            radius = np.max(np.linalg.norm(pointcloud_centered, axis=1)) * 1.5
            world_pose, pyrender_pose = get_circle_poses(5, 0, 120, radius, center)

            # Loading Mesh and Setup Camera
            camera_py = camera.get_pyrender_camera()
            renderer = pyrender.OffscreenRenderer(image_width, image_height)

            # Loop through the camera poses
            temp_distance = []
            for i in range(num_cameras):
                rendered_pc = render_point_cloud_from_viewpoint(
                    v,
                    f,
                    camera_py,
                    k,
                    image_width,
                    image_height,
                    600,
                    world_pose[i],
                    pyrender_pose[i],
                    mesh_scale=scaling_factor,
                    visualize=False,
                )
                if i != 0:
                    champfer_distance = pcu.chamfer_distance(
                        np.array(rendered_pc, order="C"),
                        np.array(prev_pointcloud, order="C"),
                    )
                    overlap = compute_pointcloud_overlap(
                        rendered_pc, prev_pointcloud, epsilon=champfer_distance
                    )
                    print(
                        f"Chamfer Distance between {i} and {i-1}: ", champfer_distance
                    )
                    print(f"Overlap Ratio between {i} and {i-1}: ", overlap)
                    draw_point_cloud(rendered_pc, overlay_pointcloud=prev_pointcloud)
                    prev_pointcloud = rendered_pc
                    temp_distance.append(champfer_distance)
                else:
                    prev_pointcloud = rendered_pc
            mean_objects_champfer_distance.append(np.mean(temp_distance))
            print(f"Mean {instance} Champfer Distance: ", np.mean(temp_distance))

        # Plot Champfer Distance of Object
        plt.plot(mean_objects_champfer_distance, label=object_class)
    plt.xlabel("Object Instance")
    plt.ylabel("Champfer Distance")
    plt.title("Champfer Distance of Object Instances")
    plt.legend()
    plt.show()

    # # Compare Metrics
    # for i in range(len(sampled_pointclouds) - 1):
    #     champfer_distance = pcu.chamfer_distance(
    #         np.array(sampled_pointclouds[i], order="C"),
    #         np.array(sampled_pointclouds[i + 1], order="C"),
    #     )
    #     print(f"Chamfer Distance between {i} and {i+1}: ", champfer_distance)
    #     draw_point_cloud(
    #         sampled_pointclouds[i], overlay_pointcloud=sampled_pointclouds[i + 1]
    #     )

    #     # Distance wrt to full point cloud
    #     champfer_distance = pcu.chamfer_distance(
    #         np.array(pointcloud, order="C"),
    #         np.array(sampled_pointclouds[i], order="C"),
    #     )
    #     print(f"Chamfer Distance between Origial and {i}: ", champfer_distance)
    #     draw_point_cloud(pointcloud, overlay_pointcloud=sampled_pointclouds[i])
