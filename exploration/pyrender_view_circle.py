import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh
import random
from point_cloud import (
    path_generator,
    sample_mesh_random,
    center_pointcloud,
    draw_point_cloud_with_cameras,
)
from pyrender_view_render import transformation_matrix, deproject_depth_image
import point_cloud_utils as pcu
import matplotlib.pyplot as plt

DATA_DIR = "/Datasets/ModelNet10/ModelNet10"

scale = 1
image_height = 500 * scale
image_width = 500 * scale
fx = 250 * scale
fy = 250 * scale
cx = image_width / 2
cy = image_height / 2
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))

W_R_C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
PY_R_W = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
PY_T_W = transformation_matrix(PY_R_W, np.zeros([1, 3]))
W_T_C = transformation_matrix(W_R_C, np.zeros([1, 3]))


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


def get_circle_poses(
    num_points,
    angle_lower,
    angle_upper,
    radius,
    center,
    pyrender_transform=PY_R_W,
    camera_transform=W_R_C,
):
    """
    Get camera poses in a circle in the world frame and pyrender frame
    """
    # Generate points on a circle
    angle_lower = np.deg2rad(angle_lower)
    angle_upper = np.deg2rad(angle_upper)
    thetas = np.linspace(angle_lower, angle_upper, num_points, endpoint=False)

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


def render_point_cloud_from_mesh(
    v,
    f,
    camera_py,
    k,
    image_width,
    image_height,
    num_points,
    world_pose,
    pyrender_pose,
    visualize=False,
):
    if visualize:
        draw_point_cloud_with_cameras(
            pointcloud,
            "Point Cloud with Circular cameras",
            cameras=[(k, [image_width, image_height], world_pose)],
        )

    # Rendering
    scene = pyrender.Scene()
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh, pose=PY_T_W)
    scene.add(camera_py, pose=pyrender_pose)

    pyrender.Viewer(
        scene, use_raymond_lighting=True, viewport_size=(image_width, image_height)
    )

    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    if visualize:
        plt.imshow(depth)
        plt.colorbar()
        plt.title("Depth Map")
        plt.show()

    # Deproject & sample PC
    visible_points = deproject_depth_image(depth, k, y_multipler=-1)
    rows_id = random.sample(range(0, visible_points.shape[0] - 1), num_points)
    visible_points = visible_points[rows_id]

    # Transform point cloud to world coordinates
    visible_points = (world_pose[:3, :3] @ visible_points.T).T + world_pose[:3, 3]

    if visualize:
        draw_point_cloud_with_cameras(
            pointcloud,
            "Point Cloud with Circular camera and visible points",
            overlay_pointcloud=visible_points,
            cameras=[(k, [image_width, image_height], world_pose)],
        )
    return visible_points


if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 2
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces
    v, f = pcu.load_mesh_vf(path_to_file)

    # Sample points from mesh
    pointcloud = sample_mesh_random(v, f, num_samples=600)
    pointcloud_centered, center = center_pointcloud(pointcloud)
    center = np.reshape(center, (1, 3))

    # Camera Poses
    num_cameras = 5
    radius = np.max(np.linalg.norm(pointcloud_centered, axis=1)) * 1.5
    world_pose, pyrender_pose = get_circle_poses(5, 0, 120, radius, center)

    # Loading Mesh and Setup Camera
    camera_py = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    # Loop through the camera poses
    for i in range(num_cameras):
        rendered_pc = render_point_cloud_from_mesh(
            v,
            f,
            camera_py,
            k,
            image_width,
            image_height,
            600,
            world_pose[i],
            pyrender_pose[i],
            visualize=True,
        )
