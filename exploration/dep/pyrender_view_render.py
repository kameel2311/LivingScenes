import numpy as np
import point_cloud_utils as pcu
from exploration.utils.pointcloud_helper import *
import pyrender
import trimesh
import random

scale = 1
image_height = 500 * scale
image_width = 500 * scale
fx = 200 * scale
fy = 200 * scale
cx = image_width / 2
cy = image_height / 2

# image_height = 2000
# image_width = 2000
# fx = 500
# fy = 500
# cx = image_width / 2
# cy = image_height / 2
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))


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


def points_around_sphere(num_points, radius=1):
    """Generate points around a sphere"""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack((x, y, z), axis=1)


# def calculate_aligning_rotation(focalpoint, k, image_width, image_height):
#     """Calculate rotation to align camera with focal point/camrea center"""
#     focalpoint = np.reshape(focalpoint, (3))
#     image_center = np.array([image_width / 2, image_height / 2, 1])
#     deproj_image_center = np.linalg.inv(k) @ image_center * 1
#     rotation = find_rotation(deproj_image_center, -1 * focalpoint)
#     return rotation


def calculate_aligning_rotation(
    focalpoint, k, image_width, image_height, c_R_w=np.eye(3)
):
    """Calculate rotation to align camera with focal point/camrea center
    Args:
        focalpoint: 3D point to align camera with in world coordinates
        k: 3x3 camera intrinsic matrix
        image_width: image width
        image_height: image height
        c_R_w: rotation matrix from world to camera coordinates"""

    focalpoint = np.reshape(focalpoint, (3))
    focalpoint = (c_R_w @ focalpoint.T).T
    image_center = np.array([image_width / 2, image_height / 2, 1])
    deproj_image_center = np.linalg.inv(k) @ image_center * 1
    rotation = find_rotation(deproj_image_center, -1 * focalpoint)
    rotation = c_R_w.T @ rotation
    return rotation


def find_rotation(v1, v2):
    """Find rotation matrix between two vectors"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        return np.eye(3)

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    return R.from_rotvec(angle * axis).as_matrix()


def convert_cam_pose_to_opengl(cv_pose):
    """
    Convert camera pose from computer vision convention (+Z forward)
    to OpenGL convention (-Z forward) and Y down to Y up
    """
    # OpenGL to CV coordinate change: flip Y and Z
    coord_change = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]  # Flip Y  # Flip Z
    )

    return coord_change @ cv_pose @ np.linalg.inv(coord_change)


def render_depth_map(mesh_vertices, mesh_faces, camera_pose):
    """
    Render depth map using pyrender

    Args:
        mesh_vertices: vertices of the mesh
        mesh_faces: faces of the mesh
        camera_pose: 4x4 camera pose matrix
        near: near plane distance
        far: far plane distance

    Returns:
        depth_map: HxW numpy array of depth values
    """
    # Create scene and add mesh
    scene = pyrender.Scene()

    # Create mesh
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # Create camera
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

    # camera_pose = np.linalg.inv(camera_pose)
    camera_pose = convert_cam_pose_to_opengl(camera_pose)
    # camera_pose = np.linalg.inv(camera_pose)

    # Add camera to scene
    scene.add(camera, pose=camera_pose)

    # Create renderer
    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    # Render depth
    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    return depth


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


def sample_viewpoint(pointcloud, k, vertices, faces, num_points):
    """
    Sample a viewpoint from the point cloud

    Args:
        pointcloud: Nx3 numpy array of 3D points
        k: 3x3 camera intrinsic matrix

    Returns:
        pose: 4x4 camera pose matrix
    """
    pointcloud_centered, center = center_pointcloud_v2(pointcloud)
    max_radius = np.max(np.linalg.norm(pointcloud_centered, axis=1))
    sphere_point = points_around_sphere(1, radius=max_radius + 1.5)[0]
    rotation = calculate_aligning_rotation(sphere_point)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = sphere_point + center
    depth = render_depth_map(vertices, faces, pose)
    visible_points = deproject_depth_image(depth, k)
    rows_id = random.sample(range(0, visible_points.shape[0] - 1), num_points)
    visible_points = visible_points[rows_id]
    return visible_points


if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 3
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces
    v, f = pcu.load_mesh_vf(path_to_file)

    # Sample points from mesh
    pointcloud = sample_mesh_random(v, f, num_samples=600)
    # pointcloud_centered = center_pointcloud(pointcloud)
    pointcloud_centered, center = center_pointcloud_v2(pointcloud)
    max_radius = np.max(np.linalg.norm(pointcloud_centered, axis=1))

    # Generate camera positions
    sphere_points = points_around_sphere(24, radius=max_radius + 1.5)

    # Render from multiple viewpoints
    depth_maps = []
    for point in sphere_points:
        # Calculate rotation
        rotation = calculate_aligning_rotation(point)

        # Create camera pose matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation
        # pose[:3, 3] = point + np.mean(pointcloud, axis=0)
        pose[:3, 3] = point + center

        # Render depth map
        depth = render_depth_map(v, f, pose)
        depth_maps.append(depth)

        print(f"Depth map shape: {depth.shape}")
        print(f"Depth range: {np.min(depth[depth >= 0])} to {np.max(depth)}")

        # Deproject depth image to point cloud
        visible_points = deproject_depth_image(depth, k)
        rows_id = random.sample(range(0, visible_points.shape[0] - 1), 600)
        visible_points = visible_points[rows_id]

        # # Transform point cloud to world coordinates
        # inv_pose = np.linalg.inv(pose)
        # visible_points = inv_pose[:3, :3] @ visible_points.T + inv_pose[:3, 3]

        # Plot camera pose
        draw_point_cloud_with_cameras(
            pointcloud,
            "Cameras",
            cameras=[(k, [image_width, image_height], pose)],
            overlay_pointcloud=visible_points,
        )

        plt.imshow(depth)
        plt.colorbar()
        plt.title("Depth Map")
        plt.show()

    # Optional: visualize first depth map
    import matplotlib.pyplot as plt

    for depth_map in depth_maps:
        plt.imshow(depth_map)
        plt.colorbar()
        plt.title("Depth Map")
        plt.show()
