import numpy as np
import point_cloud_utils as pcu
from point_cloud import *
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

scale = 10
image_height = 100 * scale
image_width = 100 * scale
fx = 100 * scale
fy = 100 * scale
cx = 50 * scale
cy = 50 * scale
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))


def points_around_sphere(num_points, radius=1):
    """
    Generate points around a sphere
    """
    # Generate points on a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)

    # Convert to cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack((x, y, z), axis=1)


def calculate_aligning_rotation(focalpoint):
    """
    Calculate the rotation to align camera with the focal point
    """
    camera_center = np.array([image_width / 2, image_height / 2, 1])
    deproj_camera_center = np.linalg.inv(k) @ camera_center * 1
    # deproj_camera_center -= focalpoint
    rotation = find_rotation(deproj_camera_center, -1 * focalpoint)
    return rotation


def find_rotation(v1, v2):
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the axis of rotation (cross product)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # If vectors are parallel, no rotation is needed
    if axis_norm < 1e-6:
        return np.eye(3)  # Identity matrix

    axis = axis / axis_norm

    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # Use scipy to compute the rotation matrix
    rot_matrix = R.from_rotvec(angle * axis).as_matrix()

    return rot_matrix


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
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation

    return transformation


if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 100
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces for a mesh
    v, f = pcu.load_mesh_vf(path_to_file)

    # Sample points from the mesh
    pointcloud_orig = sample_mesh_random(v, f, num_samples=600)
    pointcloud_center = np.mean(pointcloud_orig, axis=0)
    pointcloud = center_pointcloud(pointcloud_orig)
    print(pointcloud.shape)
    max_radius = np.max(np.linalg.norm(pointcloud, axis=1))

    # Generate points around a sphere
    sphere_points = points_around_sphere(24, radius=max_radius + 0.5)
    draw_point_cloud(pointcloud, title="Point Cloud", overlay_pointcloud=sphere_points)

    # Calculate the rotation to align the camera with the focal point
    cameras = []
    for point in sphere_points:
        rotation = calculate_aligning_rotation(point)
        pose = transformation_matrix(rotation, point)
        cameras.append((k, [image_width, image_height], pose))

    draw_point_cloud_with_cameras(pointcloud, "Cameras", cameras=cameras)

    # Trying the render code
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRasterizer,
        BlendParams,
        SoftPhongShader,
        MeshRenderer,
    )
    from pytorch3d.renderer.mesh.rasterizer import Fragments

    # Mesh data
    mesh_vertices = v
    mesh_faces = f
    vertices = torch.tensor(mesh_vertices, dtype=torch.float32).unsqueeze(
        0
    )  # [1, N, 3]
    faces = torch.tensor(mesh_faces, dtype=torch.int64).unsqueeze(0)  # [1, M, 3]

    # Create the PyTorch3D Mesh object
    mesh = Meshes(verts=vertices, faces=faces)
    center = torch.mean(vertices, dim=1, keepdim=True)
    print("Center: ", center)
    print("pointcloud_center: ", pointcloud_center)

    # Camera intrinsics and extrinsics
    image_size = image_height  # Size of the rendered image
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32)
    principal_point = torch.tensor([[cx, cy]], dtype=torch.float32)

    point_modified = pointcloud_center + point
    pose = transformation_matrix(rotation, point_modified)
    camera = [(k, [image_width, image_height], pose)]
    print("PC CENter: ", np.mean(pointcloud_orig, axis=0))
    draw_point_cloud_with_cameras(
        pointcloud_orig,
        title="Point Cloud Original with Single Camera",
        cameras=camera,
    )

    rotation = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)
    point = torch.tensor(point_modified, dtype=torch.float32).unsqueeze(0)

    # Define the camera (batch size = 1 for single mesh)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rotation,  # [1, 3, 3]
        T=point,  # [1, 3]
    )

    # Define rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a rasterizer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Rasterize the mesh
    fragments = rasterizer(mesh)

    # Depth extraction from z-buffer
    depth_map = fragments.zbuf.squeeze()  # Shape: [H, W]

    # Normalize depth values for visualization (optional)
    depth_map_normalized = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min()
    )
    print("Depth Map Shape: ", depth_map_normalized.shape)
    print("Depth Map: ", depth_map_normalized)

    import matplotlib.pyplot as plt

    plt.imshow(depth_map_normalized.cpu().numpy(), cmap="gray")
    plt.colorbar()
    plt.show()
