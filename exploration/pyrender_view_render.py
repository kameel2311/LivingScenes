import numpy as np
import point_cloud_utils as pcu
from point_cloud import *
import pyrender
import trimesh

# scale = 1
# image_height = 100 * scale
# image_width = 100 * scale
# fx = 100 * scale
# fy = 100 * scale
# cx = 50 * scale
# cy = 50 * scale

image_height = 2000
image_width = 2000
fx = 500
fy = 500
cx = image_width / 2
cy = image_height / 2
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))


def points_around_sphere(num_points, radius=1):
    """Generate points around a sphere"""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack((x, y, z), axis=1)


def calculate_aligning_rotation(focalpoint):
    """Calculate rotation to align camera with focal point"""
    camera_center = np.array([image_width / 2, image_height / 2, 1])
    deproj_camera_center = np.linalg.inv(k) @ camera_center * 1
    rotation = find_rotation(deproj_camera_center, -1 * focalpoint)
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
    to OpenGL convention (-Z forward)
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


if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 100
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces
    v, f = pcu.load_mesh_vf(path_to_file)

    # Sample points from mesh
    pointcloud = sample_mesh_random(v, f, num_samples=600)
    # pointcloud_centered = center_pointcloud(pointcloud)
    pointcloud_centered, center = center_pointcloud_v2(pointcloud)
    max_radius = np.max(np.linalg.norm(pointcloud_centered, axis=1))

    # Generate camera positions
    sphere_points = points_around_sphere(24, radius=max_radius + 0.5)

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

        # Plot camera pose
        # draw_point_cloud_with_cameras(
        #     pointcloud, "Cameras", cameras=[(k, [image_width, image_height], pose)]
        # )

        # Render depth map
        depth = render_depth_map(v, f, pose)
        depth_maps.append(depth)

        print(f"Depth map shape: {depth.shape}")
        print(f"Depth range: {np.min(depth[depth >= 0])} to {np.max(depth)}")

    # Optional: visualize first depth map
    import matplotlib.pyplot as plt

    for depth_map in depth_maps:
        plt.imshow(depth_map)
        plt.colorbar()
        plt.title("Depth Map")
        plt.show()
