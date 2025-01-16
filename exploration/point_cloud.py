# Exploring Point Cloud Utils with ModelNet10
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import point_cloud_utils as pcu

DATA_DIR = "/Datasets/ModelNet10/ModelNet10"


def draw_point_cloud(pointcloud_array, title="", overlay_pointcloud=None):
    # Visualize the point cloud using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pointcloud_array[:, 0],
        pointcloud_array[:, 1],
        pointcloud_array[:, 2],
        s=3,
        c="blue",
        alpha=0.5,
    )

    if overlay_pointcloud is not None:
        ax.scatter(
            overlay_pointcloud[:, 0],
            overlay_pointcloud[:, 1],
            overlay_pointcloud[:, 2],
            s=3,
            c="red",
            alpha=0.5,
        )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.show()


def draw_camera(ax, K, image_size, pose, scale=1.0):
    """
    Draws a camera frustum in 3D using its intrinsics and pose.

    Parameters:
        ax (mpl_toolkits.mplot3d.Axes3D): The Matplotlib 3D axis to draw on.
        K (numpy.ndarray): 3x3 camera intrinsics matrix.
        pose (numpy.ndarray): 4x4 camera pose matrix w_T_c.
        scale (float): Scaling factor for the frustum size.
        color (str): Color of the frustum lines.
    """
    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    image_width, image_height = image_size

    # Image plane corners in normalized device coordinates (NDC)
    img_corners = np.array(
        [
            [-cx / fx, -cy / fy, 1],  # Bottom-left
            [(image_width - cx) / fx, -cy / fy, 1],  # Bottom-right
            [(image_width - cx) / fx, (image_height - cy) / fy, 1],  # Top-right
            [-cx / fx, (image_height - cy) / fy, 1],  # Top-left
        ]
    ).T

    # Scale the frustum size
    img_corners *= scale

    # Camera center in world frame
    cam_center = np.array([0, 0, 0, 1])

    # Transform frustum corners and camera center using the pose matrix
    world_corners = pose @ np.vstack((img_corners, np.ones((1, 4))))
    world_cam_center = pose @ cam_center

    # Extract points in 3D
    world_corners = world_corners[:3, :].T
    world_cam_center = world_cam_center[:3]

    colours = ["red", "green", "blue", "yellow"]

    # Draw frustum edges
    for i in range(4):
        ax.plot(
            [world_cam_center[0], world_corners[i, 0]],
            [world_cam_center[1], world_corners[i, 1]],
            [world_cam_center[2], world_corners[i, 2]],
            color=colours[i],
        )

    # Draw image plane edges
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [world_corners[i, 0], world_corners[j, 0]],
            [world_corners[i, 1], world_corners[j, 1]],
            [world_corners[i, 2], world_corners[j, 2]],
            color=colours[i],
        )


def draw_point_cloud_with_cameras(
    pointcloud_array, title="", overlay_pointcloud=None, cameras=None
):
    """
    Visualize the point cloud and optionally overlay another point cloud or cameras.

    Parameters:
        pointcloud_array (numpy.ndarray): Nx3 array of point cloud data.
        title (str): Title of the plot.
        overlay_pointcloud (numpy.ndarray): Nx3 array for an overlay point cloud.
        cameras (list): List of tuples (K, pose) for camera intrinsics and poses.
    """
    # Visualize the point cloud using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pointcloud_array[:, 0],
        pointcloud_array[:, 1],
        pointcloud_array[:, 2],
        s=3,
        c="blue",
        alpha=0.5,
    )

    if overlay_pointcloud is not None:
        ax.scatter(
            overlay_pointcloud[:, 0],
            overlay_pointcloud[:, 1],
            overlay_pointcloud[:, 2],
            s=3,
            c="red",
            alpha=0.5,
        )

    if cameras:
        for K, image_size, pose in cameras:
            draw_camera(ax, K, image_size, pose, scale=1)
            ax.scatter(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                s=10,
                c="red",
                alpha=0.5,
            )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.show()
    return fig, ax


def path_generator(data_dir, object_class, file, instance):
    """
    Generate the path to the object instance
    """
    instance_string = str(instance).zfill(4)
    path = os.path.join(
        data_dir, object_class, file, f"{object_class}_{instance_string}.off"
    )
    assert os.path.exists(path), f"Path does not exist: {path}"
    return path


def sample_mesh_random(v, f, multiplier=None, num_samples=None):
    """
    Sample points from a mesh randomly
    """
    if num_samples is None and multiplier is None:
        raise ValueError("Either num_samples or multiplier must be provided")
    if num_samples is not None and multiplier is not None:
        raise ValueError("Only one of num_samples or multiplier should be provided")
    if multiplier is not None:
        number = int(v.shape[0] * multiplier)
    elif num_samples is not None:
        number = num_samples

    f_i, bc = pcu.sample_mesh_random(v, f, num_samples=number)

    # Use the face indices and barycentric coordinate to compute sample positions and normals
    v_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

    return v_sampled


def add_gaussian_noise(pointcloud, sigma=0.01):
    """
    Add Gaussian noise to the point cloud
    """
    noise = np.random.normal(0, sigma, pointcloud.shape)
    return pointcloud + noise


def random_subsample_pointcloud(pointcloud, num_per_sample, overlap=0.5, center=False):
    """
    Subsample the point cloud into two samples with overlapping points
    """
    point_cloud_indicies = np.random.choice(
        pointcloud.shape[0], num_per_sample, replace=False
    )
    shared_points_indicies = np.random.choice(
        point_cloud_indicies, int(overlap * num_per_sample), replace=False
    )
    non_overlap_points_indicies = np.setdiff1d(
        list(range(pointcloud.shape[0])), point_cloud_indicies
    )
    sampled_non_overlap_indicies = np.random.choice(
        non_overlap_points_indicies, num_per_sample - shared_points_indicies.shape[0]
    )
    point_cloud_2_indicies = np.concatenate(
        (shared_points_indicies, sampled_non_overlap_indicies)
    )
    if center:
        return center_pointcloud(pointcloud[point_cloud_indicies]), center_pointcloud(
            pointcloud[point_cloud_2_indicies]
        )
    else:
        return pointcloud[point_cloud_indicies], pointcloud[point_cloud_2_indicies]


def get_angular_position(pointcloud):
    """
    Get the angular position of the point cloud
    """
    angular_position = np.rad2deg(np.arctan2(pointcloud[:, 1], pointcloud[:, 0]))
    return get_continous_angles(angular_position)


def get_continous_angles(angular_position):
    """
    Get the continous angles
    """
    return np.where(angular_position < 0, angular_position + 360, angular_position)


def viewpoint_subsample_pointcloud(pointcloud, first_limit, second_limit, center=False):
    """
    Subsample the point cloud based on the viewpoint
    """
    angular_position = get_angular_position(pointcloud)
    if first_limit > second_limit:
        mask = np.logical_or(
            np.logical_and(angular_position >= first_limit, angular_position <= 360),
            np.logical_and(angular_position >= 0, angular_position <= second_limit),
        )
    else:
        mask = np.logical_and(
            angular_position >= first_limit, angular_position <= second_limit
        )
    if center:
        return center_pointcloud(pointcloud[mask])
    else:
        return pointcloud[mask]


def center_pointcloud(pointcloud):
    """
    Center the point cloud at the origin
    """
    center = np.mean(pointcloud, axis=0)
    return pointcloud - center, center


def center_pointcloud_v2(pointcloud):
    """
    Center the point cloud at the min max points
    """
    minimium = np.min(pointcloud, axis=0)
    maximum = np.max(pointcloud, axis=0)
    average = (minimium + maximum) / 2
    return pointcloud - average, average


def generate_random_rotation(pure_z_rotation=False):
    """
    Generate a random 3D rotation matrix (uniformly sampled from SO(3)).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    if not pure_z_rotation:
        # Generate a random quaternion
        random_rotation = R.random()
        # Convert the quaternion to a rotation matrix
        rotation_matrix = random_rotation.as_matrix()
    else:
        # Generate a random angle
        angle = np.random.uniform(0, 2 * np.pi)
        # Generate a rotation matrix around the z-axis
        rotation_matrix = R.from_rotvec([0, 0, angle]).as_matrix()
    return rotation_matrix


def rotate_pointcloud_randomly(pointcloud, pure_z_rotation=False):
    """
    Rotate the point cloud using the rotation matrix
    """
    rotation_matrix = generate_random_rotation(pure_z_rotation)
    return (rotation_matrix @ pointcloud.T).T, rotation_matrix


if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 100
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces for a mesh
    v, f = pcu.load_mesh_vf(path_to_file)
    print(f"Vertices: {v.shape}, Faces: {f.shape}")

    # Sample points from the mesh
    v_sampled = sample_mesh_random(v, f, num_samples=600)

    # Draw the point cloud
    draw_point_cloud(v_sampled, title="Chair Point Cloud", overlay_pointcloud=v)

    # Add Gaussian noise to the point cloud
    v_noisy = add_gaussian_noise(v_sampled, sigma=0.5)
    draw_point_cloud(v_noisy, title="Noisy Chair Point Cloud", overlay_pointcloud=v)

    # Subsample the point cloud into two random samples with overlapping points
    v_sample_1, v_sample_2 = random_subsample_pointcloud(
        v_sampled, num_per_sample=400, overlap=0.5
    )
    draw_point_cloud(
        v_sample_1, title="Sampled Overlapping", overlay_pointcloud=v_sample_2
    )

    print("Angular Position of the Point Cloud")
    angular_position = get_angular_position(v_sampled)
    print(max(angular_position))
    print(min(angular_position))

    # Subsample the point cloud based on the viewpoint
    v_viewpoint_1 = viewpoint_subsample_pointcloud(v_sampled, 200, 360)
    v_viewpoint_2 = viewpoint_subsample_pointcloud(v_sampled, 300, 45)
    draw_point_cloud(
        v_viewpoint_1, title="Viewpoint Sub-sampling", overlay_pointcloud=v_viewpoint_2
    )

    # Random Point Cloud Rotations
    draw_point_cloud(rotate_pointcloud_randomly(v_sampled), title="Random Rotation")
