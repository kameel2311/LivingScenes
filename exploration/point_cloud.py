# Exploring Point Cloud Utils with ModelNet10
import point_cloud_utils as pcu
import os
import matplotlib.pyplot as plt
import numpy as np

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
    plt.title(title)
    plt.show()


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
    return pointcloud - np.mean(pointcloud, axis=0)


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