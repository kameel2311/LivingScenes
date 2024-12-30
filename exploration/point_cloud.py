# Exploring Point Cloud Utils with ModelNet10
import point_cloud_utils as pcu
import os
import matplotlib.pyplot as plt

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
        alpha=1.0,
    )

    if overlay_pointcloud is not None:
        ax.scatter(
            overlay_pointcloud[:, 0],
            overlay_pointcloud[:, 1],
            overlay_pointcloud[:, 2],
            s=3,
            c="red",
            alpha=1.0,
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
    v_sampled = sample_mesh_random(v, f, multiplier=5)

    # Draw the point cloud
    draw_point_cloud(v_sampled, title="Chair Point Cloud", overlay_pointcloud=v)
