# Point Cloud Encoding Similairty Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import pandas as pd
from scipy.spatial import cKDTree


def angular_similarity(a, b):
    """
    Compute the angular similarity between two tensors
    """
    a = torch.flatten(a)
    b = torch.flatten(b)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def matrix_angular_similarity(a, b):
    """
    Compute the cross similarity Matrix between two tensors
    """
    similarity = a @ b.T
    # Compute norms for each row
    norm_a = torch.norm(a, dim=1, keepdim=True)  # Shape: (N, 1)
    norm_b = torch.norm(b, dim=1, keepdim=True)  # Shape: (M, 1)

    # Outer product of norms
    norm_matrix = norm_a @ norm_b.T  # Shape: (N, M)
    return similarity / norm_matrix


def matrix_fitness_metric(similarity_matrix, average_along_matrix=True):
    # Ensure the input is a tensor
    if isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = torch.tensor(similarity_matrix)
    elif not isinstance(similarity_matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

    num_elements = similarity_matrix.shape[0]
    off_diag_means = []
    off_diag_stds = []

    for i in range(num_elements):
        # Extract row and column
        extracted_row = similarity_matrix[i, :]
        extracted_col = similarity_matrix[:, i]

        # Ensure the diagonal element consistency
        diagonal_element = extracted_row[i]
        assert diagonal_element == extracted_col[i]

        # Remove diagonal element
        extracted_row = torch.cat((extracted_row[:i], extracted_row[i + 1 :]))
        extracted_col = torch.cat((extracted_col[:i], extracted_col[i + 1 :]))

        # Combine row and column values and take absolute values
        extracted_values = torch.cat((extracted_row, extracted_col)).abs()

        # Compute mean and std of extracted values
        mean_extracted_values = extracted_values.mean().item()
        std_extracted_values = (extracted_values - diagonal_element).abs().std().item()

        off_diag_means.append(mean_extracted_values)
        off_diag_stds.append(std_extracted_values)

    # Compute final metrics
    off_diag_mean = np.mean(off_diag_means)
    off_diag_std = np.mean(off_diag_stds)
    diag_mean = similarity_matrix.diag().mean().item()

    if average_along_matrix:
        return diag_mean, off_diag_mean, off_diag_std
    else:
        return (
            off_diag_means,
            off_diag_stds,
            similarity_matrix.diag(),
        )


def plot_data(
    dataset_diagonal_mean, dataset_off_diagonal_mean, dataset_off_diagonal_std
):
    """
    Plots histograms with density curves for diagonal mean, off-diagonal mean, and std.

    Args:
        dataset_diagonal_mean (list): List of diagonal means.
        dataset_off_diagonal_mean (list): List of off-diagonal means.
        dataset_off_diagonal_std (list): List of off-diagonal standard deviations.
    """
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Histogram with density curve for off-diagonal means
    sns.histplot(
        dataset_off_diagonal_mean,
        bins=10,
        kde=True,
        color="orange",
        label="Off-Diagonal Means",
        alpha=0.6,
        stat="density",
    )

    # Histogram with density curve for off-diagonal std
    sns.histplot(
        dataset_off_diagonal_std,
        bins=10,
        kde=True,
        color="green",
        label="Off-Diagonal Stds",
        alpha=0.6,
        stat="density",
    )

    # Histogram with density curve for diagonal means
    sns.histplot(
        dataset_diagonal_mean,
        bins=10,
        kde=True,
        color="blue",
        label="Diagonal Means",
        alpha=0.6,
        stat="density",
    )

    # Add labels and title
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Histogram and Density of Matrix Fitness Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_rre(rre, labels=None):
    """
    Plots historgram of the Rotation Error
    """
    n_bins = 20
    if labels is None:
        sns.histplot(rre, bins=n_bins, kde=True)
    else:
        multipler = len(rre) / len(labels)
        assert int(multipler) == multipler
        labels = labels * int(multipler)
        data = pd.DataFrame({"Rotation Error": rre, "Object Class": labels})
        sns.histplot(
            data=data, x="Rotation Error", hue="Object Class", kde=True, bins=n_bins
        )
    plt.title("Rotation Error")
    plt.xlabel("Rotation Error")
    plt.ylabel("Frequency")
    plt.show()


def plot_correlation(x, y, x_label, y_label, labels=None):
    # Example data
    data = {
        "x_values": x,
        "y_values": y,
        "labels": labels if labels is not None else "default",
    }

    # Create a scatter plot
    sns.scatterplot(
        x="x_values", y="y_values", hue="labels", data=data, palette="tab10"
    )

    # Add labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Scatter Plot of {x_label} vs {y_label}")

    if labels is not None:
        plt.legend(title="Labels", loc="best")
    else:
        plt.legend().set_visible(False)

    # Show the plot
    plt.show()


def compute_pointcloud_overlap(pc1, pc2, epsilon):
    """
    Compute the overlap between two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud, shape (n1, 3).
        pc2 (np.ndarray): Second point cloud, shape (n2, 3).
        epsilon (float): Distance threshold for overlap.

    Returns:
        float: Overlap ratio (number of overlapping points / total points in pc1).
    """
    # Build a KDTree for the second point cloud
    tree = cKDTree(pc2)

    # Query the KDTree with the first point cloud
    distances, _ = tree.query(pc1, k=1)  # k=1 finds the nearest neighbor

    # Count points in pc1 that have a neighbor within epsilon in pc2
    num_overlapping_points_1_in_2 = np.sum(distances <= epsilon)

    # Count points in pc2 that have a neighbor within epsilon in pc1
    tree = cKDTree(pc1)
    distances, _ = tree.query(pc2, k=1)
    num_overlapping_points_2_in_1 = np.sum(distances <= epsilon)

    # Calculate mean number of overlapping points
    mean_overlapping_points = (
        num_overlapping_points_1_in_2 + num_overlapping_points_2_in_1
    ) / 2

    # Calculate overlap ratio
    overlap_ratio = mean_overlapping_points / (
        len(pc1) + len(pc2) - mean_overlapping_points
    )
    return overlap_ratio
