# Point Cloud Encoding Similairty Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import pandas as pd
from scipy.spatial import cKDTree
import os


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
            similarity_matrix.diag(),
            off_diag_means,
            off_diag_stds,
        )


def plot_data(
    dataset_diagonal_mean,
    dataset_off_diagonal_mean,
    dataset_off_diagonal_std,
    class_title=None,
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
    if class_title is not None:
        plt.title(f"Histogram and Density of Matrix Fitness Metrics for {class_title}")
    else:
        plt.title("Histogram and Density of Matrix Fitness Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()


# TODO: Optimize plotting functions
def plot_dataset(
    dataset_diagonal_mean,
    dataset_off_diagonal_mean,
    dataset_off_diagonal_std,
    same_color=False,
):
    plt.figure(figsize=(10, 6))
    # Generate a color palette with enough unique colors for the number of keys
    keys = list(dataset_diagonal_mean.keys())
    palette = sns.color_palette("husl", len(keys))

    for i, key in enumerate(keys):
        if not same_color:  # Same for all classes
            color_diag = color_off_diag = palette[
                i
            ]  # Assign a unique color for each key
        else:
            color_off_diag = "orange"
            color_diag = "blue"

        # Histogram with density curve for off-diagonal means
        sns.histplot(
            dataset_off_diagonal_mean[key],
            bins=10,
            kde=True,
            color=color_off_diag,
            label=f"Off-Diagonal Means ({key})",
            alpha=0.6,
            stat="density",
            linestyle="dashed",  # Different line style to distinguish them
        )

        # Histogram with density curve for diagonal means
        sns.histplot(
            dataset_diagonal_mean[key],
            bins=10,
            kde=True,
            color=color_diag,
            label=f"Diagonal Means ({key})",
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


def generate_similarity_subplot(num_views, dataset, class_idx=None):
    fig, axes = plt.subplots(num_views, 1, figsize=(10, 3 * num_views))
    if num_views == 1:
        axes = [axes]

    classes = dataset["classes"]
    for i, view in enumerate(range(num_views)):
        ax = axes[i]
        diag_mean = dataset[f"view_{view}_diag_mean"]
        off_diag_mean = dataset[f"view_{view}_off_diag_mean"]
        off_diag_std = dataset[f"view_{view}_std_diag_mean"]
        overlap = dataset[f"view_{view}_overlap"]

        if class_idx is None:
            diag_mean_values = [item for sublist in diag_mean for item in sublist]
            off_diag_mean_values = [
                item for sublist in off_diag_mean for item in sublist
            ]
            off_diag_std_values = [item for sublist in off_diag_std for item in sublist]
            overlap_values = [item for sublist in overlap for item in sublist]
        else:
            diag_mean_values = [sublist[class_idx] for sublist in diag_mean]
            off_diag_mean_values = [sublist[class_idx] for sublist in off_diag_mean]
            off_diag_std_values = [sublist[class_idx] for sublist in off_diag_std]
            overlap_values = [sublist[class_idx] for sublist in overlap]

        sns.kdeplot(
            diag_mean_values,
            color="blue",
            linestyle="dashed",
            label=f"Diagonal Means (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )

        sns.kdeplot(
            off_diag_mean_values,
            color="orange",
            linestyle="dashed",
            label=f"Off-Diagonal Means (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )

        sns.kdeplot(
            overlap_values,
            color="green",
            linestyle="dashed",
            label=f"Overlap (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )

        ax.set_xlabel("Values")
        ax.set_ylabel("Density")
        if class_idx is not None:
            ax.set_title(
                f"Histogram and Density for View {view} (Class {classes[class_idx]})"
            )
        else:
            ax.set_title(f"Histogram and Density for View {view}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, 1)
        fig.subplots_adjust(hspace=0.7)
        plt.tight_layout()
    return fig, axes


def generate_similarity_correlation_subplot(num_views, dataset, class_idx=None):
    fig, axes = plt.subplots(num_views, 1, figsize=(10, 3 * num_views))
    if num_views == 1:
        axes = [axes]

    classes = dataset["classes"]
    for i, view in enumerate(range(num_views)):
        ax = axes[i]
        diag_mean = dataset[f"view_{view}_diag_mean"]
        overlap = dataset[f"view_{view}_overlap"]

        if class_idx is None:
            diag_mean_values = [item for sublist in diag_mean for item in sublist]
            overlap_values = [item for sublist in overlap for item in sublist]
        else:
            diag_mean_values = [sublist[class_idx] for sublist in diag_mean]
            overlap_values = [sublist[class_idx] for sublist in overlap]

        sns.scatterplot(
            x=overlap_values,
            y=diag_mean_values,
            color="blue",
            label=f"Overlap vs Diagonal Means (View {view})",
            alpha=0.6,
            ax=ax,
        )

        ax.set_xlabel("Overlap")
        ax.set_ylabel("Diagonal Means")
        if class_idx is not None:
            ax.set_title(
                f"Histogram and Density for View {view} (Class {classes[class_idx]})"
            )
        else:
            ax.set_title(f"Overlap vs Diagonal Means for View {view}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.subplots_adjust(hspace=0.7)
        plt.tight_layout()
    return fig, axes


def generate_classes_subplot(num_views, dataset):
    """Plots only the class overlap and diagonal mean values of the last view: logical in tracked scan tests"""
    if num_views == 1:
        axes = [axes]

    view = num_views - 1

    classes = dataset["classes"]
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 2 * len(classes)))
    for class_idx, class_name in enumerate(classes):
        ax = axes[class_idx]
        diag_mean = dataset[f"view_{view}_diag_mean"]
        off_diag_mean = dataset[f"view_{view}_off_diag_mean"]
        off_diag_std = dataset[f"view_{view}_std_diag_mean"]
        overlap = dataset[f"view_{view}_overlap"]

        diag_mean_values = [sublist[class_idx] for sublist in diag_mean]
        off_diag_mean_values = [sublist[class_idx] for sublist in off_diag_mean]
        overlap_values = [sublist[class_idx] for sublist in overlap]

        sns.kdeplot(
            diag_mean_values,
            color="blue",
            linestyle="dashed",
            label=f"Diagonal Means (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )

        sns.kdeplot(
            off_diag_mean_values,
            color="orange",
            linestyle="dashed",
            label=f"Off-Diagonal Means (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )

        sns.kdeplot(
            overlap_values,
            color="green",
            linestyle="dashed",
            label=f"Overlap (View {view})",
            alpha=0.6,
            ax=ax,
            fill=True,
        )
        ax.set_title(f"Class: {class_name}")
        ax.set_xlabel("Values")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, 1)
    fig.subplots_adjust(hspace=0.7)
    # fig.suptitle(f"Classes Metrics for Last View: Only for Tracked Test")

    plt.tight_layout()
    return fig, axes


def plot_similarity_subplots(
    dataset,
    num_views,
    plot_classes=False,
    save=False,
    save_dir=None,
    cls_subfolder=None,
):
    fig_sim, _ = generate_similarity_subplot(num_views, dataset)
    fig_corr, _ = generate_similarity_correlation_subplot(num_views, dataset)
    fig_classes, _ = generate_classes_subplot(num_views, dataset)

    # Per Dataset Plots
    if save:
        assert save_dir is not None, "Directory must be provided to save the plot"
        fig_sim.savefig(os.path.join(save_dir, "similarity_plot_database.png"))
        fig_corr.savefig(os.path.join(save_dir, "correlation_plot_database.png"))
        fig_classes.savefig(os.path.join(save_dir, "classes_plot_database.png"))
        plt.close(fig_sim)  # Close the figure after saving
        plt.close(fig_corr)
        plt.close(fig_classes)
    else:
        plt.show()

    # Plot for each class
    if plot_classes:
        if cls_subfolder is not None:
            save_dir = os.path.join(save_dir, cls_subfolder)
        classes = dataset["classes"]
        for class_idx, class_name in enumerate(classes):
            fig_sim_class, _ = generate_similarity_subplot(
                num_views, dataset, class_idx
            )
            fig_corr_class, _ = generate_similarity_correlation_subplot(
                num_views, dataset, class_idx
            )
            if save:
                fig_sim_class.savefig(
                    os.path.join(save_dir, f"similarity_subplot_{class_name}.png")
                )
                fig_corr_class.savefig(
                    os.path.join(save_dir, f"correlation_subplot_{class_name}.png")
                )
                plt.close(fig_sim_class)
                plt.close(fig_corr_class)
            else:
                plt.show()


def plot_rotational_subplots(dataset, num_views, save=False, save_dir=None):
    fig, axes = plt.subplots(num_views, 1, figsize=(10, 6 * num_views))
    if num_views == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    for i, view in enumerate(range(num_views)):
        ax = axes[i]
        rotational_errors = dataset[f"view_{view}_rotation_error"]

        sns.histplot(
            [item for sublist in rotational_errors for item in sublist],
            bins=10,
            kde=True,
            color="blue",
            label=f"Rotational Error (View {view})",
            alpha=0.6,
            stat="density",
            ax=ax,
        )

        ax.set_xlabel("Rotational Error")
        ax.set_ylabel("Density")
        ax.set_title(f"Histogram and Density for View {view}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        assert save_dir is not None, "Directory must be provided to save the plot"
        fig.savefig(os.path.join(save_dir, "rotation_error_plot_database.png"))
        plt.close(fig)  # Close the figure after saving
    else:
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


def mean_absolute_distance(gt_pc, recon_pc):
    """Computes the mean absolute distance of each point in the reconstructed
        point cloud to the nearest point in the gt point cloud.

    Args:
        gt_pc (np.ndarray): Ground truth pointcloud, shape (n1, 3).
        recon_pc (np.ndarray): Simulated Mesh pointcloud, shape (n2, 3).
    """
    # Build a KDTree for the ground truth point cloud
    tree = cKDTree(gt_pc)

    # Query the KDTree with the first point cloud
    distances, _ = tree.query(recon_pc, k=1)  # k=1 finds the nearest neighbor

    assert len(distances) == len(recon_pc)
    # Calculate the mean absolute distance
    mean_absolute_distance = np.mean(distances)
    return mean_absolute_distance


def pointcloud_coverage(gt_pc, recon_pc, epsilon):
    """Computes the perentage of points from gt_pc that are within an epsilon distance
       of the nearest point in the recon_pc.

    Args:
        gt_pc (np.ndarray): Ground truth pointcloud, shape (n1, 3).
        recon_pc (np.ndarray): Simulated Mesh pointcloud, shape (n2, 3).
        epsilon (float): Distance threshold for overlap.
    """
    # Build a KDTree for the reconstructed point cloud
    tree = cKDTree(recon_pc)

    # Query the KDTree with the ground truth point cloud
    distances, _ = tree.query(gt_pc, k=1)  # k=1 finds the nearest neighbor
    assert len(distances) == len(gt_pc)

    # Percentage of points in gt_pc that have a neighbor within epsilon in recon_pc
    return np.sum(distances <= epsilon) / len(gt_pc)
