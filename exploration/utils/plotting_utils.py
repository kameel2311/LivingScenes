import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def plot_gradual_metrics(title, x_axis, x_label, **metrics_kargs):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    colors = [
        hsv_to_rgb((i / len(metrics_kargs), 0.8, 0.8))
        for i in range(len(metrics_kargs))
    ]
    for i, (key, metrics) in enumerate(metrics_kargs.items()):
        if x_axis is None:
            x_values = np.arange(len(metrics))
        else:
            x_values = x_axis

        if x_label is None:
            x_label = "Added Object"

        ax[0].plot(
            x_values, [metric[0] for metric in metrics], color=colors[i], label=key
        )
        ax[0].set_title("Mean Absolute Distance")
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel("MAD")
        ax[1].plot(
            x_values, [metric[1] for metric in metrics], color=colors[i], label=key
        )
        ax[1].set_title("Pointcloud Coverage")
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel("Coverage")

    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    plt.suptitle(title)
    plt.show()


def plot_object_metrics(**data_dicts):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = [
        hsv_to_rgb((i / len(data_dicts), 0.8, 0.8)) for i in range(len(data_dicts))
    ]

    # Get all unique objects across all runs
    all_objects = sorted(
        set().union(*[data_dict.keys() for data_dict in data_dicts.values()])
    )

    # Common settings for both plots
    y_pos = np.arange(len(all_objects))
    bar_height = 0.8 / len(data_dicts)  # Divide available space by number of runs

    # Extract data
    for i, (key, data_dict) in enumerate(data_dicts.items()):
        # Create arrays with proper ordering and handle missing values
        mad_errors = []
        coverages = []
        for obj in all_objects:
            if obj in data_dict:
                mad_errors.append(data_dict[obj][0])
                coverages.append(data_dict[obj][1])
            else:
                mad_errors.append(0)  # or np.nan if you prefer gaps
                coverages.append(0)  # or np.nan if you prefer gaps

        # Calculate offset for this run's bars
        offset = bar_height * (i - (len(data_dicts) - 1) / 2)

        # Plot MAD errors
        ax1.barh(
            y_pos + offset,  # Offset each run's bars
            mad_errors,
            height=bar_height,
            color=colors[i],
            alpha=0.7,
            label=key,
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(all_objects)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel("MAD Error")
        ax1.set_title("Per Object MAD Error")
        ax1.grid(True)

        # Plot coverage
        ax2.barh(
            y_pos + offset,  # Offset each run's bars
            coverages,
            height=bar_height,
            color=colors[i],
            alpha=0.7,
            label=key,
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_objects)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel("Coverage")
        ax2.set_title("Per Object Coverage")
        ax2.grid(True)

    # Adjust layout and display
    ax1.legend()
    ax2.legend()
    plt.suptitle("Per Object Metrics")
    plt.tight_layout()
    plt.show()
