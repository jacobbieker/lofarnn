from typing import Tuple, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_cutout_and_bboxes(image_name: str, title: str) -> Tuple[plt.Figure, plt.Axes]:
    """

    :param image_name:
    :param title:
    :return:
    """
    print(image_name)
    cutout_and_bboxes = np.load(image_name, allow_pickle=True)
    cutout = cutout_and_bboxes[0]
    bboxes = cutout_and_bboxes[1]
    fig, ax = plt.subplots(1)
    if cutout.shape[2] == 3:
        ax.imshow(cutout)

    for bbox in bboxes:
        rect = patches.Rectangle(
            (int(bbox[0]), int(bbox[1])),
            1,
            1,
            linewidth=1,
            edgecolor="w",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.title(title)
    return fig, ax


def plot_three_channel_debug(
    image_stack: np.ndarray,
    cutouts: List[Tuple[float]],
    scale_size: float,
    source_loc: Tuple[float, float],
) -> None:
    """
    Plots the 3 channel image in debug
    """
    # Plot
    fig, axs = plt.subplots(2, 2)
    image_stack = np.asarray(image_stack)
    print(image_stack.shape)
    image_stack = image_stack[:, :, :3]  # Take first three

    cax_00 = axs[0, 0].imshow(image_stack)
    rect = patches.Rectangle(
        (float(cutouts[0][0]), float(cutouts[0][1])),
        float(cutouts[0][2]) - float(cutouts[0][0]),
        float(cutouts[0][3]) - float(cutouts[0][1]),
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0, 0].add_patch(rect)
    axs[0, 0].scatter(
        source_loc[1],
        source_loc[0],
        s=scale_size,
        edgecolor="black",
        facecolor=(1, 1, 1, 0.15),
    )
    axs[0, 0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
    axs[0, 0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels
    axs[0, 0].set_title("Combined")

    cax_01 = axs[0, 1].imshow(image_stack[:, :, 0], cmap="Reds")
    fig.colorbar(cax_01, ax=axs[0, 1])
    rect = patches.Rectangle(
        (float(cutouts[0][0]), float(cutouts[0][1])),
        float(cutouts[0][2]) - float(cutouts[0][0]),
        float(cutouts[0][3]) - float(cutouts[0][1]),
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0, 1].add_patch(rect)
    axs[0, 1].scatter(
        source_loc[1],
        source_loc[0],
        s=scale_size,
        edgecolor="black",
        facecolor=(1, 1, 1, 0.15),
    )
    axs[0, 1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0, 1].yaxis.set_major_formatter(plt.NullFormatter())
    axs[0, 1].set_title("LOFAR")

    cax_10 = axs[1, 0].imshow(image_stack[:, :, 1], cmap="Greens")
    fig.colorbar(cax_10, ax=axs[1, 0])
    rect = patches.Rectangle(
        (float(cutouts[0][0]), float(cutouts[0][1])),
        float(cutouts[0][2]) - float(cutouts[0][0]),
        float(cutouts[0][3]) - float(cutouts[0][1]),
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[1, 0].add_patch(rect)
    axs[1, 0].scatter(
        source_loc[1],
        source_loc[0],
        s=scale_size,
        edgecolor="black",
        facecolor=(1, 1, 1, 0.15),
    )
    axs[1, 0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 0].yaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 0].set_title("i Band")

    cax_11 = axs[1, 1].imshow(image_stack[:, :, 2], cmap="Blues")
    fig.colorbar(cax_11, ax=axs[1, 1])
    rect = patches.Rectangle(
        (float(cutouts[0][0]), float(cutouts[0][1])),
        float(cutouts[0][2]) - float(cutouts[0][0]),
        float(cutouts[0][3]) - float(cutouts[0][1]),
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[1, 1].add_patch(rect)
    axs[1, 1].scatter(
        source_loc[1],
        source_loc[0],
        s=scale_size,
        edgecolor="black",
        facecolor=(1, 1, 1, 0.15),
    )
    axs[1, 1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 1].yaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 1].set_title("W1 Band")
    plt.show()
