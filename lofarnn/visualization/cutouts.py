import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from lofarnn.utils.stats import bbox_stats


def plot_cutout_and_bboxes(image_name, title):
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


def plot_statistics(image_paths, save_path):
    """
    Plots different statistics given from bbox_stats
    :param image_paths: List of paths to images
    :param save_path: Path where to save the plots
    :return:
    """

    stat_dict = bbox_stats(image_paths)

    plt.hist(stat_dict["num_bbox"], bins=50, density=False)
    plt.title("Number of Bounding Boxes per image")
    plt.xlabel("Numer of Bounding Boxes")
    plt.savefig(os.path.join(save_path, "num_bbox.png"))
    plt.close()
    plt.hist(stat_dict["num_i_band_sources"], bins=50, density=False)
    plt.title("Number of i band sources per image")
    plt.xlabel("Number of i band sources")
    plt.savefig(os.path.join(save_path, "num_i_band_sources.png"))
    plt.close()
    plt.hist(stat_dict["num_w1_band_sources"], bins=50, density=False)
    plt.title("Number of W1 band sources per image")
    plt.xlabel("Numer of W1 band sources")
    plt.savefig(os.path.join(save_path, "num_w1_band_sources.png"))
    plt.close()


def plot_three_channel_debug(
    image_stack, cutouts, scale_size, source_loc, save_path="./"
):
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
    # plt.savefig(save_path)
    # plt.close(fig)
    # plt.clf()
