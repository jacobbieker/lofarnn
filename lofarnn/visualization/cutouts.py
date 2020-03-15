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
    fig,ax = plt.subplots(1)
    if cutout.shape[2] == 3:
        ax.imshow(cutout)

    for bbox in bboxes:
        rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), 1, 1, linewidth=1, edgecolor='w', facecolor='none')
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


