import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_cutout_and_bbox(image_name, title):
    """

    :param image_name:
    :param title:
    :return:
    """

    cutout_and_bboxes = np.load(image_name)
    cutout = cutout_and_bboxes[0]
    bboxes = cutout_and_bboxes[1]
    fig,ax = plt.subplots(1)
    if cutout.shape[2] == 3:
        ax.imshow(cutout)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()