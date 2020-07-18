import numpy as np


def histogram(catalog, column_name):
    """
    Makes a histogram of the column_name value from a catalog
    :param catalog: Catalog to use
    :param column_name: Name of the column to create a histogram of
    :return:
    """
    return np.histogram(catalog[column_name], bins=50, density=True)


def bbox_stats(image_paths):
    """
    Returns various stats about the bounding boxes in the images in the path
    For example, the number of bounding boxes per image,
    number of i band sources, number of W1 band sources
    :param image_paths: List of paths to images/cutout combined files
    :return: Dictionary containing the various statistics
    """
    num_bounding_boxes = []
    num_i_band_sources = []
    num_w_one_band_sources = []
    for image in image_paths:
        try:
            cutout, bboxes = np.load(image, allow_pickle=True)
        except:
            continue
        num_bounding_boxes.append(len(bboxes))
        num_i_band_sources.append(np.count_nonzero(cutout[:, :, 1]))
        num_w_one_band_sources.append(np.count_nonzero(cutout[:, :, 2]))
    return {
        "num_bbox": num_bounding_boxes,
        "num_i_band_sources": num_i_band_sources,
        "num_w1_band_sources": num_w_one_band_sources,
    }
