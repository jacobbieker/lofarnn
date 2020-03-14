import numpy as np


def histogram(catalog, column_name):
    """
    Makes a histogram of the column_name value from a catalog
    :param catalog: Catalog to use
    :param column_name: Name of the column to create a histogram of
    :return:
    """
    return np.histogram(catalog[column_name], bins=50, density=True)
