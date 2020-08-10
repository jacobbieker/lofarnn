import numpy as np
import pickle
import matplotlib.pyplot as plt
from lofarnn.models.dataloaders.utils import get_lotss_objects


def plot_axis_recall(recall_path, vac_catalog, bins=10):
    """
    Plot recall of apparent size to axis ratio
    :param recall_path: Recall path from SourceEvaluator, which has source_name and highest overlap of the limit
    :param vac_catalog: Value-added catalog location
    :param bins: Number of bins for the histogram
    :return:
    """
    data = pickle.load(open(recall_path, "rb"), fix_imports=True)
    vac_catalog = get_lotss_objects(vac_catalog)
    pred_source_names = []
    pred_source_recall = []
    for key in data.keys():
        pred_source_names.append(key)
        pred_source_recall.append(data[key])
    pred_source_recall = np.asarray(pred_source_recall)
    radio_apparent_size = np.zeros(len(pred_source_names))
    radio_apparent_width = np.zeros(len(pred_source_names))
    radio_total_flux = np.zeros(len(pred_source_names))
    radio_z = np.zeros(len(pred_source_names))
    for i, source_name in enumerate(pred_source_names):
        mask = source_name == vac_catalog['Source_Name']

        # get values
        radio_apparent_size[i] = vac_catalog[mask]['LGZ_Size'].data
        radio_apparent_width[i] = vac_catalog[mask]['LGZ_Width'].data
        radio_total_flux[i] = vac_catalog[mask]['Total_flux'].data
        radio_z[i] = vac_catalog[mask]['z_best'].data
    radio_axis_ratio = radio_apparent_size / radio_apparent_width
    data_dict = {'Axis ratio': radio_axis_ratio,
                 'Total flux [mJy]': radio_total_flux,
                 'Apparent size [arcsec]': radio_apparent_size,
                 'z': radio_z}

    ###calculate recall in bins
    # set which parameters you want to have on the X and Y axis
    xlabel = 'Apparent size [arcsec]'
    ylabel = 'Axis ratio'

    X = data_dict[xlabel]
    Y = data_dict[ylabel]
    # get edges with maxima determined using percentiles to be robust for outliers
    x_bin_edges = np.linspace(np.nanpercentile(X, 1), np.nanpercentile(X, 98), bins + 1)
    y_bin_edges = np.linspace(np.nanpercentile(Y, 1), np.nanpercentile(Y, 95), bins + 1)
    # derive bin centers
    x_bin_width = x_bin_edges[1] - x_bin_edges[0]
    x_bin_centers = x_bin_edges[1:] - x_bin_width / 2
    y_bin_width = y_bin_edges[1] - y_bin_edges[0]
    y_bin_centers = y_bin_edges[1:] - y_bin_width / 2

    recall_2D = np.zeros((x_bin_centers.shape[0], y_bin_centers.shape[0]))
    n_sources = np.zeros((x_bin_centers.shape[0], y_bin_centers.shape[0]), dtype=int)
    # now obtain recall
    for i in range(len(x_bin_centers)):
        for j in range(len(y_bin_centers)):
            # get the prediction errors in a bin
            bin_contents = pred_source_recall[
                (X > x_bin_edges[i]) & (X < x_bin_edges[i + 1]) & (Y > y_bin_edges[j]) & (Y < y_bin_edges[j + 1])]
            # determine recall
            recall_2D[i][j] = np.sum(bin_contents > 0.95) / len(bin_contents)
            # also determine the number of sources
            n_sources[i][j] = len(bin_contents)

    # now get the selection mask
    fig, ax = plt.subplots()

    # get the desired aspect ratio such that the plot is square
    aspectratio = (np.max(x_bin_centers) - np.min(x_bin_centers)) / (np.max(y_bin_centers) - np.min(y_bin_centers))
    im = ax.imshow(recall_2D.T, origin='lower', cmap='viridis',
                   vmin=0, vmax=1, zorder=2, aspect=aspectratio,
                   extent=[np.min(x_bin_edges), np.max(x_bin_edges), np.min(y_bin_edges), np.max(y_bin_edges)])

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # reset view limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # indicate the number of sources
    for i in range(n_sources.shape[0]):
        for j in range(n_sources.shape[1]):
            ax.text(x_bin_centers[i], y_bin_centers[j], str(n_sources[i, j]), ha='center', va='center', color='white',
                    fontsize=5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Recall')

    ax.set_xticks(x_bin_centers)
    ax.set_yticks(y_bin_centers)

    ax.tick_params(axis='x', labelrotation=40, labelsize='x-small')
    ax.tick_params(axis='y', labelrotation=0, labelsize='x-small')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(f'Recall for radio size vs axis ratio')
    plt.show()
    plt.close()