import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from lofarnn.models.dataloaders.utils import get_lotss_objects


def load_json_arr(json_path):
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_cutoffs(
    recall_path,
    recall_path_2,
    baseline_path,
    vac_catalog,
    bins=30,
    name="",
    recall_names=["CNN", "Fast RCNN"],
):
    data = pickle.load(open(recall_path, "rb"), fix_imports=True)
    data2 = pickle.load(open(recall_path_2, "rb"), fix_imports=True)
    baseline_data = pickle.load(open(baseline_path, "rb"), fix_imports=True)
    vac_catalog = get_lotss_objects(vac_catalog)
    qual = vac_catalog["LGZ_ID_Qual"]
    pred_source_names = []
    pred_source_recall = []
    pred_source_names2 = []
    pred_source_recall2 = []
    baseline_names = []
    baseline_recalls = []
    # vac_catalog = vac_catalog[vac_catalog["LGZ_Size"] > 15.0]
    # vac_catalog = vac_catalog[vac_catalog["Total_flux"] > 10.0]
    for key in data.keys():
        if key in vac_catalog["Source_Name"].data:
            pred_source_names.append(key)
            pred_source_recall.append(data[key])
    for key in data2.keys():
        if key in vac_catalog["Source_Name"].data:
            pred_source_names2.append(key)
            pred_source_recall2.append(data2[key])
    for key in baseline_data.keys():
        if key in vac_catalog["Source_Name"].data:
            baseline_names.append(key)
            baseline_recalls.append(baseline_data[key])
    pred_source_recall = np.asarray(pred_source_recall)
    pred_source_recall2 = np.asarray(pred_source_recall2)
    baseline_recalls = np.asarray(baseline_recalls)  # [:1630]
    radio_apparent_size = np.zeros(len(pred_source_names))
    radio_apparent_width = np.zeros(len(pred_source_names))
    radio_total_flux = np.zeros(len(pred_source_names))
    radio_z = np.zeros(len(pred_source_names))
    radio_comp = np.zeros(len(pred_source_names))
    quality = np.zeros(len(pred_source_names))
    for i, source_name in enumerate(pred_source_names):
        mask = source_name == vac_catalog["Source_Name"]
        quality[i] = np.nan_to_num(vac_catalog[mask]["LGZ_ID_Qual"].data)
        # get values
        radio_apparent_size[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux[i] = vac_catalog[mask]["Total_flux"].data
        radio_z[i] = vac_catalog[mask]["z_best"].data
        radio_comp[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio = radio_apparent_size / radio_apparent_width
    data_dict = {
        "Axis ratio": radio_axis_ratio,
        "Total flux [mJy]": radio_total_flux,
        "Apparent size [arcsec]": radio_apparent_size,
        "z": radio_z,
        "Number of Components": radio_comp,
        "quality": quality,
    }
    radio_apparent_size3 = np.zeros(len(pred_source_names2))
    radio_apparent_width3 = np.zeros(len(pred_source_names2))
    radio_total_flux3 = np.zeros(len(pred_source_names2))
    radio_z3 = np.zeros(len(pred_source_names2))
    radio_comp3 = np.zeros(len(pred_source_names2))
    for i, source_name in enumerate(pred_source_names2):
        mask = source_name == vac_catalog["Source_Name"]

        # get values
        radio_apparent_size3[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width3[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux3[i] = vac_catalog[mask]["Total_flux"].data
        radio_z3[i] = vac_catalog[mask]["z_best"].data
        radio_comp3[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio3 = radio_apparent_size3 / radio_apparent_width3
    data_dict2 = {
        "Axis ratio": radio_axis_ratio3,
        "Total flux [mJy]": radio_total_flux3,
        "Apparent size [arcsec]": radio_apparent_size3,
        "z": radio_z3,
        "Number of Components": radio_comp3,
    }
    radio_apparent_size2 = np.zeros(len(baseline_names))
    radio_apparent_width2 = np.zeros(len(baseline_names))
    radio_total_flux2 = np.zeros(len(baseline_names))
    radio_z2 = np.zeros(len(baseline_names))
    radio_comp2 = np.zeros(len(baseline_names))
    for i, source_name in enumerate(baseline_names):  # [:1630]):
        mask = source_name == vac_catalog["Source_Name"]

        # get values
        radio_apparent_size2[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width2[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux2[i] = vac_catalog[mask]["Total_flux"].data
        radio_z2[i] = vac_catalog[mask]["z_best"].data
        radio_comp2[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio2 = radio_apparent_size2 / radio_apparent_width2
    baseline_data_dict = {
        "Axis ratio": radio_axis_ratio2,
        "Total flux [mJy]": radio_total_flux2,
        "Apparent size [arcsec]": radio_apparent_size2,
        "z": radio_z2,
        "Number of Components": radio_comp2,
    }
    pred_source_recall = np.array(pred_source_recall)
    pred_source_recall2 = np.array(pred_source_recall2)
    baseline_recalls = np.array(baseline_recalls)

    # Now plot recall cutoffs
    for ylabel in [
        "Apparent size [arcsec]",
        "Total flux [mJy]",
        "Axis ratio",
        "z",
        "Number of Components",
    ]:
        Y = data_dict[ylabel]
        Y2 = baseline_data_dict[ylabel]
        Y3 = data_dict2[ylabel]
        # x_bin_edges = np.linspace(np.nanmin(X)-0.00001, np.nanpercentile(X, 98), bins+1)
        y_bin_edges = np.linspace(
            np.nanpercentile(Y, 1) - 0.00001, np.nanpercentile(Y, 95), bins + 1
        )
        # y_bin_edges = np.linspace(np.nanmin(Y)-0.00001, np.nanpercentile(Y, 98), bins + 1)
        # derive bin centers
        y_bin_width = y_bin_edges[1] - y_bin_edges[0]
        y_bin_centers = y_bin_edges[1:] - y_bin_width / 2
        # now obtain recall
        recall = np.zeros((y_bin_centers.shape[0]))
        recall2 = np.zeros((y_bin_centers.shape[0]))
        recall3 = np.zeros((y_bin_centers.shape[0]))
        qualities = np.zeros((y_bin_centers.shape[0]))
        num_sources = np.zeros((y_bin_centers.shape[0]))
        for j in range(len(y_bin_centers)):
            # get the prediction errors in a bin
            bin_mask = (Y > y_bin_edges[j]) & (Y < y_bin_edges[j + 1])
            bin_contents = pred_source_recall[bin_mask]
            bin_quality = data_dict["quality"][bin_mask]
            bin_contents2 = baseline_recalls[
                (Y2 > y_bin_edges[j]) & (Y2 < y_bin_edges[j + 1])
            ]
            bin_contents3 = pred_source_recall2[
                (Y3 > y_bin_edges[j]) & (Y3 < y_bin_edges[j + 1])
            ]
            # determine recall
            # print(f"Bin Sum: {np.nansum(bin_contents > 0.95)}, len: {len(bin_contents)}")
            recall[j] = np.nansum(bin_contents > 0.95) / len(bin_contents)
            recall2[j] = np.nansum(bin_contents2 > 0.95) / len(bin_contents2)
            recall3[j] = np.nansum(bin_contents3 > 0.95) / len(bin_contents3)
            qualities[j] = np.nansum(bin_quality) / len(bin_quality)
            num_sources[j] = len(bin_contents)
        # Now plot
        fig, (ax3, ax1, ax2) = plt.subplots(
            3, 1, sharex="all", gridspec_kw={"height_ratios": [1, 3, 1], "hspace": 0}
        )
        # gs = fig.add_gridspec(3, hspace=0)
        # ax2, ax1, ax3 = gs.subplots(sharex=True, sharey=False)
        ax1.plot(y_bin_centers, recall, label=f"{recall_names[0]}")
        ax1.plot(y_bin_centers, recall3, label=f"{recall_names[1]}")
        ax1.plot(y_bin_centers, recall2, linestyle="--", label="Baseline")
        fig.suptitle(f"Recall vs {ylabel}")
        ax1.set_ylabel("Recall")
        ax1.set_xlabel(ylabel)
        # ax2 = ax1.twinx()
        ax2.set_xlabel(ylabel)
        ax2.bar(
            y_bin_centers,
            num_sources,
            width=(max(y_bin_centers) - min(y_bin_centers)) / len(y_bin_centers),
            edgecolor="black",
            color="none",
            zorder=10,
        )
        ax2.set_ylabel("# Sources")
        num_sources = num_sources / sum(num_sources)
        ax3.plot(y_bin_centers, np.cumsum(num_sources), c="black")
        ax3.set_ylabel("% Sources")
        ax3.set_xlabel(ylabel)
        if ylabel == "Apparent size [arcsec]":
            ax1.axvline(x=70.0, linestyle="dashed", color="black")
            ax2.axvline(x=70.0, linestyle="dashed", color="black")
            ax3.axvline(x=70.0, linestyle="dashed", color="black")
            mask = y_bin_centers <= 70
            s = (
                f"Baseline: {np.round(np.nanmean(recall2[mask]), 3)}\n"
                f"{recall_names[0]}: {np.round(np.nanmean(recall[mask]), 3)}\n"
                f"{recall_names[1]}: {np.round(np.nanmean(recall3[mask]), 3)}\n "
            )
            ax1.text(x=1, y=0.5, s=s, size=8)
            mask = y_bin_centers > 70
            s = (
                f"Baseline: {np.round(np.nanmean(recall2[mask]), 3)}\n"
                f"{recall_names[0]}: {np.round(np.nanmean(recall[mask]), 3)}\n"
                f"{recall_names[1]}: {np.round(np.nanmean(recall3[mask]), 3)}\n "
            )
            ax1.text(x=72, y=0.5, s=s, size=8)
        ax1.legend(loc="best")
        fig.savefig(f"{ylabel}_Recall_{name}.png", dpi=300)
        plt.clf()
        plt.cla()
        # Now read out recalls for apparent size
        if ylabel == "Apparent size [arcsec]":
            # Size cut
            mask = y_bin_centers <= 70
            print(f"Baseline Mean Recall <= 70 arcseconds: {np.nanmean(recall2[mask])}")
            print(f"CNN Mean Recall <= 70 arcseconds: {np.nanmean(recall[mask])}")
            print(f"FRCNN Mean Recall <= 70 arcseconds: {np.nanmean(recall3[mask])}")
            mask = y_bin_centers > 70
            print(f"Baseline Mean Recall > 70 arcseconds: {np.mean(recall2[mask])}")
            print(f"CNN Mean Recall > 70 arcseconds: {np.mean(recall[mask])}")
            print(f"FRCNN Mean Recall > 70 arcseconds: {np.mean(recall3[mask])}")


def plot_compared_axis_recall(
    recall_path,
    recall_path_2,
    vac_catalog,
    limit,
    jelle_cut=False,
    bins=10,
    output_dir="./",
):
    """
    Plot recall of apparent size to axis ratio
    :param recall_path: Recall path from SourceEvaluator, which has source_name and highest overlap of the limit
    :param vac_catalog: Value-added catalog location
    :param bins: Number of bins for the histogram
    :param limit: str, limit for the recall value for use in saving, title, etc.
    :return:
    """
    data = pickle.load(open(recall_path, "rb"), fix_imports=True)
    data2 = pickle.load(open(recall_path_2, "rb"), fix_imports=True)
    vac_catalog = get_lotss_objects(vac_catalog)
    pred_source_names = []
    pred_source_recall = []
    pred_source_names2 = []
    pred_source_recall2 = []
    if jelle_cut:
        vac_catalog = vac_catalog[vac_catalog["LGZ_Size"] > 15.0]
        vac_catalog = vac_catalog[vac_catalog["Total_flux"] > 10.0]
    for key in data.keys():
        if key in vac_catalog["Source_Name"].data:
            pred_source_names.append(key)
            pred_source_recall.append(data[key])
    for key in data2.keys():
        if key in vac_catalog["Source_Name"].data:
            pred_source_names2.append(key)
            pred_source_recall2.append(data2[key])
    pred_source_recall = np.asarray(pred_source_recall)
    pred_source_recall2 = np.asarray(pred_source_recall2)  # [:1630]
    radio_apparent_size = np.zeros(len(pred_source_names))
    radio_apparent_width = np.zeros(len(pred_source_names))
    radio_total_flux = np.zeros(len(pred_source_names))
    radio_z = np.zeros(len(pred_source_names))
    radio_comp = np.zeros(len(pred_source_names))
    for i, source_name in enumerate(pred_source_names):
        mask = source_name == vac_catalog["Source_Name"]

        # get values
        radio_apparent_size[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux[i] = vac_catalog[mask]["Total_flux"].data
        radio_z[i] = vac_catalog[mask]["z_best"].data
        radio_comp[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio = radio_apparent_size / radio_apparent_width
    data_dict = {
        "Axis ratio": radio_axis_ratio,
        "Total flux [mJy]": radio_total_flux,
        "Apparent size [arcsec]": radio_apparent_size,
        "z": radio_z,
        "Number of Components": radio_comp,
    }
    radio_apparent_size2 = np.zeros(len(pred_source_names2))
    radio_apparent_width2 = np.zeros(len(pred_source_names2))
    radio_total_flux2 = np.zeros(len(pred_source_names2))
    radio_z2 = np.zeros(len(pred_source_names2))
    radio_comp2 = np.zeros(len(pred_source_names2))
    for i, source_name in enumerate(pred_source_names2):  # [:1630]):
        mask = source_name == vac_catalog["Source_Name"]

        # get values
        radio_apparent_size2[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width2[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux2[i] = vac_catalog[mask]["Total_flux"].data
        radio_z2[i] = vac_catalog[mask]["z_best"].data
        radio_comp2[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio2 = radio_apparent_size2 / radio_apparent_width2
    data_dict2 = {
        "Axis ratio": radio_axis_ratio2,
        "Total flux [mJy]": radio_total_flux2,
        "Apparent size [arcsec]": radio_apparent_size2,
        "z": radio_z2,
        "Number of Components": radio_comp2,
    }

    ###calculate recall in bins
    # set which parameters you want to have on the X and Y axis
    for xlabel in ["Apparent size [arcsec]"]:
        for ylabel in ["Total flux [mJy]", "Axis ratio", "z", "Number of Components"]:
            X = data_dict[xlabel]
            Y = data_dict[ylabel]
            X2 = data_dict2[xlabel]  # [:1595]
            Y2 = data_dict2[ylabel]  # [:1595]
            # get edges with maxima determined using percentiles to be robust for outliers
            x_bin_edges = np.linspace(
                np.nanpercentile(X, 1) - 0.00001, np.nanpercentile(X, 98), bins + 1
            )
            # x_bin_edges = np.linspace(np.nanmin(X)-0.00001, np.nanpercentile(X, 98), bins+1)
            y_bin_edges = np.linspace(
                np.nanpercentile(Y, 1) - 0.00001, np.nanpercentile(Y, 95), bins + 1
            )
            # y_bin_edges = np.linspace(np.nanmin(Y)-0.00001, np.nanpercentile(Y, 98), bins + 1)
            # derive bin centers
            x_bin_width = x_bin_edges[1] - x_bin_edges[0]
            x_bin_centers = x_bin_edges[1:] - x_bin_width / 2
            y_bin_width = y_bin_edges[1] - y_bin_edges[0]
            y_bin_centers = y_bin_edges[1:] - y_bin_width / 2

            recall_2D = np.zeros((x_bin_centers.shape[0], y_bin_centers.shape[0]))
            recall_2D2 = np.zeros((x_bin_centers.shape[0], y_bin_centers.shape[0]))
            n_sources = np.zeros(
                (x_bin_centers.shape[0], y_bin_centers.shape[0]), dtype=int
            )
            # now obtain recall
            for i in range(len(x_bin_centers)):
                for j in range(len(y_bin_centers)):
                    # get the prediction errors in a bin
                    bin_contents = pred_source_recall[
                        (X > x_bin_edges[i])
                        & (X < x_bin_edges[i + 1])
                        & (Y > y_bin_edges[j])
                        & (Y < y_bin_edges[j + 1])
                    ]
                    bin_2 = pred_source_recall2[
                        (X2 > x_bin_edges[i])
                        & (X2 < x_bin_edges[i + 1])
                        & (Y2 > y_bin_edges[j])
                        & (Y2 < y_bin_edges[j + 1])
                    ]

                    # determine recall
                    # print(f"Bin Sum: {np.nansum(bin_contents > 0.95)}, len: {len(bin_contents)}")
                    recall_2D[i][j] = np.nansum(bin_contents > 0.95) / len(bin_contents)
                    recall_2D2[i][j] = np.nansum(bin_2 > 0.95) / len(bin_2)
                    # print(f"Recall Model:\n {recall_2D}")
                    # print(f"Recall Baseline:\n {recall_2D2}")
                    # also determine the number of sources
                    n_sources[i][j] = len(bin_contents)
            recall_2D = recall_2D - recall_2D2

            # now get the selection mask
            fig, ax = plt.subplots()

            # get the desired aspect ratio such that the plot is square
            aspectratio = (np.max(x_bin_centers) - np.min(x_bin_centers)) / (
                np.max(y_bin_centers) - np.min(y_bin_centers)
            )
            im = ax.imshow(
                recall_2D.T,
                origin="lower",
                cmap="bwr_r",
                vmin=-1,
                vmax=1,
                zorder=2,
                aspect=aspectratio,
                extent=[
                    np.min(x_bin_edges),
                    np.max(x_bin_edges),
                    np.min(y_bin_edges),
                    np.max(y_bin_edges),
                ],
            )

            xlims = ax.get_xlim()
            ylims = ax.get_ylim()

            # reset view limits
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            # indicate the number of sources
            for i in range(n_sources.shape[0]):
                for j in range(n_sources.shape[1]):
                    ax.text(
                        x_bin_centers[i],
                        y_bin_centers[j],
                        str(n_sources[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=5,
                    )

            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Recall Improvement Over Baseline")

            ax.set_xticks(x_bin_centers)
            ax.set_yticks(y_bin_centers)

            ax.tick_params(axis="x", labelrotation=40, labelsize="x-small")
            ax.tick_params(axis="y", labelrotation=0, labelsize="x-small")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_title(f"{limit} recall vs Baseline for {xlabel} vs {ylabel}")
            fig.savefig(
                os.path.join(
                    output_dir, f"{xlabel}-{ylabel}_jelle{jelle_cut}_{limit}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_axis_recall(
    recall_path, vac_catalog, limit, jelle_cut=False, bins=10, output_dir="./"
):
    """
    Plot recall of apparent size to axis ratio
    :param recall_path: Recall path from SourceEvaluator, which has source_name and highest overlap of the limit
    :param vac_catalog: Value-added catalog location
    :param bins: Number of bins for the histogram
    :param limit: str, limit for the recall value for use in saving, title, etc.
    :return:
    """
    data = pickle.load(open(recall_path, "rb"), fix_imports=True)
    vac_catalog = get_lotss_objects(vac_catalog)
    pred_source_names = []
    pred_source_recall = []
    if jelle_cut:
        vac_catalog = vac_catalog[vac_catalog["LGZ_Size"] > 15.0]
        vac_catalog = vac_catalog[vac_catalog["Total_flux"] > 10.0]
    for key in data.keys():
        if key in vac_catalog["Source_Name"].data:
            pred_source_names.append(key)
            pred_source_recall.append(data[key])
    pred_source_recall = np.asarray(pred_source_recall)
    radio_apparent_size = np.zeros(len(pred_source_names))
    radio_apparent_width = np.zeros(len(pred_source_names))
    radio_total_flux = np.zeros(len(pred_source_names))
    radio_z = np.zeros(len(pred_source_names))
    radio_comp = np.zeros(len(pred_source_names))
    for i, source_name in enumerate(pred_source_names):
        mask = source_name == vac_catalog["Source_Name"]

        # get values
        radio_apparent_size[i] = vac_catalog[mask]["LGZ_Size"].data
        radio_apparent_width[i] = vac_catalog[mask]["LGZ_Width"].data
        radio_total_flux[i] = vac_catalog[mask]["Total_flux"].data
        radio_z[i] = vac_catalog[mask]["z_best"].data
        radio_comp[i] = vac_catalog[mask]["LGZ_Assoc"].data
    radio_axis_ratio = radio_apparent_size / radio_apparent_width
    data_dict = {
        "Axis ratio": radio_axis_ratio,
        "Total flux [mJy]": radio_total_flux,
        "Apparent size [arcsec]": radio_apparent_size,
        "z": radio_z,
        "Number of Components": radio_comp,
    }

    ###calculate recall in bins
    # set which parameters you want to have on the X and Y axis
    for xlabel in ["Apparent size [arcsec]"]:
        for ylabel in ["Total flux [mJy]", "Axis ratio", "z", "Number of Components"]:
            X = data_dict[xlabel]
            Y = data_dict[ylabel]
            # get edges with maxima determined using percentiles to be robust for outliers
            x_bin_edges = np.linspace(
                np.nanpercentile(X, 1) - 0.00001, np.nanpercentile(X, 98), bins + 1
            )
            # x_bin_edges = np.linspace(np.nanmin(X)-0.00001, np.nanpercentile(X, 98), bins+1)
            y_bin_edges = np.linspace(
                np.nanpercentile(Y, 1) - 0.00001, np.nanpercentile(Y, 95), bins + 1
            )
            # y_bin_edges = np.linspace(np.nanmin(Y)-0.00001, np.nanpercentile(Y, 98), bins + 1)
            # derive bin centers
            x_bin_width = x_bin_edges[1] - x_bin_edges[0]
            x_bin_centers = x_bin_edges[1:] - x_bin_width / 2
            y_bin_width = y_bin_edges[1] - y_bin_edges[0]
            y_bin_centers = y_bin_edges[1:] - y_bin_width / 2

            recall_2D = np.zeros((x_bin_centers.shape[0], y_bin_centers.shape[0]))
            n_sources = np.zeros(
                (x_bin_centers.shape[0], y_bin_centers.shape[0]), dtype=int
            )
            # now obtain recall
            for i in range(len(x_bin_centers)):
                for j in range(len(y_bin_centers)):
                    # get the prediction errors in a bin
                    bin_contents = pred_source_recall[
                        (X > x_bin_edges[i])
                        & (X < x_bin_edges[i + 1])
                        & (Y > y_bin_edges[j])
                        & (Y < y_bin_edges[j + 1])
                    ]
                    # determine recall
                    recall_2D[i][j] = np.sum(bin_contents > 0.95) / len(bin_contents)
                    # also determine the number of sources
                    n_sources[i][j] = len(bin_contents)

            # now get the selection mask
            fig, ax = plt.subplots()

            # get the desired aspect ratio such that the plot is square
            aspectratio = (np.max(x_bin_centers) - np.min(x_bin_centers)) / (
                np.max(y_bin_centers) - np.min(y_bin_centers)
            )
            im = ax.imshow(
                recall_2D.T,
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=1,
                zorder=2,
                aspect=aspectratio,
                extent=[
                    np.min(x_bin_edges),
                    np.max(x_bin_edges),
                    np.min(y_bin_edges),
                    np.max(y_bin_edges),
                ],
            )

            xlims = ax.get_xlim()
            ylims = ax.get_ylim()

            # reset view limits
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            # indicate the number of sources
            for i in range(n_sources.shape[0]):
                for j in range(n_sources.shape[1]):
                    ax.text(
                        x_bin_centers[i],
                        y_bin_centers[j],
                        str(n_sources[i, j]),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=5,
                    )

            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Recall")

            ax.set_xticks(x_bin_centers)
            ax.set_yticks(y_bin_centers)

            ax.tick_params(axis="x", labelrotation=40, labelsize="x-small")
            ax.tick_params(axis="y", labelrotation=0, labelsize="x-small")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_title(f"Recall for {xlabel} vs {ylabel}, limit: {limit}")
            fig.savefig(
                os.path.join(
                    output_dir, f"{xlabel}-{ylabel}_limit{limit}_jelle{jelle_cut}.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()


def plot_plots(
    metrics_files,
    experiment_name,
    experiment_dir,
    cuts,
    labels,
    title,
    output_dir,
    colors,
):
    """
    Plot a variety of different metrics, including recall, precision, loss, etc.
    :param metrics_files:
    :param experiment_name:
    :param labels:
    :param output_dir:
    :return:
    """
    metrics_data = []
    for f in metrics_files:
        metrics_data.append(load_json_arr(os.path.join(f)))

    # Now extract the useful information: own recall and precision are the important ones since COCO seems off
    recall = {}
    val_recall = {}
    precision = {}
    val_precision = {}
    iteration = {}
    loss = {}
    val_loss = {}
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"] = []
                precision[
                    f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                ] = []
                val_precision[
                    f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                ] = []
                recall[f"{experiment_name}_train_test/own_recall_{j}/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_{j}/recall"] = []
                precision[f"{experiment_name}_train_test/own_recall_{j}/precision"] = []
                val_precision[f"{experiment_name}_val/own_recall_{j}/precision"] = []

    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                iteration[metrics_files[i]] = []
                loss[metrics_files[i]] = []
                val_loss[metrics_files[i]] = []
                val_recall[f"{experiment_name}_val/own_recall_1/recall"] = []
                val_precision[f"{experiment_name}_val/own_recall_1/precision"] = []
                precision[f"{experiment_name}_train_test/own_recall_1/precision"] = []
                recall[f"{experiment_name}_train_test/own_recall_1/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_1_{cut}/recall"] = []
                val_precision[
                    f"{experiment_name}_val/own_recall_1_{cut}/precision"
                ] = []
                precision[
                    f"{experiment_name}_train_test/own_recall_1_{cut}/precision"
                ] = []
                # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                recall[f"{experiment_name}_train_test/own_recall_1_{cut}/recall"] = []
                for line in metric:
                    try:
                        val_recall[
                            f"{experiment_name}_val/own_recall_{j}_{cut}/recall"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"]
                        )
                        val_precision[
                            f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                            ]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                            ]
                        )
                        recall[
                            f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"
                            ]
                        )
                        val_recall[
                            f"{experiment_name}_val/own_recall_{j}/recall"
                        ].append(line[f"{experiment_name}_val/own_recall_{j}/recall"])
                        val_precision[
                            f"{experiment_name}_val/own_recall_{j}/precision"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{j}/precision"]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_{j}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}/precision"
                            ]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_{j}/recall"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall_{j}/recall"]
                        )
                        val_recall[f"{experiment_name}_val/own_recall_1/recall"].append(
                            line[f"{experiment_name}_val/own_recall/recall"]
                        )
                        val_precision[
                            f"{experiment_name}_val/own_recall_1/precision"
                        ].append(line[f"{experiment_name}_val/own_recall/precision"])
                        precision[
                            f"{experiment_name}_train_test/own_recall_1/precision"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall/precision"]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_1/recall"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall/recall"]
                        )
                        val_recall[
                            f"{experiment_name}_val/own_recall_1_{cut}/recall"
                        ].append(line[f"{experiment_name}_val/own_recall_{cut}/recall"])
                        val_precision[
                            f"{experiment_name}_val/own_recall_1_{cut}/precision"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{cut}/precision"]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_1_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{cut}/precision"
                            ]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_1_{cut}/recall"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{cut}/recall"
                            ]
                        )
                        val_loss[metrics_files[i]].append(line["validation_loss"])
                        loss[metrics_files[i]].append(line["total_loss"])
                        iteration[metrics_files[i]].append(line["iteration"])
                    except:
                        continue

    # Plot the iteration vs loss for the models
    for i, metric in enumerate(metrics_data):
        if "1.0" in labels[i]:
            print(len(iteration[metrics_files[i]]))
        else:
            print(len(iteration[metrics_files[i]]))
        iteration[metrics_files[i]] = iteration[metrics_files[i]][:35]
        loss[metrics_files[i]] = loss[metrics_files[i]][:35]
        val_loss[metrics_files[i]] = val_loss[metrics_files[i]][:35]
        plt.plot(
            iteration[metrics_files[i]],
            loss[metrics_files[i]],
            label=f"{labels[i]} Train",
            color=colors[i],
        )
        plt.plot(
            iteration[metrics_files[i]],
            val_loss[metrics_files[i]],
            linestyle="dashed",
            label=f"{labels[i]} Val",
            color=colors[i],
        )

    plt.legend(loc="lower left")
    plt.title(f"Total Training Loss {title}")
    # plt.xlim(0,200000)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, f"Total_Training_Loss_{title}.png"), dpi=300)
    plt.clf()
    plt.cla()

    # Plot recall for the different cuts for the same models
    for i, metric in enumerate(metrics_data):
        for j in [1, 2, 5, 10, 100]:
            for k, cut in enumerate(cuts):
                plt.plot(
                    iteration[metrics_files[i]],
                    recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"],
                    label=f"{labels[k]} Train",
                    color=colors[k],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"],
                    label=f"{labels[k]} Val",
                    linestyle="dashed",
                    color=colors[k],
                )

            plt.legend(loc="lower right")
            plt.title(f"Recall for limit {j}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Recall_limit{j}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot precision for different cuts for same models
    for i, metric in enumerate(metrics_data):
        for j in [1, 2, 5, 10, 100]:
            for k, cut in enumerate(cuts):
                plt.plot(
                    iteration[metrics_files[i]],
                    precision[
                        f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{cut} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_precision[
                        f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{cut} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Precision for limit {j}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Precision")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Precision_limit{j}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot recall and precision based on different limits
    # Plot recall for the different cuts for the same models
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                plt.plot(
                    iteration[metrics_files[i]],
                    recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"],
                    label=f"{j} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"],
                    label=f"{j} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Recall for cut {cut}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Recall_cut{cut}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot precision for different cuts for same models
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                plt.plot(
                    iteration[metrics_files[i]],
                    precision[
                        f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{j} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_precision[
                        f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{j} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Precision for cut {cut}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Precision")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Precision_cut{cut}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot recall for different models

    # Plot precision for different models


def plot_combo_plots(
    metrics_files,
    experiment_name,
    experiment_dir,
    cuts,
    labels,
    title,
    output_dir,
    colors,
):
    """
        Plot a variety of different metrics, including recall, precision, loss, etc.
        :param metrics_files:
        :param experiment_name:
        :param labels:
        :param output_dir:
        :return:
        """
    metrics_data = []
    for f in metrics_files:
        metrics_data.append(load_json_arr(os.path.join(experiment_dir, f + ".json")))

    # Now extract the useful information: own recall and precision are the important ones since COCO seems off
    recall = {}
    val_recall = {}
    precision = {}
    val_precision = {}
    iteration = {}
    loss = {}
    val_loss = {}
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"] = []
                precision[
                    f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                ] = []
                val_precision[
                    f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                ] = []
                recall[f"{experiment_name}_train_test/own_recall_{j}/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_{j}/recall"] = []
                precision[f"{experiment_name}_train_test/own_recall_{j}/precision"] = []
                val_precision[f"{experiment_name}_val/own_recall_{j}/precision"] = []

    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [2, 5, 10, 100]:
                iteration[metrics_files[i]] = []
                loss[metrics_files[i]] = []
                val_loss[metrics_files[i]] = []
                val_recall[f"{experiment_name}_val/own_recall_1/recall"] = []
                val_precision[f"{experiment_name}_val/own_recall_1/precision"] = []
                precision[f"{experiment_name}_train_test/own_recall_1/precision"] = []
                recall[f"{experiment_name}_train_test/own_recall_1/recall"] = []
                val_recall[f"{experiment_name}_val/own_recall_1_{cut}/recall"] = []
                val_precision[
                    f"{experiment_name}_val/own_recall_1_{cut}/precision"
                ] = []
                precision[
                    f"{experiment_name}_train_test/own_recall_1_{cut}/precision"
                ] = []
                # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                recall[f"{experiment_name}_train_test/own_recall_1_{cut}/recall"] = []
                for line in metric:
                    try:
                        val_recall[
                            f"{experiment_name}_val/own_recall_{j}_{cut}/recall"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"]
                        )
                        val_precision[
                            f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                            ]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                            ]
                        )
                        recall[
                            f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"
                            ]
                        )
                        val_recall[
                            f"{experiment_name}_val/own_recall_{j}/recall"
                        ].append(line[f"{experiment_name}_val/own_recall_{j}/recall"])
                        val_precision[
                            f"{experiment_name}_val/own_recall_{j}/precision"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{j}/precision"]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_{j}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{j}/precision"
                            ]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_{j}/recall"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall_{j}/recall"]
                        )
                        val_recall[f"{experiment_name}_val/own_recall_1/recall"].append(
                            line[f"{experiment_name}_val/own_recall/recall"]
                        )
                        val_precision[
                            f"{experiment_name}_val/own_recall_1/precision"
                        ].append(line[f"{experiment_name}_val/own_recall/precision"])
                        precision[
                            f"{experiment_name}_train_test/own_recall_1/precision"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall/precision"]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_1/recall"
                        ].append(
                            line[f"{experiment_name}_train_test/own_recall/recall"]
                        )
                        val_recall[
                            f"{experiment_name}_val/own_recall_1_{cut}/recall"
                        ].append(line[f"{experiment_name}_val/own_recall_{cut}/recall"])
                        val_precision[
                            f"{experiment_name}_val/own_recall_1_{cut}/precision"
                        ].append(
                            line[f"{experiment_name}_val/own_recall_{cut}/precision"]
                        )
                        precision[
                            f"{experiment_name}_train_test/own_recall_1_{cut}/precision"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{cut}/precision"
                            ]
                        )
                        # print(line[f"{experiment_name}_train_test/own_recall_{j}/recall"])
                        recall[
                            f"{experiment_name}_train_test/own_recall_1_{cut}/recall"
                        ].append(
                            line[
                                f"{experiment_name}_train_test/own_recall_{cut}/recall"
                            ]
                        )
                        val_loss[metrics_files[i]].append(line["validation_loss"])
                        loss[metrics_files[i]].append(line["total_loss"])
                        iteration[metrics_files[i]].append(line["iteration"])
                    except:
                        continue

    # Plot the iteration vs loss for the models
    for i, metric in enumerate(metrics_data):
        plt.plot(
            iteration[metrics_files[i]],
            loss[metrics_files[i]],
            label=f"{labels[i]} Train",
            color=colors[i],
        )
        plt.plot(
            iteration[metrics_files[i]],
            val_loss[metrics_files[i]],
            linestyle="dashed",
            label=f"{labels[i]} Val",
            color=colors[i],
        )

    plt.legend(loc="upper right")
    plt.title(f"Total Training Loss {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, f"Total_Training_Loss_{title}.png"), dpi=300)
    plt.clf()
    plt.cla()

    # Plot recall for the different cuts for the same models
    for i, metric in enumerate(metrics_data):
        for j in [1, 2, 5, 10, 100]:
            for k, cut in enumerate(cuts):
                plt.plot(
                    iteration[metrics_files[i]],
                    recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"],
                    label=f"{labels[k]} Train",
                    color=colors[k],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"],
                    label=f"{labels[k]} Val",
                    linestyle="dashed",
                    color=colors[k],
                )

            plt.legend(loc="lower right")
            plt.title(f"Recall for limit {j}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Recall_limit{j}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot precision for different cuts for same models
    for i, metric in enumerate(metrics_data):
        for j in [1, 2, 5, 10, 100]:
            for k, cut in enumerate(cuts):
                plt.plot(
                    iteration[metrics_files[i]],
                    precision[
                        f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{cut} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_precision[
                        f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{cut} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Precision for limit {j}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Precision")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Precision_limit{j}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot recall and precision based on different limits
    # Plot recall for the different cuts for the same models
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                plt.plot(
                    iteration[metrics_files[i]],
                    recall[f"{experiment_name}_train_test/own_recall_{j}_{cut}/recall"],
                    label=f"{j} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_recall[f"{experiment_name}_val/own_recall_{j}_{cut}/recall"],
                    label=f"{j} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Recall for cut {cut}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Recall_cut{cut}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot precision for different cuts for same models
    for i, metric in enumerate(metrics_data):
        for cut in cuts:
            for j in [1, 2, 5, 10, 100]:
                plt.plot(
                    iteration[metrics_files[i]],
                    precision[
                        f"{experiment_name}_train_test/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{j} Train",
                    # color=colors[i],
                )
                plt.plot(
                    iteration[metrics_files[i]],
                    val_precision[
                        f"{experiment_name}_val/own_recall_{j}_{cut}/precision"
                    ],
                    label=f"{j} Val",
                    linestyle="dashed",
                    # color=colors[i],
                )

            plt.legend(loc="lower right")
            plt.title(f"Precision for cut {cut}: {title}, {metrics_files[i]}")
            plt.xlabel("Iteration")
            plt.ylabel("Precision")
            # plt.yscale("log")
            plt.savefig(
                os.path.join(
                    output_dir, f"Precision_cut{cut}_{title}_{metrics_files[i]}.png"
                ),
                dpi=300,
            )
            plt.clf()
            plt.cla()

    # Plot recall for different models

    # Plot precision for different models


def plot_wcs(filename, name, pred, target, aux, comp_catalog):
    """
    Plot the radio images with the WCS for proper RA and DEC in the images
    :param aux: CNN Aux data for this source
    :param name: Name of the source
    """
    from matplotlib import pyplot as plt
    from astropy.coordinates import SkyCoord

    (
        img_array,
        bounding_boxes,
        proposal_boxes,
        sem_seg_five,
        sem_seg_prop_five,
        sem_seg_three,
        sem_seg_prop_three,
        wcs,
    ) = np.load(filename, fix_imports=True, allow_pickle=True)
    ra_array = np.array(aux[1:, 2], dtype=float)
    dec_array = np.array(aux[1:, 3], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")
    pixel_coords = sky_coords.to_pixel(wcs)
    components = comp_catalog[comp_catalog["Source_Name"] == name]
    print(components)
    if len(components) > 1:
        coords = SkyCoord(components["RA"], components["DEC"], unit="deg")
        fluxes = components["Total_flux"].data
        comp_coords = coords.to_pixel(wcs)
        # Now need flux weighted center
        center_x = 0
        center_y = 0
        fluxes = fluxes / np.sum(fluxes)
        for c in range(len(comp_coords[0])):
            center_x += comp_coords[0][c] * fluxes[c]
            center_y += comp_coords[1][c] * fluxes[c]
    else:
        center_x = img_array[:, :, 0].shape[0] / 2
        center_y = img_array[:, :, 0].shape[1] / 2

    # Get Baseline
    source_coord = SkyCoord.from_pixel(xp=center_x, yp=center_y, wcs=wcs)
    d2d = source_coord.separation(sky_coords)
    baseline_pred = np.argmin(d2d.data)  # Need to convert back from Angle to non-Angle
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    im = plt.imshow(
        np.clip(img_array[:, :, 0], 0.0, 1000.0), origin="lower", cmap=plt.cm.viridis
    )
    ax.scatter(pixel_coords, size=2, label="Optical Source")
    ax.scatter(pixel_coords[pred - 1], size=5, c="red", marker="*", label="Predicted")
    ax.scatter(pixel_coords[target - 1], size=5, c="green", marker="x", label="Actual")
    ax.scatter(pixel_coords[1], size=5, c="orange", marker="1", label="Closest")
    ax.scatter(
        pixel_coords[baseline_pred], size=5, c="purple", marker="2", label="Baseline"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Flux (mJy)")
    plt.title(name)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend(loc="best")
    plt.savefig(f"{name}_prediction_plot.png", dpi=300)
    plt.cla()
    plt.clf()
