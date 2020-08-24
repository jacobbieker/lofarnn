import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from lofarnn.models.dataloaders.utils import get_lotss_objects


def load_json_arr(json_path):
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_axis_recall(recall_path, vac_catalog, limit, jelle_cut=False, bins=10, output_dir="./"):
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
        "Number of Components": radio_comp
    }

    ###calculate recall in bins
    # set which parameters you want to have on the X and Y axis
    for xlabel in ["Apparent size [arcsec]"]:
        for ylabel in ["Total flux [mJy]", "Axis ratio", "z", "Number of Components"]:
            X = data_dict[xlabel]
            Y = data_dict[ylabel]
            # get edges with maxima determined using percentiles to be robust for outliers
            x_bin_edges = np.linspace(
                np.nanpercentile(X, 1)-0.00001, np.nanpercentile(X, 98), bins + 1
            )
            #x_bin_edges = np.linspace(np.nanmin(X)-0.00001, np.nanpercentile(X, 98), bins+1)
            y_bin_edges = np.linspace(
                np.nanpercentile(Y, 1)-0.00001, np.nanpercentile(Y, 98), bins + 1
            )
            #y_bin_edges = np.linspace(np.nanmin(Y)-0.00001, np.nanpercentile(Y, 98), bins + 1)
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
            fig.savefig(os.path.join(output_dir,
                f"{xlabel}-{ylabel}_limit{limit}_jelle{jelle_cut}.png"),
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
