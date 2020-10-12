from lofarnn.visualization.metrics import plot_axis_recall, plot_plots, plot_compared_axis_recall, plot_cutoffs
import os
import pickle
import numpy as np

recall_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize8_lr0.002_frac1.0/"
recall_files = [
"/home/jacob/Development/test_lofarnn/lofarnn/Test_source_recall_epoch92.pkl"
]
baseline = "/home/jacob/Development/test_lofarnn/lofarnn/baselines/test_closest_baseline_recall.pkl"
recall_limits = ["BaseComp", "t1", "v2", "t2", "v5", "t5"]
vac_catalog = "/home/jacob/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
experiment_name = "rotated_f_redux"
experiment_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize8_lr0.002_frac1.0/inference/"
output_dir = "./"
cuts = ["single_comp", "multi_comp", "size15.0_flux10.0"]
labels = ["Single", "Multi", "Jelle"]
title = ["Rotated Fixed"]
colors = ["blue", "green", "black", "orange", "red"]
metrics_files = ["metrics"]

report_dir = "/home/jacob/Development/reports/"
generation_dirs = []
generation_filenames = []
experiment_dirs = []
titles = []



for i, f in enumerate(recall_files):
    plot_cutoffs(recall_path=f, recall_path_2=baseline, vac_catalog=vac_catalog, bins=10)
    plot_compared_axis_recall(
        recall_path=f,
        recall_path_2=baseline,
        vac_catalog=vac_catalog,
        bins=4,
        jelle_cut=False,
        limit=recall_limits[i],
        output_dir="./"
    )
    plot_compared_axis_recall(
        recall_path=f,
        recall_path_2=baseline,
        vac_catalog=vac_catalog,
        bins=4,
        jelle_cut=True,
        limit=recall_limits[i],
    )
exit()
for path, subdirs, files in os.walk(report_dir):
    for name in files:
        if "limit1.pkl" in name:
            generation_filenames.append(os.path.join(path, name))
            generation_dirs.append(path)
for path, subdirs, files in os.walk(report_dir):
    for name in files:
        if "metrics.json" in name:
            experiment_dirs.append(path)
        if "gauss" in path:
            titles.append("Gaussian")
        elif "rotated_v" in path:
            titles.append("Rotated Variable")
        elif "frcnn" in path:
            titles.append("Faster RCNN")
        else:
            titles.append("Rotated Fixed")
for i, item in enumerate(experiment_dirs):
    try:
        plot_plots(
            metrics_files=metrics_files,
            cuts=cuts,
            experiment_dir=experiment_dirs[i],
            output_dir=experiment_dirs[i],
            labels=labels,
            title=titles[i],
            colors=colors,
            experiment_name=experiment_name,
        )
    except Exception as e:
        print(f"Failed: {e}")
for i, item in enumerate(generation_filenames):
    limit = "t1" if "train" in item else "v1"
    plot_axis_recall(
        recall_path=item,
        vac_catalog=vac_catalog,
        bins=4,
        jelle_cut=False,
        limit=limit,
        output_dir=generation_dirs[i],
    )
    plot_axis_recall(
        recall_path=item,
        vac_catalog=vac_catalog,
        bins=4,
        jelle_cut=True,
        limit=limit,
        output_dir=generation_dirs[i],
    )
exit()
# for i, f in enumerate(recall_files):
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=False, limit=recall_limits[i])
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=True, limit=recall_limits[i])
