from lofarnn.visualization.metrics import plot_axis_recall, plot_plots, plot_compared_axis_recall, plot_cutoffs
import os
import pickle
import numpy as np
import csv

recall_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize8_lr0.002_frac1.0/"
recall_files = [
"/home/jacob/Development/test_lofarnn/lofarnn/analysis/eval_final_testfinal_eval_test/Test_source_recall_epoch39.pkl",
"/home/jacob/Development/test_lofarnn/lofarnn/final_eval_test_size400_prop4096_depth101_batchSize8_lr0.001_frac1.0/inference/final_eval_test_test_recall_limit1.pkl"
]
baseline = "/home/jacob/Development/test_lofarnn/lofarnn/analysis/final_test_closest_baseline_recall.pkl"
recall_limits = ["CNN Test", "Fast RCNN Test", "v2", "t2", "v5", "t5"]
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

b = "/home/jacob/frac_test/"
folds = ["0.1", "0.25", "0.5", "0.75", "0.9", "1.0"]
cs = ["blue", "green", "orange", "red", "black", "purple"]
names = ["Train_test_source_recall_epoch", "Val_Test_source_recall_epoch"]
epochs = list(range(15))
# Plot them all
from numpy import genfromtxt
import matplotlib.pyplot as plt

for i, fol in enumerate(folds):
    train = genfromtxt(os.path.join(b, fol, "Train_test_loss.csv"))
    num_per_epoch = int(len(train)/15)
    avg_loss = []
    for j in epochs:
        avg_loss.append(np.mean(train[j*num_per_epoch:(j+1)*num_per_epoch]))
    val = genfromtxt(os.path.join(b, fol, "Val_Test_loss.csv"))
    avg_val = []
    num_per_epoch = int(len(val) / 15)
    for j in epochs:
        avg_val.append(np.mean(val[j*num_per_epoch:(j+1)*num_per_epoch]))
    plt.plot(epochs, avg_loss, c=cs[i], label=fol)
    plt.plot(epochs, avg_val, c=cs[i], linestyle='--')
plt.title("Recall vs Training Dataset Size")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
#plt.ylim(0.7,0.85)
plt.show()
exit()

for i, f in enumerate(recall_files):
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
plot_cutoffs(recall_path=f,
             recall_path_2="/home/jacob/Development/test_lofarnn/lofarnn/final_eval_test_size400_prop4096_depth101_batchSize8_lr0.001_frac1.0/inference/final_eval_test_test_recall_limit1.pkl",
             baseline_path=baseline, vac_catalog=vac_catalog, bins=10, name="baseline_95")
plot_cutoffs(recall_path=f,
             recall_path_2="/home/jacob/Development/test_lofarnn/noZ_rest/Test_source_recall_epoch35.pkl",
             baseline_path=baseline, vac_catalog=vac_catalog, bins=10, name="compare_z_effect_99",
             recall_names=["Z", "No Z"])
plot_cutoffs(recall_path=f,
             recall_path_2="/home/jacob/Development/test_lofarnn/eval_final_test_Resave_40_LEGAC_With_Redshiftfinal_eval_test/Test_source_recall_epoch10.pkl",
             baseline_path=baseline, vac_catalog=vac_catalog, bins=10, name="legac_with_z_99",
             recall_names=["CNN", "Legacy w/ Z"])
plot_cutoffs(recall_path=f,
             recall_path_2="/home/jacob/Development/test_lofarnn/Legac_current_final_eval_test/Test_source_recall_epoch15.pkl",
             baseline_path=baseline, vac_catalog=vac_catalog, bins=10, name="legac_no_z_99",
             recall_names=["CNN", "Legacy w/o Z"])
plot_cutoffs(
    recall_path="/home/jacob/Development/test_lofarnn/eval_final_test_Resave_40_LEGAC_With_Redshiftfinal_eval_test/Test_source_recall_epoch10.pkl",
    recall_path_2="/home/jacob/Development/test_lofarnn/Legac_current_final_eval_test/Test_source_recall_epoch15.pkl",
    baseline_path=baseline, vac_catalog=vac_catalog, bins=10, name="legac_compare_99",
    recall_names=["Legacy w/ Z", "Legacy w/o Z"])
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
