from lofarnn.visualization.metrics import plot_axis_recall, plot_plots
import os

recall_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize4_lr0.0001_frac1.0/inference/"
recall_files = [
    os.path.join(recall_dir, "rotated_f_redux_val_recall_limit1.pkl"),
    os.path.join(recall_dir, "rotated_f_redux_train_test_recall_limit1.pkl"),
    os.path.join(recall_dir, "rotated_f_redux_val_recall_limit2.pkl"),
    os.path.join(recall_dir, "rotated_f_redux_train_test_recall_limit2.pkl"),
    os.path.join(recall_dir, "rotated_f_redux_val_recall_limit5.pkl"),
    os.path.join(recall_dir, "rotated_f_redux_train_test_recall_limit5.pkl"),
]
recall_limits = ["v1", "t1", "v2", "t2", "v5", "t5"]
vac_catalog = "/home/jacob/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
experiment_name = "rotated_f_redux"
experiment_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize4_lr0.0001_frac1.0/"
output_dir = "./"
cuts = ["single_comp", "multi_comp", "size15.0_flux10.0"]
labels = ["Single", "Multi", "Jelle"]
title = ["Rotated Fixed"]
colors = ["blue", "green", "black", "orange", "red"]
metrics_files = ["metrics", "metrics2"]

# for i, f in enumerate(recall_files):
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=False, limit=recall_limits[i])
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=True, limit=recall_limits[i])
plot_plots(
    metrics_files=metrics_files,
    cuts=cuts,
    experiment_dir=experiment_dir,
    output_dir=output_dir,
    labels=labels,
    title=title[0],
    colors=colors,
    experiment_name=experiment_name,
)
