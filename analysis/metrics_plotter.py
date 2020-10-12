from lofarnn.visualization.metrics import plot_axis_recall, plot_plots
import os

recall_dir = "/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/"
recall_files = [
    os.path.join(recall_dir, "inference", "rotated_v_val_recall_limit1.pkl"),
    os.path.join(recall_dir, "inference","rotated_v_train_test_recall_limit1.pkl"),
    os.path.join(recall_dir, "inference","rotated_v_val_recall_limit2.pkl"),
    os.path.join(recall_dir, "inference","rotated_v_train_test_recall_limit2.pkl"),
    os.path.join(recall_dir,"inference", "rotated_v_val_recall_limit5.pkl"),
    os.path.join(recall_dir, "inference","rotated_v_train_test_recall_limit5.pkl"),
]
recall_limits = ["v1", "t1", "v2", "t2", "v5", "t5"]
vac_catalog = "/run/media/jacob/SSD_Backup/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
experiment_name = "rotated_v"
experiment_dir = "/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/"
output_dir = "./"
cuts = ["single_comp", "multi_comp", "size15.0_flux10.0"]
labels = ["Single", "Multi", "Jelle"]
title = ["Rotated Variable"]
colors = ["blue", "green", "black", "orange", "red"]
metrics_files = ["metrics"]

report_dir = "/home/jacob/Development/reports/"
generation_dirs = []
generation_filenames = []
experiment_dirs = []
titles = []

plot_axis_recall(recall_path="/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Test_source_recall_epoch3.pkl",
                 vac_catalog=vac_catalog,
                 bins=4,
                 jelle_cut=False,
                 limit="VSingle4",
                 output_dir="./")
plot_axis_recall(recall_path="/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Test_source_recall_epoch3.pkl",
                 vac_catalog=vac_catalog,
                 bins=4,
                 jelle_cut=True,
                 limit="VJSingle4",
                 output_dir="./")
plot_axis_recall(recall_path="/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Train_test_source_recall_epoch3.pkl",
                 vac_catalog=vac_catalog,
                 bins=4,
                 jelle_cut=False,
                 limit="TSingle4",
                 output_dir="./")
plot_axis_recall(recall_path="/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Train_test_source_recall_epoch3.pkl",
                 vac_catalog=vac_catalog,
                 bins=4,
                 jelle_cut=True,
                 limit="TJSingle4",
                 output_dir="./")
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
    plot_axis_recall(recall_path=item, vac_catalog=vac_catalog, bins=4, jelle_cut=False, limit=limit, output_dir=generation_dirs[i])
    plot_axis_recall(recall_path=item, vac_catalog=vac_catalog, bins=4, jelle_cut=True, limit=limit, output_dir=generation_dirs[i])
exit()
#for i, f in enumerate(recall_files):
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=False, limit=recall_limits[i])
#    plot_axis_recall(recall_path=f, vac_catalog=vac_catalog, bins=6, jelle_cut=True, limit=recall_limits[i])
