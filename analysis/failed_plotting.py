from lofarnn.visualization.metrics import plot_axis_recall, plot_plots, plot_compared_axis_recall, plot_cutoffs
import pickle
import numpy as np
import matplotlib.pyplot as plt

recall_dir = "/home/jacob/Development/reports/rotated_f_redux_size400_prop4096_depth101_batchSize8_lr0.002_frac1.0/"
recall_files = [
"/home/jacob/Development/test_lofarnn/lofarnn/analysis/eval_final_testfinal_eval_test/Test_source_recall_epoch39.pkl"
]
baseline = "/home/jacob/Development/test_lofarnn/lofarnn/analysis/final_test_closest_baseline_recall.pkl"
recall_limits = ["FinalBaseComp40", "t1", "v2", "t2", "v5", "t5"]
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

mydict = pickle.load(open(recall_files[0], "rb"), fix_imports=True)
"""
Plot failed prediction images, don't actually have what the prediction was, but have it wrong
"""
json_file = ""
annotations = pickle.load(open(json_file, "rb"), fix_imports=True)

def get_image(name, annotations):
    for anno in annotations:
        source_name = anno["file_name"].split("/")[-1].split(".cnn")[0]
        if source_name == name:
            return np.load(anno["file_name"], fix_imports=True)


for key, value in mydict.items():
    if value == 0:
        image = get_image(key, annotations)
        plt.imshow(image)
        plt.title(key)
        plt.savefig(f"failed_{key}.png", format="png", dpi=300)