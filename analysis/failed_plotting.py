from lofarnn.visualization.metrics import plot_axis_recall, plot_plots, plot_compared_axis_recall, plot_cutoffs
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lofarnn.models.dataloaders.utils import get_lotss_objects
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


vac_catalog = get_lotss_objects("/home/jacob/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits")
json_file = "cnn_test_normTrue_extra.pkl"

annotations = pickle.load(open(json_file, "rb"), fix_imports=True)

qual = vac_catalog["LGZ_ID_Qual"]

qual_vals = np.unique(np.round(np.nan_to_num(qual), 1))

def get_image(name, annotations):
    for anno in annotations:
        source_name = anno["file_name"].split("/")[-1].split(".cnn")[0]
        if source_name == name:
            return np.load(anno["file_name"], fix_imports=True)


quals = {}
recalls = {}
misses = {}
counts = {}
for value in qual_vals:
    quals[value] = value
    recalls[value] = 0
    counts[value] = 0
    misses[value] = 0

print(counts)
all_quals = []
for anno in annotations:
    source_name = anno["file_name"].split("/")[-1].split(".cnn")[0]
    mask = vac_catalog["Source_Name"] == source_name
    v = np.round(np.nan_to_num(qual[mask].data), 1)[0]
    tmp = np.round(np.nan_to_num(qual[mask].data), 1)[0]
    if not np.isclose(tmp, 0.0):
        all_quals.append(tmp)
    if source_name in mydict:
        counts[v] += 1
        if mydict[source_name] == 1:
            recalls[v] += 1
        else:
            misses[v] += 1

x = []
y = []
y2 = []
for key in qual_vals:
    if key != 0.0:
        if counts[key] != 0:
            x.append(key)
            y.append(recalls[key]/counts[key])
            y2.append(misses[key]/counts[key])
        else:
            x.append(key)
            y.append(recalls[key] / 1)
            y2.append(misses[key] / 1)

fig, ax1 = plt.subplots()

heights,bins = np.histogram(all_quals,bins=20)
heights = heights/sum(heights)
ax2 = ax1.twinx()
ax2.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), edgecolor='black', color='none', zorder=10)
ax2.set_ylabel("Percentage of Sources")
fig.suptitle("Recall vs LGZ Quality")
ax1.plot(x, y, label=f"Correct: Median Quality: {np.round(np.median(y), 3)}")
ax1.plot(x, y2, label=f"Incorrect: Median Quality: {np.round(np.median(y2), 3)}")
ax1.set_ylabel("Recall")
ax1.set_xlabel("LGZ Quality")
ax1.legend(loc='best')
fig.tight_layout()
plt.show()
