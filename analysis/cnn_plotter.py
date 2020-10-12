from lofarnn.visualization.metrics import plot_axis_recall, plot_plots
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from lofarnn.models.dataloaders.utils import get_lotss_objects


recall_dir = "/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/"
recall_files = []
train_test = []
test = []
for path, subdirs, files in os.walk(recall_dir):
    for name in files:
        if "Train_test" in name and ".pkl" in name:
            train_test.append(os.path.join(path, name))
        elif "Test" in name and ".pkl" in name:
            test.append(os.path.join(path, name))
vac_catalog = "/run/media/jacob/SSD_Backup/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
experiment_name = "multi_only_rotated_f_redux"
experiment_dir = "/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/"
output_dir = "./"
cuts = ["single_comp", "multi_comp", "size15.0_flux10.0"]
labels = ["Single", "Multi", "Jelle"]
title = ["Rotated Fixed Multi-Only"]
colors = ["blue", "green", "black", "orange", "red"]
metrics_files = ["metrics"]

report_dir = "/home/jacob/Development/reports/"
generation_dirs = []
generation_filenames = []
experiment_dirs = []
titles = []

# Plot overall losses
train_test_loss = np.loadtxt("/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Train_test_recall.csv")
test_loss = np.loadtxt("/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/Test_recall.csv")
train_loss = np.loadtxt("/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources20_normTrue_lossfocal_schedulerplateau/train_loss.csv")
#print(train_test_loss)
vac_catalog = get_lotss_objects(vac_catalog)
# Get only LGZ Ones

def load_files(paths, vac_catalog):
    data = [pickle.load(open(os.path.join(p), "rb"), fix_imports=True) for p in paths]
    cuts = {"Single": [], "Multi": [], "Jelle": []}
    vac_single = vac_catalog[vac_catalog["LGZ_Assoc"] == 1]
    vac_multi = vac_catalog[vac_catalog["LGZ_Assoc"] > 1]
    vac_catalog = vac_catalog[vac_catalog["LGZ_Size"] > 15.0]
    vac_catalog = vac_catalog[vac_catalog["Total_flux"] > 10.0]
    num_single = 0
    num_multi = 0
    num_jelle = 0
    for key in data[0].keys():
        if key in vac_multi["Source_Name"]:
            num_multi += 1
        if key in vac_single["Source_Name"]:
            num_single += 1
        if key in vac_catalog["Source_Name"]:
            num_jelle += 1
    print(num_single)
    print(num_multi)
    print(num_jelle)

    # Now go through each epoch and get recall for those ones
    s_recall = []
    m_recall = []
    j_recall = []
    for element in data:
        single_recall = 0
        multi_recall = 0
        jelle_recall = 0
        for key, value in element.items():
            if key in vac_multi["Source_Name"] and value == 1:
                multi_recall += 1
            if key in vac_single["Source_Name"] and value == 1:
                single_recall += 1
            if key in vac_catalog["Source_Name"] and value == 1:
                jelle_recall += 1
        s_recall.append(float(single_recall)/num_single)
        m_recall.append(float(multi_recall)/num_multi)
        j_recall.append(float(jelle_recall)/num_jelle)
    return s_recall, m_recall, j_recall


train_test = reversed(np.sort(train_test))
test = reversed(np.sort(test))
s, m, j = load_files(test, vac_catalog)
st, mt, jt = load_files(train_test, vac_catalog)

plt.plot(list(range(len(s))), s, label="Single Val")
plt.plot(list(range(len(m))), m, label="Multi Val")
plt.plot(list(range(len(j))), j, label="Jelle Val")
plt.plot(list(range(len(st))), st, label="Single Train")
plt.plot(list(range(len(mt))), mt, label="Multi Train")
plt.plot(list(range(len(jt))), jt, label="Jelle Train")
plt.ylabel("Recall")
plt.xlabel("Epoch")
plt.title("Recall for fixed CNN Multi ")
plt.legend(loc='best')
plt.show()
exit()





exit()
#plt.plot(list(range(len(train_loss))), train_loss, label="Train")
plt.plot(list(range(len(test_loss))), test_loss, label="Test")
plt.plot(list(range(len(train_test_loss))), train_test_loss, label="Train Test")
plt.legend(loc="best")
plt.show()

f = "/home/jacob/all_best_lr0.00024128_b8_singleFalse_sources41_normTrue_losscross-entropy_schedulercyclical/Train_test_source_recall_epoch283.pkl"
plot_axis_recall(
    recall_path=f,
    vac_catalog=vac_catalog,
    bins=4,
    jelle_cut=False,
    limit="CNN_MultiTrain",
)
plot_axis_recall(
    recall_path=f,
    vac_catalog=vac_catalog,
    bins=4,
    jelle_cut=True,
    limit="CNN_MultiTrain",
)
f = "/home/jacob/all_best_lr0.00024128_b8_singleFalse_sources41_normTrue_losscross-entropy_schedulercyclical/Test_source_recall_epoch60.pkl"
plot_axis_recall(
    recall_path=f,
    vac_catalog=vac_catalog,
    bins=4,
    jelle_cut=False,
    limit="CNN_MultiVal",
)
plot_axis_recall(
    recall_path=f,
    vac_catalog=vac_catalog,
    bins=4,
    jelle_cut=True,
    limit="CNN_MultiVal",
)
exit()
for i, f in enumerate(recall_files):
    plot_axis_recall(
        recall_path=f,
        vac_catalog=vac_catalog,
        bins=4,
        jelle_cut=False,
        limit=recall_limits[i],
    )
    plot_axis_recall(
        recall_path=f,
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
