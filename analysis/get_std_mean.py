import pickle

import matplotlib.pyplot as plt
import numpy as np
import optuna

json_file = "/home/jacob/Development/test_lofarnn/lofarnn/cnn_test_normTrue.pkl"
annotations = pickle.load(open(json_file, "rb"), fix_imports=True)

count = 0
positions = []
first_source = 0
named_recalls = {}
flux_named_recall = {}


def _get_source_name(name):
    source_name = name.split("/")[-1].split(".cnn")[0]
    return source_name


for anno in annotations:
    if isinstance(anno, np.ndarray):
        anno = anno.item()
    for i, elem in enumerate(anno["optical_sources"]):
        if i > 40:
            break
        positions.append(elem[0].value)
        if i == 0 and anno["optical_labels"][0] == 1:
            first_source += 1
            named_recalls[_get_source_name(anno["file_name"])] = 1
        elif i == 0:
            named_recalls[_get_source_name(anno["file_name"])] = 0
pickle.dump(
    named_recalls, open(f"train_closest_baseline_recall.pkl", "wb"),
)
# nonz = np.count_nonzero(anno["optical_labels"])
# if nonz == 0:
#    print("No Optical Source")
#    print(anno["optical_labels"])
# if nonz > 2:
#    print("Extra Ones")
#    print(anno["optical_labels"])
# count += nonz - 1
# print(count)
print(f"Recall for First Source: {first_source/len(annotations)}")
exit()
print(np.min(positions))
print(np.max(positions))
print(np.median(positions))
plt.hist(positions)
plt.show()
exit()
hist = optuna.get_all_study_summaries(
    storage="sqlite:///lotss_dr2_False_cross-entropy.db"
)
print(hist)
for elem in hist:
    print(elem.study_name)
    print(elem.best_trial)
    print("  Params: ")
    if elem.best_trial is not None:
        for key, value in elem.best_trial.params.items():
            print("    {}: {}".format(key, value))
exit()
print(hist[0].study_name)
print(hist[0].best_trial)
print(hist[1].study_name)
print(hist[1].best_trial)
print("  Params: ")
for key, value in hist[1].best_trial.params.items():
    print("    {}: {}".format(key, value))

json_file = "/run/media/jacob/SSD_Backup/cnn_train_test_normTrue_extra.pkl"
annotations = pickle.load(open(json_file, "rb"), fix_imports=True)

count = 0
positions = []
for anno in annotations:
    positions.append(np.argmax(anno["optical_labels"]))
    nonz = np.count_nonzero(anno["optical_labels"])
    if nonz == 0:
        print("No Optical Source")
        print(anno["optical_labels"])
    if nonz > 2:
        print("Extra Ones")
        print(anno["optical_labels"])
    count += nonz - 1
print(count)
print(np.min(positions))
print(np.max(positions))
print(np.median(positions))
plt.hist(positions, range=(1, 100))
plt.show()
