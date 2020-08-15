import numpy as np
from sys import argv

path_to_output = str("/home/jacob/Development/lofarnn/analysis/slurm-24273.out")

recall = []
experiment = ""
with open(path_to_output, "r") as data:
    for line in data:
        if "Experiment: " in line:
            experiment = line[11:].strip()
            print(experiment)
        elif (
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = "
            in line
        ):
            value = line.split(
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = "
            )[-1]
            print(value)
            recall.append(float(value))

np.save(experiment + "recall", recall)
print(recall)
import matplotlib.pyplot as plt

x = list(range(len(recall)))
plt.plot(x, recall)
plt.show()
