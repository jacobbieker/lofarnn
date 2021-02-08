from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

from lofarnn.data.cutouts import convert_to_valid_color

onlyfiles = [
    f
    for f in listdir("/home/jacob/LGZ_Pres/")
    if isfile(join("/home/jacob/LGZ_Pres/", f))
]

for f in onlyfiles:
    data = np.load(
        join("/home/jacob/LGZ_Pres/", f), allow_pickle=True, fix_imports=True
    )
    print(len(data))
    img = data[0]
    img[0] = convert_to_valid_color(img[0])
    img[:, :, 0] = convert_to_valid_color(
        img[:, :, 0],
        clip=True,
        lower_clip=0.0,
        upper_clip=500,
        normalize=True,
        scaling=None,
    )
    img[:, :, 1] = convert_to_valid_color(
        img[:, :, 1],
        clip=True,
        lower_clip=10.0,
        upper_clip=30.0,
        normalize=True,
        scaling=None,
    )
    img[:, :, 2] = convert_to_valid_color(
        img[:, :, 2],
        clip=True,
        lower_clip=10.0,
        upper_clip=30.0,
        normalize=True,
        scaling=None,
    )
    img = img[41:-41, 41:-41]
    plt.imshow(img[:, :, :3])
    plt.show()
