"""

Copies the mosaiic diretories that are used in the DR1 data stuff to a new directory

"""

import os

import numpy as np

from lofarnn.data.datasets import get_lotss_objects

vac = "/home/jacob/Development/lofarnn/LoTSS_predicted_v0_merge.fits"
source_loc = "/run/media/jacob/easystore1/Research/LOFAR/mosaics/"
dest_loc = "/run/media/jacob/768E313E8E30F7E7/mosaics/"
l_objects = get_lotss_objects(vac, True)
mosaic_names = np.sort(list(set(l_objects["Mosaic_ID"])))
print("Mosaic IDs: ")
print(mosaic_names)
exit()
l_objects = l_objects[~np.isnan(l_objects["LGZ_Size"])]
l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
mosaic_names = np.sort(list(set(l_objects["Mosaic_ID"])))
print("Mosaic IDs: ")
print(mosaic_names)

for mosaic in mosaic_names:
    m_path = os.path.join(source_loc, mosaic)
    m_dest = os.path.join(dest_loc, mosaic)
    try:
        os.system(f"mkdir -p {m_dest} && cp -r {m_path}/* {m_dest}")
        if "Hetde" in mosaic:
            os.system(f"mkdir -p {m_dest} && cp -r {m_path}*/* {m_dest}")
    except Exception as s:
        print(s)
        os.system(f"mkdir -p {m_dest} && cp -r {m_path}*/* {m_dest}")
    # try:
    #    destination = shutil.copytree(m_path, m_dest)
    # except Exception as e:
    #    print(e)
