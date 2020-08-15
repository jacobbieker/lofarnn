from lofarnn.utils.coco import get_all_pixel_mean_and_std_multi
from pathlib import Path

image_directory = "/home/jacob/variable_test_all_fluxlimit/COCO/train/"
image_paths = Path(image_directory).rglob("*.png")

get_all_pixel_mean_and_std_multi(image_paths, num_layers=3)
