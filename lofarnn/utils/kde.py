import pickle
import numpy as np
from json import load
import os


def check_box(bounding_box, kde_prediction):
    """
    Check if a point, usually from KDE prediction, is within the bounding box
    :param bounding_box:
    :param kde_prediction:
    :return:
    """

    return bounding_box[0] <= kde_prediction[0] <= (
        bounding_box[0] + bounding_box[2]
    ) and bounding_box[1] <= kde_prediction[1] <= (bounding_box[1] + bounding_box[3])


def check_predictions(nn_prediction_path, training_path, save_cutout_location):
    """
    Compare Jelle KDE vs NN predictions
    :param nn_prediction_path:
    :param training_path:
    :return:
    """
    predictions = load(open(nn_prediction_path, "r"))
    training_data = load(open(training_path, "r"))
    training_data = training_data["images"]
    compared_predictions = {}
    fraction_matched = 0
    total = 0
    for image_data in training_data:
        # Now go through each dict to get the source Name, ground truth, npy as its 10 channel
        # print(image_data)
        source_name = image_data["file_name"].split("/")[-1].split(".npy")[0]
        # print(source_name)
        image_id = image_data["id"]
        # Get the Jelle Prediction
        print(os.path.join(save_cutout_location, f"{source_name}.pkl"))
        if os.path.exists(os.path.join(save_cutout_location, f"{source_name}.pkl")):
            print("Found!")
            jelle_prediction = pickle.load(
                open(os.path.join(save_cutout_location, f"{source_name}.pkl"))
            )
        else:
            continue
        print(jelle_prediction)
        # Get all the predictions for this image_id, taking the highest one as the final one
        # Since in the predictions are ordered by highest to lowest, can skip until next one
        found = False
        for pred_data in predictions:
            if found:
                break
            if pred_data["image_id"] == image_id:
                bbox = pred_data["bbox"]
                in_pred = check_box(bbox, jelle_prediction)
                compared_predictions[source_name] = in_pred
                if in_pred:
                    fraction_matched += 1
                    total += 1
                else:
                    total += 1
                found = True
    pickle.dump(compared_predictions, open(f"compared_predictions.pkl", "wb"))
    print(f"Fraction Matched: {float(fraction_matched)/total}")


check_predictions(
    nn_prediction_path="/home/jacob/Development/lofarnn/inference_fixed/inference_fixed/coco_instances_results.json",
    training_path="/home/jacob/Development/lofarnn/inference_fixed/inference_fixed/fast_rcnn_all_fixed_test_coco_format.json",
    save_cutout_location="/home/jacob/variable_lgz_tester/COCO/all/",
)
