# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from cv2 import imread
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from glob import glob
from shutil import copyfile, copyfileobj
import matplotlib.pyplot as plt

import sys
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.io import fits
import pandas as pd
from collections import Counter
from operator import itemgetter


import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation.evaluator import DatasetEvaluator


class SourceEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection,
    outputs using LOFAR relevant metrics.
    The relevant metric measures whether the proposed boxes cover the actual source location, and how far
    those box centers are from the ground truth bounding box, which was originally determined by LOFAR Galaxy Zoo
    That is: for all proposed boxes that cover the middle pixel of the input image check which
    sources from the component catalogue are inside.
    The predicted box can fail in three different ways:
    1. No predicted box covers the middle pixel
    2. The predicted box misses a number of components
    3. The predicted box encompasses too many components
    4. The prediction score for the predicted box is lower than other boxes that cover the middle
        pixel
    5. The prediction score is lower than x


    """

    def __init__(self, dataset_name, cfg,distributed, imsize,
                 overwrite=False, gt_data=None ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        # tasks are just ("bbox") in our case as we do not do segmentation or
        # keypoint prediction
        self._tasks = ("bbox",)
        self._distributed = distributed
        self._output_dir = cfg.OUTPUT_DIR
        self._dataset_name = dataset_name
        self._gt_data = np.array(gt_data)
        self._overwrite = overwrite
        self._imsize = imsize

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, self._output_dir)
            self._metadata.json_file = cache_path

        #json_file = PathManager.get_local_path(self._metadata.json_file)
        #with contextlib.redirect_stdout(io.StringIO()):
        #    self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        #self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._flattened_predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "file_name":input["file_name"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            self._predictions.append(prediction)
            #self._inputs.append(input)

    def evaluate(self):
        # for parallel execution
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            #self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[SourceEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        includes_associated_fail_fraction, includes_unassociated_fail_fraction = \
            _evaluate_predictions_on_lofar_score(self._dataset_name, self._predictions,
                                                 self._imsize, self._output_dir, save_appendix=self._dataset_name, scale_factor=self._scale_factor,
                                                 overwrite=self._overwrite, summary_only=True,
                                                 comp_cat_path=self._component_cat_path,
                                                 fits_dir=self._fits_path, gt_data=self._gt_data,
                                                 image_dir=self._image_dir, metadata=self._metadata)

        self._results = OrderedDict()
        self._results["bbox"] = {"assoc_single_fail_fraction": includes_associated_fail_fraction[0],
                                 "assoc_multi_fail_fraction": includes_associated_fail_fraction[1],
                                 "unassoc_single_fail_fraction": includes_unassociated_fail_fraction[0],
                                 "unassoc_multi_fail_fraction": includes_unassociated_fail_fraction[1]}
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.

        That is: for all proposed boxes that cover the middle pixel of the input image check which
        sources from the component catalogue are inside.
        The predicted box can fail in three different ways:
        1. No predicted box covers the middle pixel
        2. The predicted box misses a number of components
        3. The predicted box encompasses too many components
        4. The prediction score for the predicted box is lower than other boxes that cover the middle
            pixel
        5. The prediction score is lower than x

        """
        self._logger.info("Preparing results for COCO format ...")
        self._flattened_predictions = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._flattened_predictions))
                f.flush()

        #if not self._do_evaluation:
        #    self._logger.info("Annotations are not available for evaluation.")
        #    return

        self._logger.info("Evaluating predictions with LoTSS relevant metric...")


        # tasks are just ("bbox") in our case as we do not do segmentation or
        # keypoint prediction
        for task in sorted(tasks):
            print("Evaluating task:", task)
            raise NotImplementedError("we should not end up here")
            coco_eval = (
                _evaluate_predictions_on_lofar_score(lofar_gt,
                                                     self._flattened_predictions, task
                                                     )
                if len(self._flattened_predictions) > 0
                else None  # cocoapi does not handle empty results very well
            )
            # Format the data I guess?
            res = self._derive_flattened_predictions(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _eval_box_proposals(self):
        """
        Evaluate the box proposals in self._predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        raise NotImplementedError("This method is not looked at yet by Rafael.")
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in self._predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(
                    self._predictions, self._coco_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

    def _derive_flattened_predictions(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def number_of_components_in_dataset(output_dir, dataset_name, component_cat_path, predictions, fits_dir,
                                    overwrite=True, save_appendix='', only_zero_rotation=True):
    """Counts the number and percentage of single component sources in all_image_dir.
    Returns the filenames of the multi-component sources."""
    if not only_zero_rotation:
        raise NotImplementedError
    source_names_fits_names_save_path = f'{output_dir}/save_source_names_fits_paths_{save_appendix}.pkl'
    if overwrite or not os.path.exists(source_names_fits_names_save_path):
        # Load component catalogue
        comp_cat = pd.read_hdf(component_cat_path.replace('.fits','.h5'),'df')
        comp_name_to_source_name_dict = {n:i for i,n in zip(comp_cat.Source_Name.values,
                                                            comp_cat.Component_Name.values)}

        # Count the number of components per source name in the catalogue
        counts = pd.value_counts(comp_cat['Source_Name'])

        # Remove the end of the filename to retrieve the central source names
        png_file_names = [p["file_name"] for p in predictions]
        source_names = [f.split('/')[-1].split('_')[0] for f in png_file_names]
        fits_paths = [os.path.join(fits_dir, sn + '_radio_DR2.fits') for sn in source_names]

        # Check for duplicates (those should not exist)
        print("check for dups", len(predictions),len(fits_paths),len(source_names))
        assert len(source_names) == len(set(source_names)), 'duplicates should not exist'

        # Retrieve number of components per central source
        comps = [counts[comp_name_to_source_name_dict[source_name]] for source_name in source_names]

        # Get number of single comp
        single_comp = comps.count(1)
        print(f'There are {single_comp} single component sources and {len(source_names)-single_comp} multi.')
        print(f'Thus {single_comp/len(source_names)*100:.0f}% of the dataset is single component.')
        # Names of multi_comp sources
        multi_comp_source_names = [source_name for source_name in source_names
                                   if counts[comp_name_to_source_name_dict[source_name]] > 1]
        save_obj(source_names_fits_names_save_path, [source_names, fits_paths] )
    else:
        source_names, fits_paths = load_obj(source_names_fits_names_save_path)
    return np.array(source_names), fits_paths

def save_obj(file_path, obj):
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as input:
        return pickle.load(input)

def get_bounding_boxes(output):
    """Return bounding boxes inside inference output as numpy array
    """
    assert "instances" in output
    instances = output["instances"].to(torch.device("cpu"))


    return instances.get_fields()['pred_boxes'].tensor.numpy()


def is_within(x,y,xmin,ymin,xmax,ymax):
    """Return true if x, y lies within xmin,ymin,xmax,ymax.
    False otherwise.
    """
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    else:
        return False

def area(bbox):
    """Return area."""
    xmin,ymin,xmax,ymax = bbox
    width = xmax-xmin
    height = ymax-ymin
    area = width*height
    if area < 0:
        return None
    return area


def intersect_over_union(bbox1, bbox2):
    """Return intersection over union or IoU."""
    xmin1,ymin1,xmax1,ymax1 = bbox1
    xmin2,ymin2,xmax2,ymax2 = bbox2

    intersection_area = area([max(xmin1,xmin2),
                              max(ymin1,ymin2),
                              min(xmax1,xmax2),
                              min(ymax1,ymax2)])
    if intersection_area is None:
        return 0

    union_area = area(bbox1)+area(bbox1)-intersection_area
    assert intersection_area <= union_area
    return intersection_area / union_area


def collect_misboxed(predictions,image_dir, output_dir, fail_dir_name, fail_indices, source_names,metadata,
                     gt_data,gt_locs):
    """Collect ground truth bounding boxes that fail to encapsulate the ground truth pybdsf
    components so that they can be inspected to improve the box-draw-process"""
    # Make dir to collect the failed images in
    fail_dir = os.path.join(output_dir, fail_dir_name)
    os.makedirs(fail_dir,exist_ok=True)
    # Remove old directory but first check that it contains only pngs
    for f in os.listdir(fail_dir):
        assert f.endswith('.png'), 'Directory should only contain images.'
    for f in os.listdir(fail_dir):
        os.remove(os.path.join(fail_dir,f))

    # Copy debug images to this dir
    print('misboxed output dir',fail_dir, 'fail_indices:', fail_indices)
    full_image_dir = os.path.join(image_dir,"LGZ_v5_more_rotations/LGZ_COCOstyle/all")
    print('image dir is:', full_image_dir)
    print('sourcenames len is:', len(source_names), source_names[0])

    #    print('fail_indices:', source_names)
    #    print('fail_indices:', fail_indices)
    # if code fails here the debug source name or path is probably incorrect
    #image_source_paths = [os.path.join(full_image_dir, "*"+ source_name + "_rotated0deg.png")
    #        for source_name in source_names[fail_indices]]
    #print(image_source_paths[0])
    image_source_paths = [os.path.join(full_image_dir,source_name + "_radio_DR2_rotated0deg.png")
                          for source_name in source_names[fail_indices]]
    image_dest_paths = [os.path.join(fail_dir, image_source_path.split('/')[-1])
                        for image_source_path in image_source_paths]
    #[copyfile(src, dest) for src, dest in zip(image_source_paths, image_dest_paths)]
    image_only=False
    scale=2
    scale_factor=1.5151515151515151
    imsize=200
    final_imsize=imsize*scale
    if image_only:

        for src, dest in zip(image_source_paths, image_dest_paths):
            with open(src, 'rb') as fin:
                with open(dest, 'wb') as fout:
                    copyfileobj(fin, fout, 128*1024)
    else:

        (locs, focus_locs, close_comp_locs) = gt_locs
        (locs, focus_locs, close_comp_locs) = (np.array(locs)[fail_indices], np.array(focus_locs)[fail_indices],
                                               np.array(close_comp_locs)[fail_indices])
        print('locs,flocs, closecomplocs')
        print(locs[0])
        print(focus_locs[0])
        print(close_comp_locs[0])
        plt.close()

        for pred,gt, l, focus_l, close_l, src, dest in zip(np.array(predictions)[fail_indices],gt_data[fail_indices],
                                                           locs, focus_locs, close_comp_locs, image_source_paths, image_dest_paths):

            #print(pred)
            #print(dest)
            # Open mispredicted image
            im = imread(src)
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=scale,
                           instance_mode=ColorMode.IMAGE #_BW   # remove the colors of unsegmented pixels
                           )
            # Create another visualizer object as Deepcopy does not exist
            v2 = Visualizer(im[:, :, ::-1],
                            metadata=metadata,
                            scale=scale,
                            instance_mode=ColorMode.IMAGE #_BW   # remove the colors of unsegmented pixels
                            )

            v_gt = v.draw_dataset_dict(gt).get_image()[:, :, ::-1]
            v_pred = v2.draw_instance_predictions(pred["instances"].to("cpu")).get_image()[:, :, ::-1]
            # Plot figure
            #v_gt = v.overlay_instances(labels=['lol' for i in range(len(d['annotations']))],
            #                                           boxes=[x['bbox'] for x in d['annotations']],
            #                                           masks=None, keypoints=None).get_image()[:, :,::-1]



            # Plot figure
            f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,10))
            # Radio intensity + ground truth bboxes
            ax1.imshow(v_gt)
            ax1.set_title('Ground truth labels')
            # Radio intensity + predicted bboxes
            ax2.imshow(v_pred)
            ax2.set_title('Predicted labels')
            # Component locations
            s = scale*scale_factor
            # Filter out unassociated close components outside of cutout
            close_l = np.array([[x,y] for x,y in zip(close_l[0],close_l[1]) if (0 <= x < 200) and (0 <= y < 200)]).T
            #print(close_l)

            if not l.size == 0:
                ax1.plot(l[0]*s,final_imsize-l[1]*s,'ro')
                ax2.plot(l[0]*s,final_imsize-l[1]*s,'ro')
            ax1.plot(focus_l[0]*s,final_imsize-focus_l[1]*s,'rs')
            ax2.plot(focus_l[0]*s,final_imsize-focus_l[1]*s,'rs')
            if not close_l.size == 0:
                ax1.plot(close_l[0]*s,final_imsize-close_l[1]*s,color='orange', marker='o', linestyle='None')
                ax2.plot(close_l[0]*s,final_imsize-close_l[1]*s,color='orange', marker='o', linestyle='None')
            #plt.show()
            plt.savefig(dest, bbox_inches='tight')
            plt.close()

def _get_gt_bboxes(lofar_gt, focus_locs, close_comp_locs, save_appendix=None, overwrite=False):
    """Get ground truth bounding boxes"""
    central_bboxes_save_path = f'cache/save_central_bboxes_{save_appendix}.pkl'

    if overwrite or not os.path.exists(central_bboxes_save_path):
        # Get bounding boxes per image as numpy arrays
        bboxes_per_image = [[d['bbox'] for d in image_dict['annotations']] for image_dict in lofar_gt]


        # Filter out bounding box per image that covers the central pixel of the focussed box
        central_bboxes = [[tuple(bbox) for bbox in bboxes
                           if is_within(x*scale_factor,imsize-y*scale_factor, bbox[0],bbox[1],bbox[2],bbox[3])]
                          for bboxes, (x, y) in zip(bboxes_per_image, focus_locs)]

        # Filter out duplicates
        central_bboxes = [list(set(bboxes)) for bboxes in central_bboxes]
        # Assumption: Take only the smallest box left from this list
        smallest_area_indices = [np.argmin([area(bbox) for bbox in bboxes]) if not bboxes == [] else None
                                 for bboxes in central_bboxes]

        central_bboxes = [bboxes[ind] for bboxes, ind in zip(central_bboxes,smallest_area_indices)]

        save_obj(central_bboxes_save_path, central_bboxes)
        print('Done saving central_bboxes.')

    else:
        central_bboxes = load_obj(central_bboxes_save_path)
    return central_bboxes


def _evaluate_predictions_on_lofar_score(dataset_name, predictions, imsize, output_dir,
                                         save_appendix='',
                                         overwrite=True, summary_only=False,
                                         comp_cat_path=None, gt_data=None,
                                         only_zero_rotation=True,
                                         image_dir=None, metadata=None):
    """
    Evaluate the results using our LOFAR Source Finding appropriate score.

        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.

        That is: for all proposed boxes, see the difference between the center of the box and the ground truth box.
        The predicted box can fail in three different ways:
        4. The prediction score for the predicted box is lower than other boxes
        5. The prediction score is lower than x

    """
    debug=True

    ###################### ground truth

    if debug:
        print("source_names", "val_fits_path")

    # Get pixel locations of ground truth components
    #print('len valsourcenames,fitpaths',len(source_names),  len(val_fits_paths))
    gt_locs = (locs,focus_locs, close_comp_locs)
    if debug:
        print("ncomps", "locs, centrallocs, closecomplocs")
    print('len closecomplocs',len(close_comp_locs),  len(n_comps), len(locs), len(focus_locs))


    ###################### prediction
    # Get bounding boxes per image as numpy arrays
    pred_bboxes_scores = [(image_dict['instances'].get_fields()['pred_boxes'].tensor.numpy(),
                           image_dict['instances'].get_fields()['scores'].numpy())
                          for image_dict in predictions]
    if debug:
        print("pred_bboxes_scores")
        print(pred_bboxes_scores[0])

    if debug:
        print("pred_bboxes_scores after filtering out the focussed pixel")
        print(pred_central_bboxes_scores[0])

    # 1. No predicted box covers the middle pixel
    # can now be checked
    #     fail_fraction_1 = (len(central_bboxes)-len(pred_central_bboxes_scores))/len(central_bboxes)
    #     print(f'{(len(central_bboxes)-len(pred_central_bboxes_scores))} predictions '
    #           f'(or {fail_fraction_1:.1%}) fail to cover the central component of the source.')

    # Take only the highest scoring bbox from this list
    pred_central_bboxes_scores = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[0]
                                  if len(bboxes_scores) > 0 else [[-1,-1,-1,-1],0] for bboxes_scores in pred_central_bboxes_scores]

    if debug:
        print("pred_bboxes_scores after filtering out the focussed pixel")
    #return pred_central_bboxes_scores
    # Return IoU with ground truth
    if not summary_only:
        central_bboxes = _get_gt_bboxes(lofar_gt, focus_locs, close_comp_locs,
                                        save_appendix=save_appendix, overwrite=overwrite)

        iou_pred_central_bboxes = [intersect_over_union(bbox,bbox_score[0])
                                   for bbox, bbox_score in zip(central_bboxes, pred_central_bboxes_scores)]
        print(f'Mean IoU of predicted box for central source is {np.mean(iou_pred_central_bboxes):.2f}'
              f' with a std. dev. of {np.std(iou_pred_central_bboxes):.2f}')

    # Check if other source comps fall inside predicted central box
    #print([loc for loc in locs])
    #print([[x,y for x,y in np.dstack(loc)[0]] for loc in locs])
    # TODO yaxis flip hack
    comp_scores = [np.sum([is_within(x*scale_factor,imsize-y*scale_factor, bbox[0],bbox[1],bbox[2],bbox[3])
                           for x,y in list(zip(loc[0],loc[1]))])
                   for loc, (bbox, score) in zip(locs, pred_central_bboxes_scores)]
    #nana = [(scale_factor*x, scale_factor*y) for x, y in np.dstack(locs[inspect_id])[0]]
    #print("locs scaled", nana)

    return includes_associated_fail_fraction, includes_unassociated_fail_fraction

def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def get_bounding_boxes(output):
    """Return bounding boxes inside inference output as numpy array
    """
    assert "instances" in output
    instances = output["instances"].to(torch.device("cpu"))

    return instances.get_fields()['pred_boxes'].tensor.numpy()


def load_fits(fits_filepath, dimensions_normal=True):
    """Load a fits file and return its header and content"""
    # Load first fits file
    hdulist = fits.open(fits_filepath)
    # Header
    hdr = hdulist[0].header
    if dimensions_normal:
        hdu = hdulist[0].data
    else:
        hdu = hdulist[0].data[0,0]
    hdulist.close()
    return hdu, hdr


'''
def _evaluate_predictions_on_lofar_score(lofar_gt, coco_results, task):
    """
    Evaluate the results using our LOFAR appropriate score.

        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.

        That is: for all proposed boxes that cover the middle pixel of the input image check which
        sources from the component catalogue are inside. 
        The predicted box can fail in three different ways:
        1. No predicted box covers the middle pixel
        2. The predicted box misses a number of components
        3. The predicted box encompasses too many components
        4. The prediction score for the predicted box is lower than other boxes that cover the middle
            pixel
        5. The prediction score is lower than x
    
    """
    assert len(coco_results) > 0
    assert task == ('bbox'), 'segmentation and keypoints are not used by LOFAR score'
    print('Hier is coco_results:')
    print(type(coco_results))
    print(coco_results)
    

    # Get bounding boxes as numpy arrays
    bboxes = [get_bounding_boxes(output) for output in outputs]

    # 
    
    #coco_eval.evaluate()
    #coco_eval.accumulate()
    #coco_eval.summarize()

    return coco_eval
  '''