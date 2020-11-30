from detectron2.utils.logger import setup_logger

setup_logger()
import os
import numpy as np
from lofarnn.models.dataloaders.utils import get_lofar_dicts, get_only_mutli_dicts

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from detectron2.engine import default_argument_parser, default_setup, launch
from lofarnn.models.dataloaders.utils import make_physical_dict
from lofarnn.models.trainer.SourceTrainer import SourceTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg


def calc_epochs(cfg, num_train_samples):
    """
    Calculate the number of epochs with the current batch size, so that all the models go through the same number of epochs?
    :param cfg: CFG instance
    :param num_train_samples:
    :return: new iteration number for equal epochs to 200000 at batch size of 4
    """

    iterations_per_epoch = int(np.ceil(num_train_samples / cfg.SOLVER.IMS_PER_BATCH))
    baseline_epochs = int(np.ceil(200000 / (num_train_samples / 4)))
    new_max_iterations = baseline_epochs * iterations_per_epoch
    print(f"New Max Iterations: {new_max_iterations}")
    return new_max_iterations


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    experiment_name = (
        args.experiment + f"_size{cfg.INPUT.MIN_SIZE_TRAIN[0]}"
        f"_prop{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}"
        f"_depth{cfg.MODEL.RESNETS.DEPTH}"
        f"_batchSize{cfg.SOLVER.IMS_PER_BATCH}"
        f"_lr{cfg.SOLVER.BASE_LR}"
        f"_frac{args.fraction_train}"
    )
    if environment == "XPS":
        cfg.OUTPUT_DIR = os.path.join("/home/jacob/", "reports", experiment_name)
    else:
        cfg.OUTPUT_DIR = os.path.join(
            "/home/s2153246/data/", "reports", experiment_name
        )
    # Register train set with fraction
    if "multi_only" in args.experiment:
        for d in ["train"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_only_mutli_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    multi=True,
                    vac=args.vac_file,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )
        # Keep val and test set the same so that its always testing on the same stuff
        for d in ["val", "test", "train_test"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_lofar_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    fraction=1,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )
    elif "single_only" in args.experiment:
        for d in ["train"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_only_mutli_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    multi=False,
                    vac=args.vac_file,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )
        # Keep val and test set the same so that its always testing on the same stuff
        for d in ["val", "test", "train_test"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_lofar_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    fraction=1,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )
    else:
        for d in ["train"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_lofar_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    fraction=args.fraction_train,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )
        # Keep val and test set the same so that its always testing on the same stuff
        for d in ["val", "test", "train_test"]:
            DatasetCatalog.register(
                f"{args.experiment}_" + d,
                lambda d=d: get_lofar_dicts(
                    os.path.join(
                        args.dataset,
                        f"json_{d}_prop{args.precompute}_all{args.all_channel}_multi{args.multi_bbox}_seg{args.semseg}_norm{args.norm}.pkl",
                    ),
                    fraction=1,
                ),
            )
            MetadataCatalog.get(f"{args.experiment}_" + d).set(
                thing_classes=["Optical source"]
            )

    cfg.DATASETS.TRAIN = (
        f"{args.experiment}_train",
        f"{args.experiment}_val",
    )  # Now train on val too
    cfg.DATASETS.VAL = (f"{args.experiment}_test",)
    cfg.DATASETS.TEST = (
        f"{args.experiment}_val",
        f"{args.experiment}_train_test",
        f"{args.experiment}_test",
    )  # Swapped because TEST is used for eval, and val is not, but can be used later
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    if ".npy" in args.vac_file:
        physical_dict = np.load(args.vac_file, allow_pickle=True)
    else:
        physical_dict = make_physical_dict(
            args.vac_file,
            size_cut=args.size_cut,
            flux_cut=args.flux_cut,
            multi=True,
            lgz=True,
        )
    trainer = SourceTrainer(cfg, physical_dict=physical_dict)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
