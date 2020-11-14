from lofarnn.visualization.models import visuaize_maps, fancy_visuaize_maps
from lofarnn.models.base.cnn import RadioMultiSourceModel, RadioSingleSourceModel
from lofarnn.models.base.utils import setup, default_argument_parser
from torch.utils.data import dataset, dataloader
from lofarnn.visualization.metrics import plot_wcs
import os
import torch
import torch.nn.functional as F
import numpy as np

directory = "/home/jacob/reports/test_crossentropy_lr0.00057_b6_singleTrue_sources4_normTrue_lossfocal_schedulerplateau/"
directory = "/run/media/jacob/SSD_Backup/all_best_lr0.00024128_b8_singleFalse_sources41_normTrue_losscross-entropy_schedulercyclical/"


def main(args):
    train_dataset, train_test_dataset, val_dataset = setup(args)
    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    train_test_loader = dataloader.DataLoader(
        train_test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    test_loader = dataloader.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "act": "leaky",
        "fc_out": 186,
        "fc_final": 136,
        "single": args.single,
        "loss": args.loss,
        "gamma": 2,
        "alpha_1": 0.12835728,
    }
    config["alpha_2"] = 1.0 - config["alpha_1"]
    # if args.single:
    #    model = RadioSingleSourceModel(1, 12, config=config).to(device)
    # else:
    #    model = RadioMultiSourceModel(1, args.classes, config=config).to(device)
    model = torch.load(os.path.join("/home/jacob/reports/eval_final_test_Resave_40_Fixed_Redshiftfinal_eval_test/", "model_15.pth"))
    model = model.to(device)
    images = []
    sources = []
    for data in test_loader:
        image, source, labels, names = (
            data["images"].to(device),
            data["sources"].to(device),
            data["labels"].to(device),
            data["names"],
        )
        print(source.ndim)
        print(F.softmax(labels, dim=-1).argmax(dim=1, keepdim=True).cpu().numpy())
        if np.all(F.softmax(labels, dim=-1).argmax(dim=1, keepdim=True).cpu().numpy() != 1):

        #if np.all(F.softmax(labels, dim=-1).argmax(dim=1, keepdim=True).cpu().numpy() == 1):
        fancy_visuaize_maps(
            model=model,
            inputs=(image, source),
            labels=labels,
            title=data["names"][0],
            second_occlusion=(1, 1, 1),
        )
        fancy_visuaize_maps(
            model=model,
            inputs=(image, source),
            labels=labels,
            title=data["names"][0],
            second_occlusion=(1, 1, 1),
            baselines=(1, 1),
        )

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
