import os

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from lofarnn.models.base.cnn import (
    RadioSingleSourceModel,
    RadioMultiSourceModel,
)
from lofarnn.models.base.utils import default_argument_parser, setup, test, train
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import dataloader
import torch


def main(args):
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        args.fraction = frac
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
            shuffle=False,
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
        experiment_name = args.experiment + f"frac_{frac}"
        if environment == "XPS":
            output_dir = os.path.join("/home/jacob/", "reports", experiment_name)
        else:
            output_dir = os.path.join(
                "/home/s2153246/data/", "reports", experiment_name
            )
        os.makedirs(output_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = {
            "act": "leaky",
            "fc_out": 186,
            "fc_final": 136,
            "single": args.single,
            "loss": args.loss,
            "gamma": 2,
            "alpha_1": 0.25,
        }
        config["alpha_2"] = 1.0 - config["alpha_1"]
        if args.single:
            model = RadioSingleSourceModel(1, 12, config=config).to(device)
        else:
            model = RadioMultiSourceModel(1, args.classes, config=config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.lr_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
        elif args.lr_type == "cyclical":
            scheduler = CyclicLR(
                optimizer,
                base_lr=args.lr,
                max_lr=0.1 if args.lr < 0.1 else 10 * args.lr,
            )
        else:
            scheduler = None
        print("Model created")
        for epoch in range(args.epochs):
            train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                output_dir,
                config,
            )
            test(
                args,
                model,
                device,
                train_test_loader,
                epoch,
                "Train_test",
                output_dir,
                config,
            )
            test(
                args, model, device, test_loader, epoch, "Val_Test", output_dir, config
            )
            if epoch % 5 == 0:  # Save every 5 epochs
                torch.save(
                    model, os.path.join(output_dir, f"model_{epoch}_frac{frac}.pth")
                )
        torch.save(model, os.path.join(output_dir, f"model_final.pth"))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
