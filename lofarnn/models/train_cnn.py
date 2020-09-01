import os
try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from lofarnn.models.base.cnn import (
    RadioSingleSourceModel,
    RadioMultiSourceModel,
    f1_loss,
)
from lofarnn.models.base.utils import default_argument_parser, setup, test, train
from torch.utils.data import dataset, dataloader
import torch


def main(args):
    train_dataset, train_test_dataset, val_dataset = setup(args)
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True
    )
    train_test_loader = dataloader.DataLoader(
        train_test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )
    test_loader = dataloader.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )
    experiment_name = (
        args.experiment
        + f"_lr{args.lr}_b{args.batch}_single{args.single}_sources{args.num_sources}_norm{args.norm}_loss{args.loss}"
    )
    if environment == "XPS":
        output_dir = os.path.join("/home/jacob/", "reports", experiment_name)
    else:
        output_dir = os.path.join("/home/s2153246/data/", "reports", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "act": 'relu',
        "fc_out": 83,
        "fc_final": 140,
        "alpha_1": 0.25,
        "gamma": 2,
        "single": args.single,
        "loss": args.loss,
    }
    config["alpha_2"] = 1.0 - config["alpha_1"]
    if args.single:
        model = RadioSingleSourceModel(1, 11, config=config).to(device)
    else:
        model = RadioMultiSourceModel(1, args.classes, config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model created")
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, output_dir, config)
        test(args, model, device, train_test_loader, epoch, "Train_test", output_dir, config)
        test(args, model, device, test_loader, epoch, "Test", output_dir, config)
        if epoch % 5 == 0:  # Save every 5 epochs
            torch.save(model, os.path.join(output_dir, "model.pth"))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
