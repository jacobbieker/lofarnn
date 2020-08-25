import os
import numpy as np
import argparse
from lofarnn.models.dataloaders.datasets import RadioSourceDataset

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
from lofarnn.models.base.resnet import BinaryFocalLoss
from torch.utils.data import dataset, dataloader
import torch.nn.functional as F
import torch


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="whether to augment input data, default False",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="whether to normalize magnitudes or not, default False",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="whether to use single source or multiple, default False",
    )
    parser.add_argument(
        "--num-sources", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--classes", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="path to dataset annotations files"
    )
    parser.add_argument(
        "--loss", type=str, default="cross-entropy", help="loss to use, from 'cross-entropy' (default), 'focal', 'f1' "
    )
    parser.add_argument(
        "--experiment", type=str, default="", help="experiment name"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="number of minibatches between logging",
    )

    return parser


def test(args, model, device, test_loader, name="test", output_dir="./"):
    save_test_loss = []
    save_correct = []
    save_recalls = []
    recall = 0
    precision = 0

    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = BinaryFocalLoss(alpha=[0.25, 0.75], gamma=2, reduction='mean')
    with torch.no_grad():
        for data in test_loader:
            image, source, labels = (
                data["image"].to(device),
                data["sources"].to(device),
                data["labels"].to(device),
            )
            output = model(image, source)
            # sum up batch loss
            if args.loss == "cross-entropy":
                test_loss += F.binary_cross_entropy(F.softmax(output, dim=-1), labels).item()
            elif args.loss == "f1":
                test_loss += f1_loss(output, labels, is_training=False).item()
            elif args.loss == "focal":
                test_loss += loss_fn(output, labels).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            label = labels.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            save_test_loss.append(test_loss)
            save_correct.append(correct)
            if torch.eq(pred, label) and label.item() == 1:
                recall += 1

    recall /= len(test_loader.dataset.annotations)  # One source per annotation
    save_recalls.append(recall)
    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) Recall: {:.2f}%\n".format(
            test_loss,
            correct,
            len(test_loader),
            100.0 * correct / len(test_loader),
            100.0 * recall,
        )
    )
    a = np.asarray(save_test_loss)
    with open(os.path.join(output_dir, f"{name}_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
    a = np.asarray(save_recalls)
    with open(os.path.join(output_dir, f"{name}_recall.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")


def train(args, model, device, train_loader, optimizer, epoch, output_dir="./"):
    save_loss = []
    total_loss = 0
    model.train()
    loss_fn = BinaryFocalLoss(alpha=[0.25, 0.75], gamma=2, reduction='mean')
    for batch_idx, data in enumerate(train_loader):
        image, source, labels = (
            data["image"].to(device),
            data["sources"].to(device),
            data["labels"].to(device),
        )
        optimizer.zero_grad()
        output = model(image, source)
        if args.loss == "cross-entropy":
            loss = F.binary_cross_entropy(F.softmax(output, dim=-1), labels)
        elif args.loss == "f1":
            loss = f1_loss(output, labels, is_training=True)
        elif args.loss == "focal":
            loss = loss_fn(output, labels)
        else:
            raise Exception("Loss not one of 'cross-entropy', 'focal', 'f1' ")
        loss.backward()

        save_loss.append(loss.item())

        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {}\tLoss: {:.6f} \t Average loss {:.6f}".format(
                    epoch, loss.item(), np.mean(save_loss[-args.log_interval :])
                )
            )
    a = np.asarray(save_loss)
    with open(os.path.join(output_dir, "train_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")


def setup(args):
    """
    Setup dataset and dataloaders for these new datasets
    """
    train_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}.pkl"),
        single_source_per_img=args.single,
        shuffle=True,
    )
    train_test_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_val_norm{args.norm}.pkl"),
        single_source_per_img=args.single,
        shuffle=True,
    )
    val_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_test_norm{args.norm}.pkl"),
        single_source_per_img=args.single,
        shuffle=True,
    )
    return train_dataset, train_test_dataset, val_dataset


def main(args):
    train_dataset, train_test_dataset, val_dataset = setup(args)
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count()
    )
    train_test_loader = dataloader.DataLoader(
        train_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    test_loader = dataloader.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()
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
    if args.single:
        model = RadioSingleSourceModel(1, 10).to(device)
    else:
        model = RadioMultiSourceModel(1, args.classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model created")
    for epoch in range(args.epochs):
        test(args, model, device, test_loader, output_dir)
        train(args, model, device, train_loader, optimizer, epoch, output_dir)
        test(args, model, device, train_test_loader, "train_test", output_dir)
        if epoch % 5 == 0:  # Save every 5 epochs
            torch.save(model, os.path.join(output_dir, "model.pth"))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
