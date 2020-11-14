import argparse
import os
import pickle
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from lofarnn.models.dataloaders.datasets import RadioSourceDataset
import torch.nn.functional as F
from lofarnn.models.base.cnn import (
    RadioSingleSourceModel,
    RadioMultiSourceModel,
    f1_loss,
)
from lofarnn.models.base.resnet import BinaryFocalLoss
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
        "--shuffle",
        action="store_true",
        help="whether to use shuffle multisource order, no effect for single source, default False",
    )
    parser.add_argument(
        "--num-sources", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="max number of nodes to use",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument(
        "--num-trials", type=int, default=100, help="max number of trials to perform",
    )
    parser.add_argument(
        "--classes", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="path to dataset annotations files"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross-entropy",
        help="loss to use, from 'cross-entropy' (default), 'focal', 'f1' ",
    )
    parser.add_argument("--experiment", type=str, default="", help="experiment name")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr-type",
        type=str,
        default="",
        help="learning rate type: None (default), 'plataeu', or 'cyclical' ",
    )
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--fraction", type=float, default=1.0, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="number of minibatches between logging",
    )

    return parser


def only_image_transforms(image, sources):
    """
    Only applies transforms to the image, and leaves the sources as they are
    """
    sequence = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.RandomErasing(),
        ]
    )
    return sequence(image), sources


def radio_transforms(image, sources):
    if np.random.random() > 0.5:
        angle = np.random.randint(-180, 180)
        image = TF.rotate(image, angle)
        # Add that angle, in radians, to the sources
        if sources.ndim == 1:  # Single source
            sources[1] = sources[1] + (
                np.deg2rad(angle) / (2 * np.pi)
            )  # Add radians to source in 0 to 1 scale
            sources[1] = (sources[1] + 1.0) % 1.0  # keep between 0 and 1
        elif sources.ndim == 2:
            for i, item in enumerate(sources):
                sources[i][1] = sources[i][1] + (
                    np.deg2rad(angle) / (2 * np.pi)
                )  # Add radians to source
                sources[i][1] = (sources[i][1] + 1.0) % 1.0
    # Random flips
    # image = TF.hflip(image)
    # image = TF.vflip(image)
    return image, sources


def setup(args):
    """
    Setup dataset and dataloaders for these new datasets
    """
    train_dataset = RadioSourceDataset(
        [os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}_extra.pkl"), os.path.join(args.dataset, f"cnn_val_norm{args.norm}_extra.pkl")],
        single_source_per_img=args.single,
        shuffle=args.shuffle,
        norm=not args.norm,
        num_sources=args.num_sources,
        fraction=args.fraction,
        transform=only_image_transforms if args.augment else None,
    )
    train_test_dataset = RadioSourceDataset(
        [os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}_extra.pkl"), os.path.join(args.dataset, f"cnn_val_norm{args.norm}_extra.pkl")],
        single_source_per_img=args.single,
        shuffle=args.shuffle,
        norm=not args.norm,
        num_sources=args.num_sources,
    )
    val_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_test_norm{args.norm}_extra.pkl"),
        single_source_per_img=False,
        shuffle=args.shuffle,
        norm=not args.norm,
        num_sources=args.num_sources,
    )
    args.world_size = args.gpus * args.nodes
    return train_dataset, train_test_dataset, val_dataset


def test(
    args,
    model,
    device,
    test_loader,
    epoch,
    name="Test",
    output_dir="./",
    config={"loss": "cross-entropy"},
):
    save_test_loss = []
    save_correct = []
    save_recalls = []
    recall = 0
    precision = 0

    named_recalls = {}

    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = BinaryFocalLoss(
        alpha=[config["alpha_1"], config["alpha_2"]],
        gamma=config["gamma"],
        reduction="mean",
    )
    with torch.no_grad():
        for data in test_loader:
            image, source, labels, names = (
                data["images"].to(device),
                data["sources"].to(device),
                data["labels"].to(device),
                data["names"],
            )
            output = model(image, source)
            # sum up batch loss
            if config["loss"] == "cross-entropy":
                try:
                    test_loss += F.binary_cross_entropy(
                        F.softmax(output, dim=-1), labels
                    ).item()
                except RuntimeError:
                    print(output)
            elif config["loss"] == "f1":
                test_loss += f1_loss(
                    output, labels.argmax(dim=1), is_training=False
                ).item()
            elif config["loss"] == "focal":
                test_loss += loss_fn(output, labels).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            label = labels.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            # Now get named recall ones
            if not args.single:
                for i in range(len(names)):
                    # Assumes testing is with batch size of 1
                    named_recalls[names[i]] = pred.eq(label.view_as(pred)).sum().item()
                    recall += pred.eq(label.view_as(pred)).sum().item()
            else:
                for i in range(len(names)):
                    if (
                        label.item() == 0
                    ):  # Label is source, don't care about the many negative examples
                        if pred.item() == 0:  # Prediction is source
                            named_recalls[names[i]] = 1  # Value is correct
                            recall += 1
                        else:  # Prediction is not correct
                            named_recalls[names[i]] = 0  # Value is incorrect

            save_test_loss.append(test_loss)
            save_correct.append(correct)
    recall /= len(test_loader.dataset.annotations)
    save_recalls.append(recall)
    test_loss /= len(test_loader)

    print(
        "\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) Recall: {:.2f}%\n".format(
            name,
            test_loss,
            correct,
            len(test_loader),
            100.0 * correct / len(test_loader),
            100.0 * recall,
        )
    )
    pickle.dump(
        named_recalls,
        open(os.path.join(output_dir, f"{name}_source_recall_epoch{epoch}.pkl"), "wb"),
    )
    a = np.asarray(save_test_loss)
    with open(os.path.join(output_dir, f"{name}_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
    a = np.asarray(save_recalls)
    with open(os.path.join(output_dir, f"{name}_recall.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
    if config["single"]:
        return correct
    else:
        return test_loss

def test2(
        args,
        model,
        device,
        test_loader,
        wanted_names,
        name="Test",
        output_dir="./",
        config={"loss": "cross-entropy"},
):

    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = BinaryFocalLoss(
        alpha=[config["alpha_1"], config["alpha_2"]],
        gamma=config["gamma"],
        reduction="mean",
    )
    mydict = pickle.load(open("/home/jacob/reports/eval_final_testfinal_eval_test/Test_source_recall_epoch39.pkl", "rb"), fix_imports=True)

    with torch.no_grad():
        for data in test_loader:
            image, source, labels, names = (
                data["images"],
                data["sources"],
                data["labels"],
                data["names"],
            )
            for names in names:
                if mydict[name] == 0:
                    output = model(image, source)
                    # sum up batch loss
                    if config["loss"] == "cross-entropy":
                        try:
                            test_loss += F.binary_cross_entropy(
                                F.softmax(output, dim=-1), labels
                            ).item()
                        except RuntimeError:
                            print(output)
                    elif config["loss"] == "f1":
                        test_loss += f1_loss(
                            output, labels.argmax(dim=1), is_training=False
                        ).item()
                    elif config["loss"] == "focal":
                        test_loss += loss_fn(output, labels).item()
                    # get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)
                    label = labels.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label.view_as(pred)).sum().item()


def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    output_dir="./",
    config={"loss": "cross-entropy"},
):
    save_loss = []
    total_loss = 0
    model.train()
    loss_fn = BinaryFocalLoss(
        alpha=[config["alpha_1"], config["alpha_2"]],
        gamma=config["gamma"],
        reduction="mean",
    )
    for batch_idx, data in enumerate(train_loader):
        image, source, labels, names = (
            data["images"].to(device),
            data["sources"].to(device),
            data["labels"].to(device),
            data["names"],
        )
        optimizer.zero_grad()
        output = model(image, source)
        if config["loss"] == "cross-entropy":
            loss = F.binary_cross_entropy(F.softmax(output, dim=-1), labels)
        elif config["loss"] == "f1":
            loss = f1_loss(output, labels.argmax(dim=1), is_training=True)
        elif config["loss"] == "focal":
            loss = loss_fn(output, labels)
        else:
            raise Exception("Loss not one of 'cross-entropy', 'focal', 'f1' ")
        loss.backward()

        save_loss.append(loss.item())

        optimizer.step()

        if args.lr_type == "cyclical":
            scheduler.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {}\tLoss: {:.6f} \t Average loss {:.6f}".format(
                    epoch, loss.item(), np.mean(save_loss[-args.log_interval :])
                )
            )
    if args.lr_type == "plateau":
        scheduler.step(total_loss)  # LROnPlateau step is after each epoch
    a = np.asarray(save_loss)
    with open(os.path.join(output_dir, "train_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
