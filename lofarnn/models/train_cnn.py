import os
import numpy as np
import argparse
from lofarnn.models.dataloaders.utils import get_lofar_dicts

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
        help="whether to normalize point locations, default False",
    )
    parser.add_argument(
        "--num-sources", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument("--dataset", type=str, default="", help="path to dataset file")
    parser.add_argument(
        "--test-dataset", type=str, default="", help="path to test dataset file"
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


def test(args, model, device, test_loader):
    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, data.y, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(data.y.view_as(pred)).sum().item()

            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader), 100.0 * correct / len(test_loader)
        )
    )
    a = np.asarray(save_test_loss)
    with open(os.path.join(args.output_dir, "test_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")


def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []
    total_loss = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
        loss.backward()

        save_loss.append(loss.item())

        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {}\tLoss: {:.6f} \t Average loss {:.6f}".format(
                    epoch, loss.item(), total_loss / (batch_idx + 1)
                )
            )
    a = np.asarray(save_loss)
    with open(os.path.join(args.output_dir, "train_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")


def setup(args):
    """
    Setup dataset and dataloaders for these new datasets
    """
    train_dataset = dataset.TensorDataset()
    test_dataset = dataset.TensorDataset()
    return train_dataset, test_dataset


def main(args):
    train_dataset, test_dataset = setup(args)
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count()
    )
    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False, num_workers=os.cpu_count()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadioSingleSourceModel(args.classes, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model created")
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
