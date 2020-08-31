import os
import numpy as np
import argparse
import pickle
from lofarnn.models.dataloaders.datasets import RadioSourceDataset, collate_variable_fn

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
from lofarnn.models.base.utils import default_argument_parser
from torch.utils.data import dataset, dataloader
import torch.nn.functional as F
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import optuna

def test(
        args,
        model,
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
    loss_fn = BinaryFocalLoss(alpha=[0.25, 0.75], gamma=2, reduction="mean")
    with torch.no_grad():
        for data in test_loader:
            image, source, labels, names = (
                data["images"].cuda(non_blocking=True),
                data["sources"].cuda(non_blocking=True),
                data["labels"].cuda(non_blocking=True),
                data["names"]
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
            else:
                for i in range(len(names)):
                    if label.item() == 0:  # Label is source, don't care about the many negative examples
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
    pickle.dump(named_recalls, open(os.path.join(output_dir, f"{name}_source_recall_epoch{epoch}.pkl"), "wb"))
    a = np.asarray(save_test_loss)
    with open(os.path.join(output_dir, f"{name}_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
    a = np.asarray(save_recalls)
    with open(os.path.join(output_dir, f"{name}_recall.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")
    return test_loss


def train(
        args,
        model,
        train_loader,
        optimizer,
        epoch,
        output_dir="./",
        config={"loss": "cross-entropy"},
):
    save_loss = []
    total_loss = 0
    model.train()
    loss_fn = BinaryFocalLoss(alpha=[0.25, 0.75], gamma=2, reduction="mean")
    for batch_idx, data in enumerate(train_loader):
        image, source, labels, names = (
            data["images"].cuda(non_blocking=True),
            data["sources"].cuda(non_blocking=True),
            data["labels"].cuda(non_blocking=True),
            data["names"]
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
        optimizer.step()

        save_loss.append(loss.item())
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {}\tLoss: {:.6f} \t Average loss {:.6f}".format(
                    epoch, loss.item(), np.mean(save_loss[-args.log_interval:])
                )
            )
    a = np.asarray(save_loss)
    with open(os.path.join(output_dir, "train_loss.csv"), "ab") as f:
        np.savetxt(f, a, delimiter=",")


def setup(args, single):
    """
    Setup dataset and dataloaders for these new datasets
    """
    train_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}.pkl"),
        single_source_per_img=single,
        shuffle=args.shuffle,
        norm=not args.norm,
    )
    train_test_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}.pkl"),
        single_source_per_img=single,
        shuffle=args.shuffle,
        norm=not args.norm,
    )
    val_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_val_norm{args.norm}.pkl"),
        single_source_per_img=single,
        shuffle=args.shuffle,
        norm=not args.norm,
    )
    return train_dataset, train_test_dataset, val_dataset


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)

    # Generate model

    config = {
        "act": "relu",
        "fc_out": 256,
        "fc_final": 256,
        "single": args.single,
        "loss": args.loss,
    }

    train_dataset, train_test_dataset, val_dataset = setup(args, config["single"])

    if config["single"]:
        model = RadioSingleSourceModel(1, 11, config=config)
    else:
        model = RadioMultiSourceModel(1, args.classes, config=config)

    # generate optimizers
    optimizer_name = "Adam"
    lr = 1e-5
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_variable_fn,
        drop_last=True,
        sampler=train_sampler
    )
    train_test_sampler = torch.utils.data.distributed.DistributedSampler(
        train_test_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_test_loader = dataloader.DataLoader(
        train_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=train_test_sampler
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    test_loader = dataloader.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=test_sampler
    )
    experiment_name = (
            args.experiment
            + f"_lr{lr}_b{args.batch}_single{config['single']}_sources{args.num_sources}_norm{args.norm}_loss{config['loss']}"
    )
    if environment == "XPS":
        output_dir = os.path.join("/home/jacob/", "reports", experiment_name)
    else:
        output_dir = os.path.join("/home/s2153246/data/", "reports", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[gpu])
    print("Model created")
    for epoch in range(args.epochs):
        train(args, model, train_loader, optimizer, epoch, output_dir, config)
        test(args, model, train_test_loader, epoch, "Train_test", output_dir, config)
        test(args, model, test_loader, epoch, "Test", output_dir, config)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.world_size = args.gpus * args.nodes
    print("Command Line Args:", args)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(main, nprocs=args.gpus, args=(args,))