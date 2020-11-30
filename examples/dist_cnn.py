import os
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
from lofarnn.models.base.utils import default_argument_parser, setup, test, train
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import dataset, dataloader
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )
    torch.cuda.set_device(gpu)

    # Generate model
    config = {
        "act": "leaky",
        "fc_out": 186,
        "fc_final": 136,
        "single": args.single,
        "loss": args.loss,
        "gamma": 2,
        "alpha_1": 0.12835728,
    }

    train_dataset, train_test_dataset, val_dataset = setup(args)

    if config["single"]:
        model = RadioSingleSourceModel(1, 12, config=config)
    else:
        model = RadioMultiSourceModel(1, args.classes, config=config)

    # generate optimizers
    optimizer_name = "Adam"
    lr = 0.00057
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    if args.lr_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    elif args.lr_type == "cyclical":
        scheduler = CyclicLR(
            optimizer, base_lr=args.lr, max_lr=0.1 if args.lr < 0.1 else 10 * args.lr
        )
    else:
        scheduler = None
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_variable_fn,
        drop_last=True,
        sampler=train_sampler,
    )
    train_test_sampler = torch.utils.data.distributed.DistributedSampler(
        train_test_dataset, num_replicas=args.world_size, rank=rank
    )
    train_test_loader = dataloader.DataLoader(
        train_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=train_test_sampler,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=rank
    )
    test_loader = dataloader.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=test_sampler,
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    print("Model created")
    for epoch in range(args.epochs):
        train(
            args, model, train_loader, optimizer, scheduler, epoch, output_dir, config
        )
        test(args, model, train_test_loader, epoch, "Train_test", output_dir, config)
        test(args, model, test_loader, epoch, "Test", output_dir, config)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.world_size = args.gpus * args.nodes
    print("Command Line Args:", args)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(main, nprocs=args.gpus, args=(args,))
