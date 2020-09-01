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
from lofarnn.models.base.utils import default_argument_parser, setup, train, test
from torch.utils.data import dataset, dataloader
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import optuna


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def objective(trial):
    # Generate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "act": trial.suggest_categorical("activation", ["relu", "elu", "leaky"]),
        "fc_out": trial.suggest_int("fc_out", 8, 256),
        "fc_final": trial.suggest_int("fc_final", 8, 256),
        "single": args.single,
        "loss": args.loss,
    }

    train_dataset, train_test_dataset, val_dataset = setup(args)

    if config["single"]:
        model = RadioSingleSourceModel(1, 11, config=config).to(device)
    else:
        model = RadioMultiSourceModel(1, args.classes, config=config).to(device)

    # generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
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
        rank=args.rank
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
        rank=args.rank
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
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.gpu])
    print("Model created")
    try:
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch, output_dir, config)
            test(args, model, device, train_test_loader, epoch, "Train_test", output_dir, config)
            accuracy = test(args, model, device, test_loader, epoch, "Test", output_dir, config)
            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    except:
        # Failure, like NaN loss or out of memory errors
        raise optuna.exceptions.TrialPruned()

    return accuracy


def main(gpu, args):
    if environment == "XPS":
        db = os.path.join("/home/jacob/", "reports", f"lotss_dr2_{args.single}_{args.loss}.db")
    else:
        db = os.path.join("/home/s2153246/data/", f"lotss_dr2_{args.single}_{args.loss}.db")
    study = optuna.create_study(
        study_name=args.experiment,
        direction="minimize",
        storage="sqlite:///" + db,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto"),
    )
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)
    args.gpu = gpu
    args.rank = rank
    study.optimize(objective, n_trials=args.num_trials)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig = optuna.visualization.plot_param_importances(study=study)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    mp.spawn(main, nprocs=args.gpus, args=(args,))