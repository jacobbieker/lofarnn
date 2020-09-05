import os
from lofarnn.models.dataloaders.datasets import collate_variable_fn
from lofarnn.models.base.utils import default_argument_parser, setup, train, test

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
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch
import optuna


def objective(trial):
    # Generate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "act": trial.suggest_categorical("activation", ["relu", "elu", "leaky"]),
        "fc_out": trial.suggest_int("fc_out", 8, 256),
        "fc_final": trial.suggest_int("fc_final", 8, 256),
        "alpha_1": trial.suggest_uniform("alpha_1", 0.01, 0.99),
        "gamma": trial.suggest_int("gamma", 0, 9),
        "single": args.single,
        "loss": args.loss,
    }
    config["alpha_2"] = 1.0 - config["alpha_1"]

    train_dataset, train_test_dataset, val_dataset = setup(args)

    if config["single"]:
        model = RadioSingleSourceModel(1, 12, config=config).to(device)
    else:
        model = RadioMultiSourceModel(1, args.classes, config=config).to(device)

    # generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    if args.lr_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    elif args.lr_type == "cyclical":
        scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=0.1 if lr < 0.1 else 10 * lr)
    else:
        scheduler = None

    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_variable_fn,
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
    experiment_name = (
        args.experiment
        + f"_lr{lr}_b{args.batch}_single{config['single']}_sources{args.num_sources}_norm{args.norm}_loss{config['loss']}_scheduler{args.lr_type}"
    )
    if environment == "XPS":
        output_dir = os.path.join("/home/jacob/", "reports", experiment_name)
    else:
        output_dir = os.path.join("/home/s2153246/data/", "reports", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Model created")
    try:
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
            accuracy = test(
                args, model, device, test_loader, epoch, "Test", output_dir, config
            )
            if epoch % 5 == 0:  # Save every 5 epochs
                torch.save(model, os.path.join(output_dir, "model.pth"))

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    except:
        # Failure, like NaN loss or out of memory errors
        raise optuna.exceptions.TrialPruned()

    return accuracy


def main(args):
    if environment == "XPS":
        db = os.path.join(
            "/home/jacob/Development/lofarnn", f"lotss_dr2_{args.single}_{args.loss}.db"
        )
    else:
        db = os.path.join(
            "/home/s2153246/data/", f"lotss_dr2_{args.single}_{args.loss}.db"
        )
    study = optuna.create_study(
        study_name=args.experiment,
        direction="maximize" if args.single else "minimize",
        storage="sqlite:///" + db,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto"),
    )
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
    main(args)
