import os

from lofarnn.models.base.baselines import closest_point_model  # , flux_weighted_model

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]

from lofarnn.models.base.utils import default_argument_parser, setup
from torch.utils.data import dataloader
import torch
import pickle


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
    experiment_name = "baselines"
    if environment == "XPS":
        output_dir = os.path.join("/home/jacob/", "reports", experiment_name)
    else:
        output_dir = os.path.join("/home/s2153246/data/", "reports", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    named_recalls = {}
    flux_named_recall = {}

    with torch.no_grad():
        for data in test_loader:
            image, source, labels, names = (
                data["images"],
                data["sources"],
                data["labels"],
                data["names"],
            )
            output = closest_point_model(source)
            # out2 = flux_weighted_model(names)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            pred = torch.add(pred, 1)
            label = labels.argmax(dim=1, keepdim=True)
            print(pred)
            print(label)
            # pred2 = out2.numpy()
            # pred2 += 1 # Adds the 1 because of the 0 size default
            # Now get named recall ones
            if not args.single:
                for i in range(len(names)):
                    # Assumes testing is with batch size of 1
                    named_recalls[names[i]] = pred.eq(label.view_as(pred)).sum().item()
            else:
                for i in range(len(names)):
                    if (
                        label.item() == 0
                    ):  # Label is source, don't care about the many negative examples
                        if pred.item() == 0:  # Prediction is source
                            named_recalls[names[i]] = 1  # Value is correct
                        else:  # Prediction is not correct
                            named_recalls[names[i]] = 0  # Value is incorrect
    pickle.dump(
        named_recalls,
        open(os.path.join("./", f"final_test_closest_baseline_recall.pkl"), "wb"),
    )
    pickle.dump(
        flux_named_recall,
        open(os.path.join("./", f"final_test_flux_baseline_recall.pkl"), "wb"),
    )

    named_recalls = {}
    flux_named_recall = {}
    with torch.no_grad():
        for data in test_loader:
            image, source, labels, names = (
                data["images"],
                data["sources"],
                data["labels"],
                data["names"],
            )
            output = closest_point_model(source)
            # out2 = flux_weighted_model(names)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            label = labels.argmax(dim=1, keepdim=True)
            pred2 = out2.numpy()
            pred2 += 1  # Adds the 1 because of the 0 size default
            # Now get named recall ones
            if not args.single:
                for i in range(len(names)):
                    # Assumes testing is with batch size of 1
                    named_recalls[names[i]] = pred.eq(label.view_as(pred)).sum().item()
                    if pred2 == label.numpy():
                        flux_named_recall[names[i]] = 1
                    else:
                        flux_named_recall[names[i]] = 0
            else:
                for i in range(len(names)):
                    if (
                        label.item() == 0
                    ):  # Label is source, don't care about the many negative examples
                        if pred.item() == 0:  # Prediction is source
                            named_recalls[names[i]] = 1  # Value is correct
                        else:  # Prediction is not correct
                            named_recalls[names[i]] = 0  # Value is incorrect
            #           if pred2.item() == 0:
            #               flux_named_recall[names[i]] = 1
            #           else:
            #               flux_named_recall[names[i]] = 0
    pickle.dump(
        named_recalls,
        open(os.path.join(output_dir, f"test_closest_baseline_recall.pkl"), "wb"),
    )
    pickle.dump(
        flux_named_recall,
        open(os.path.join(output_dir, f"test_flux_baseline_recall.pkl"), "wb"),
    )

    with torch.no_grad():
        for data in train_test_loader:
            image, source, labels, names = (
                data["images"],
                data["sources"],
                data["labels"],
                data["names"],
            )
            output = closest_point_model(source)
            # out2 = flux_weighted_model(source)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            label = labels.argmax(dim=1, keepdim=True)
            pred2 = out2.numpy()
            pred2 += 1  # Adds the 1 because of the 0 size default
            # Now get named recall ones
            if not args.single:
                for i in range(len(names)):
                    # Assumes testing is with batch size of 1
                    named_recalls[names[i]] = pred.eq(label.view_as(pred)).sum().item()
                    if pred2 == label.numpy():
                        flux_named_recall[names[i]] = 1
                    else:
                        flux_named_recall[names[i]] = 0
            else:
                for i in range(len(names)):
                    if (
                        label.item() == 0
                    ):  # Label is source, don't care about the many negative examples
                        if pred.item() == 0:  # Prediction is source
                            named_recalls[names[i]] = 1  # Value is correct
                        else:  # Prediction is not correct
                            named_recalls[names[i]] = 0  # Value is incorrect
            #           if pred2.item() == 0:
            #               flux_named_recall[names[i]] = 1
            #           else:
            #               flux_named_recall[names[i]] = 0
    pickle.dump(
        named_recalls,
        open(os.path.join(output_dir, f"train_closest_baseline_recall.pkl"), "wb"),
    )
    pickle.dump(
        flux_named_recall,
        open(os.path.join(output_dir, f"train_flux_baseline_recall.pkl"), "wb"),
    )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
