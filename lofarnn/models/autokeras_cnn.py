import tensorflow as tf
import autokeras as ak
import os
from lofarnn.models.dataloaders.datasets import RadioSourceDataset, collate_autokeras_fn

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from lofarnn.models.base.utils import default_argument_parser
from torch.utils.data import dataloader


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
        os.path.join(args.dataset, f"cnn_train_test_norm{args.norm}.pkl"),
        single_source_per_img=args.single,
        shuffle=True,
    )
    val_dataset = RadioSourceDataset(
        os.path.join(args.dataset, f"cnn_val_norm{args.norm}.pkl"),
        single_source_per_img=args.single,
        shuffle=True,
    )
    return train_dataset, train_test_dataset, val_dataset


def main(args):
    train_dataset, train_test_dataset, val_dataset = setup(args)
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=2000, shuffle=False, collate_fn=collate_autokeras_fn
    )
    test_loader = dataloader.DataLoader(
        val_dataset, batch_size=1000, shuffle=False, collate_fn=collate_autokeras_fn
    )

    data = next(train_loader)
    image, source, labels, names = (
        data["images"].numpy(),
        data["sources"].numpy(),
        data["labels"].numpy(),
        data["names"].numpy(),
    )
    val_data = next(test_loader)
    val_image, val_source, val_labels, val_names = (
        val_data["images"].numpy(),
        val_data["sources"].numpy(),
        val_data["labels"].numpy(),
        val_data["names"].numpy(),
    )

    model = ak.AutoModel(
        inputs=[
            ak.ImageInput(),
            ak.StructuredDataInput(
                column_names=[
                    "angle",
                    "separation",
                    "iFApMag",
                    "w1Mag",
                    "gFApMag",
                    "rFApMag",
                    "zFApMag",
                    "yFApMag",
                    "w2Mag",
                    "w3Mag",
                    "w4Mag",
                ]
            ),
        ],
        outputs=[
            ak.ClassificationHead(loss="categorical_crossentropy", metrics=["accuracy"])
        ],
        overwrite=True,
        max_trials=50,
    )

    # Fit the model with prepared data.
    image_val = val_image
    structured_val = val_source
    classification_val = val_labels

    image_data = image
    structured_data = source
    classification_target = labels

    model.fit(
        [image_data, structured_data],
        [classification_target],
        # Use your own validation set.
        validation_data=([image_val, structured_val], [classification_val]),
        epochs=10,
    )

    keras_model = model.export_model()

    print(type(keras_model))  # <class 'tensorflow.python.keras.engine.training.Model'>
    print(keras_model)

    try:
        keras_model.save("model_autokeras", save_format="tf")
    except:
        keras_model.save("model_autokeras.h5")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
