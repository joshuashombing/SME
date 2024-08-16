"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""
from argparse import ArgumentParser
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks, ModelCheckpoint, ImageVisualizerCallback, MetricVisualizerCallback
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a model config file")

    return parser


def evaluate_step(config):
    if config.project.seed:
        seed_everything(config.project.seed)

    config.trainer.resume_from_checkpoint = str(Path(config.project.path) / "weights/lightning/model.ckpt")

    datamodule = get_datamodule(config)
    model = get_model(config)
    callbacks = get_callbacks(config)

    callbacks_to_remove = [ImageVisualizerCallback, MetricVisualizerCallback, ModelCheckpoint]
    callbacks = [
        callback
        for callback in callbacks
        if not any(isinstance(callback, cb_remove) for cb_remove in callbacks_to_remove)
    ]

    # force disable intermediate checkpointing
    config.trainer.enable_checkpointing = False
    datamodule.setup()
    dataloader = DataLoader(
        datamodule.test_data,
        batch_size=config.dataset.eval_batch_size,
        num_workers=config.dataset.num_workers
    )

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    predictions = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)

    return predictions


def evaluate_single_model(config_path):
    model_config = OmegaConf.load(config_path)
    print(f"Evaluating {model_config.dataset.category} ...")
    predictions = evaluate_step(model_config)

    labels = []
    preds = []
    for prediction in predictions:
        label = prediction["label"].numpy().astype(int).tolist()
        pred = prediction["pred_labels"].numpy().astype(int).tolist()
        labels.extend(label)
        preds.extend(pred)

    print(f"Classification Report {model_config.dataset.category}:")
    print(classification_report(labels, preds, target_names=["Good", "Defect"]))
    print("Confusion Matrix")
    print(confusion_matrix(labels, preds))


def evaluate_many():
    project_path = Path(__file__).parent.parent

    all_labels = []
    all_predictions = []

    for path in (project_path / "configs").iterdir():
        print(path)
        model_config = OmegaConf.load(path)
        print(f"Evaluating {model_config.dataset.category} ...")
        predictions = evaluate_step(model_config)

        labels = []
        preds = []
        for prediction in predictions:
            label = prediction["label"].numpy().astype(int).tolist()
            pred = prediction["pred_labels"].numpy().astype(int).tolist()
            labels.extend(label)
            preds.extend(pred)

        all_labels.extend(labels)
        all_predictions.extend(preds)
        print(f"Classification Report {model_config.dataset.category}:")
        print(classification_report(labels, preds, target_names=["Good", "Defect"]))
        print("Confusion Matrix")
        print(confusion_matrix(labels, preds))

    print(f"Classification Report Overall:")
    print(classification_report(all_labels, all_predictions, target_names=["Good", "Defect"]))
    print("Confusion Matrix")
    print(confusion_matrix(all_labels, all_predictions))


if __name__ == "__main__":
    # evaluate_many()
    # conf_path = "results/patchcore/mvtec/spring_sheet_metal/run.2024-05-24_01-54-16/config.yaml"
    args = get_parser().parse_args()
    evaluate_single_model(args.config)
