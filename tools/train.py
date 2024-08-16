"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="patchcore", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def evaluate_results(predictions, output_dir: Path = None):
    labels = []
    preds = []
    for prediction in predictions:
        label = prediction["label"].numpy().astype(int).tolist()
        pred = prediction["pred_labels"].numpy().astype(int).tolist()
        labels.extend(label)
        preds.extend(pred)

        if output_dir is not None:
            for y_true, y_pred, path in zip(prediction["label"], prediction["pred_labels"], prediction["image_path"]):
                y_true, y_pred = int(y_true), int(y_pred)
                if y_true == 1 and y_pred == 1:
                    out = "TP"
                elif y_true == 1 and y_pred == 0:
                    out = "FN"
                elif y_true == 0 and y_pred == 1:
                    out = "FP"
                elif y_true == 0 and y_pred == 0:
                    out = "TN"
                else:
                    raise Exception("ERROR")

                output_path = output_dir / out / Path(path).name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, output_path)

    report = classification_report(labels, preds, target_names=["Good", "Defect"])
    cm = confusion_matrix(labels, preds)
    print(f"Classification Report:")
    print(report)
    print("Confusion Matrix")
    print(cm)
    return {"Classification Report": report, "Confusion Matrix": cm}


def write_log(results, root_dir, version):
    with open(Path(root_dir) / f"version_{version}/report.txt", "w") as f:
        for name, result in results.items():
            f.write(name + "\n\n")
            f.write(str(result) + "\n\n")


def train(model_name="patchcore", config_path=None, log_level="INFO"):
    configure_logger(level=log_level)

    if log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=model_name, config_path=config_path)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)
        dataloader = DataLoader(
            datamodule.test_data,
            batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers
        )
        predictions = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True)
        results = evaluate_results(
            predictions,
            Path(config.project.path) / "outputs"
        )
        write_log(results, experiment_logger[0].root_dir, experiment_logger[0].version)


# python tools/inference/openvino_inference.py  --weights results/kamera-atas/run.2024-06-29_14-31-45/weights/openvino/model.bin  --metadata results/kamera-atas/run.2024-06-29_14-31-45/weights/openvino/metadata.json --input "D:\InspectionResult\SpringSheetMetal\images\2024-06-23_23-10-43\kamera-atas\train\good\object-0_2024-06-23_23-28-45-260345.jpg"      --output results/kamera-atas/run.2024-06-29_14-31-45/images

if __name__ == "__main__":
    args_ = get_parser().parse_args()
    train(args_.model, args_.config, args_.log_level)
