import time
from pathlib import Path

import numpy as np
import torch
import multiprocessing as mp

from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from omegaconf import OmegaConf
from ultralytics import YOLO


class DefectDetector:
    def __init__(self, config_path, yolo_path="yolov8n.pt", device="auto", root="."):
        self.root = Path(root)
        self.config_path = config_path
        self.yolo_path = yolo_path

        self.device = torch.device(
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )

        self.defect_predictor = None
        self.transform = None
        self.detector = None

    def _build_defect_model(self):
        config = OmegaConf.load(self.config_path)
        config.visualization.show_images = False
        config.visualization.save_images = False
        config.visualization.log_images = False
        config.visualization.image_save_path = None

        config.trainer.resume_from_checkpoint = str(self.root / Path(config.project.path) / "weights/lightning/model.ckpt")
        weight_path = config.trainer.resume_from_checkpoint

        self.defect_predictor = get_model(config)
        self.defect_predictor.load_state_dict(torch.load(weight_path, map_location=self.device)["state_dict"])
        self.defect_predictor.to(self.device)
        self.defect_predictor.eval()

        # get the transforms
        transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
        image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
        center_crop = config.dataset.get("center_crop")

        if center_crop is not None:
            center_crop = tuple(center_crop)

        normalization = InputNormalizationMethod(config.dataset.normalization)
        self.transform = get_transforms(
            config=transform_config,
            image_size=image_size,
            center_crop=center_crop,
            normalization=normalization
        )

    def build(self):
        self._build_defect_model()

        self.detector = YOLO(self.yolo_path)
        self.detector(verbose=False, device=self.device)

    def pre_process(self, image: np.ndarray) -> torch.Tensor:

        processed_image = self.transform(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image.to(self.device)


class MultiProcessDefectDetector(DefectDetector):
    def __init__(self, config_path, yolo_path="yolov8n.pt", device="auto", num_processes=4):
        super().__init__(config_path, yolo_path=yolo_path, device=device)
        self.num_processes = num_processes

        self.input_yolo_queue = [mp.Queue() for _ in range(num_processes)]
        self.input_defect_queue = [mp.Queue() for _ in range(num_processes)]
        self.result_defect_queue = [mp.Queue() for _ in range(num_processes)]

    def start(self):
        # Create and start worker processes
        yolo_processes = []
        for _ in range(self.num_processes):
            p = mp.Process(target=self._yolo_worker)
            p.start()
            yolo_processes.append(p)

        prediction_processes = []
        for _ in range(self.num_processes):
            p = mp.Process(target=self._predictor_worker)
            p.start()
            prediction_processes.append(p)

        # Wait for all worker processes to finish
        # for p in yolo_processes:
        #     p.join()
        #
        # # Wait for all worker processes to finish
        # for p in prediction_processes:
        #     p.join()

    def _yolo_worker(self):
        while not self.input_yolo_queue.empty():
            task_id, input_data = self.input_yolo_queue.get(timeout=1)
            # Perform the task with the input data
            result = self._process_yolo(task_id, input_data)
            # Put the result in the output queue
            self.input_defect_queue.put((time.time(), result))

    def _predictor_worker(self):
        while not self.input_defect_queue.empty():
            task_id, input_data = self.input_defect_queue.get(timeout=1)
            # Perform the task with the input data
            result = self._process_prediction(task_id, input_data)
            # Put the result in the output queue
            self.result_defect_queue.put((time.time(), result))

    def process(self, frame):
        self.input_yolo_queue.put((time.time(), frame))

    def _process_yolo(self, task_id, input_data):
        # Placeholder for the actual task processing
        return f"Processed data for task {task_id}: {input_data}"

    def _process_prediction(self, task_id, input_data):
        # Placeholder for the actual task processing
        return f"Processed data for task {task_id}: {input_data}"

    def get(self):
        return self.result_defect_queue.get()
