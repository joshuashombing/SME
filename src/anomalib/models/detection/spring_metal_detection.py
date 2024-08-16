import threading
from collections import OrderedDict
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Union

import cv2
import numpy as np
import torch
from skimage.transform import rotate
from ultralytics import YOLO

from anomalib.models.detection.object_tracking import ObjectTracker, Direction
from anomalib.pre_processing.utils import resize_image
from .utils import (
    select_closest_bbox,
    transform_mask_to_rect,
    get_rect_angle_rotation,
    expand_bbox,
    unormalize_bbox,
    center_crop, remove_close_centers
)


class DetectionResult:
    def __init__(self, result):
        self.boxes = result.boxes.xyxy.cpu().numpy()
        self.boxes_n = result.boxes.xyxyn.cpu().numpy()
        self.masks = result.masks.xy if result.masks is not None else None
        self.track_ids = []
        self._sort_x()
        del result

    def _sort_x(self):
        centers_x = [(xyxy[0] + xyxy[2]) / 2 for xyxy in self.boxes]
        if len(centers_x) > 1:
            sorted_indices = np.argsort(centers_x)
            self.boxes = self.boxes[sorted_indices]
            self.boxes_n = self.boxes_n[sorted_indices]
            if self.masks is not None:
                self.masks = [self.masks[i] for i in sorted_indices]


class SpringMetalDetector:
    def __init__(
            self, path, distance_thresholds=(0.4, 0.5), conf_threshold=0.5, output_patch_shape=(680, 560),
            expand_bbox_percentage=0.2, device="auto", transform: Callable = None, camera_name="kamera-atas"
    ):

        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.path = path

        self.distance_thresholds = distance_thresholds
        self.conf_threshold = conf_threshold
        self.output_patch_shape = output_patch_shape
        self.expand_bbox_percentage = expand_bbox_percentage
        self.camera_name = camera_name

        self.transform = transform
        self.model: Union[YOLO, None] = None

        self.tracker = None

        self.patches = OrderedDict()

    @staticmethod
    def pre_process(image):
        image = resize_image(image, width=640)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def build(self):
        self.model = YOLO(self.path)
        self.model(verbose=False)
        self.tracker = ObjectTracker(direction=Direction.RIGHT_TO_LEFT)
        return self

    # @timeit
    def predict(self, image, **kwargs) -> dict:
        image = self.pre_process(image)

        if self.transform is not None:
            image = self.transform(image)

        result = self.model.predict(
            image,
            device=self.device,
            verbose=False,
            stream=False,
            agnostic_nms=True,
            **kwargs
        )[0]
        result = select_closest_bbox(
            result,
            distance_thresholds=self.distance_thresholds,
            conf_threshold=self.conf_threshold
        )
        result = remove_close_centers(result, min_distance=20)
        result = DetectionResult(result)
        result = self.track(result)
        return result.__dict__

        # test_result = DetectionResult(result)
        # print(test_result.__dict__)
        # centers_x = [(xyxy[0] + xyxy[2]) / 2 for xyxy in result.boxes.xyxy.cpu().numpy()]
        # if len(centers_x) > 1:
        #     sorted_indices = np.argsort(centers_x)
        #     result = result[sorted_indices]

        # result = select_closest_bbox(
        #     result,
        #     distance_thresholds=self.distance_thresholds,
        #     conf_threshold=self.conf_threshold
        # )
        # print(result.masks)
        # result = self.track(result)
        # return result

    def track(self, result: DetectionResult):
        self.tracker.update(result.boxes)

        track_ids = []
        for box in result.boxes:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            distances = [
                self.tracker.calculate_distance((cx, cy), center)
                for center in self.tracker.objects.values()
            ]
            track_ids.append(list(self.tracker.objects.keys())[np.argmin(distances)])

        result.track_ids = track_ids
        return result


class SpringMetalDetectorThread:
    def __init__(
            self, path, distance_thresholds=(0.35, 0.5), conf_threshold=0.5, output_patch_shape=(680, 560),
            expand_bbox_percentage=0.2, device="auto", pre_processor: Callable = None, queue_size=128
    ):
        self.path = path
        self.distance_thresholds = distance_thresholds
        self.conf_threshold = conf_threshold
        self.output_patch_shape = output_patch_shape
        self.expand_bbox_percentage = expand_bbox_percentage
        self.device = device
        self.pre_processor = pre_processor

        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)

        self.thread_detect = None
        self.stopped = False

    def start(self):
        self.stopped = False
        self.thread_detect = threading.Thread(target=self.detect_objects, daemon=True)
        self.thread_detect.start()
        return self

    def stop(self):
        self.stopped = True
        if self.thread_detect:
            self.thread_detect.join()

    def process(self, image):
        self.frame_queue.put(image)

    def get_result(self):
        return self.result_queue.get()

    def detect_objects(self):
        model = SpringMetalDetector(
            path=self.path,
            distance_thresholds=self.distance_thresholds,
            conf_threshold=self.conf_threshold,
            output_patch_shape=self.output_patch_shape,
            expand_bbox_percentage=self.expand_bbox_percentage,
            device=self.device,
            transform=self.pre_processor
        ).build()

        while not self.stopped:
            try:
                frame = self.frame_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
                if frame is not None:
                    result = model.predict(frame)
                    patches = model.post_process(frame, result)
                    del frame
                    self.result_queue.put((result, patches))
            except Exception as e:
                print(f"Error in detect_objects: {e}")

        del model


if __name__ == "__main__":
    yolo_path = "runs/segment/train/weights/best.pt"
    cls = SpringMetalDetector(path=yolo_path)
    cls.build()
