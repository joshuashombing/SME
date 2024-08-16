# import the necessary packages
import multiprocessing as mp
import time
from threading import Thread, Event
from typing import Union

import numpy as np

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.pre_processing.utils import resize_image, draw_bbox
from config import (
    PREPROCESSOR_PARAMS_TOP_CAMERA,
    PREPROCESSOR_PARAMS_SIDE_CAMERA,
    INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA, TIME_TO_PUSH_OBJECT
)
from relay import Relay


def log_relay_push(index, obj):
    current_time = time.time()
    delay = current_time - obj['to_push_at']
    # Format and print the messages
    print(f"Push object {index} at scheduled time {obj['to_push_at']}")
    print(f"Actual push time for object {index}: {current_time}")
    print(f"Delay in pushing object {index}: {delay:.2f} seconds")

    # Check and print running time if available
    if "last_inspected_at" in obj and "start_tracked_at" in obj:
        running_time = obj['last_inspected_at'] - obj['start_tracked_at']
        print(f"Running time for object {index}: {running_time:.2f} seconds")


def _is_object_defect(defects, scores):
    """
    Determines if an object is defective.

    :param defects: List or array-like structure with boolean values indicating the presence of defects.
    :param scores: List or array-like structure with scores corresponding to the defects (not used).
    :return: bool - True if the object is defective. If there are 3 or more defect indicators,
                     returns True if any defect is present. Otherwise, returns True by default.
    """

    if len(defects) >= 3:
        return np.any(defects)
    return True


class AIEngine:
    def __init__(self):
        # initialize the AI Engine along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.frame_queue = mp.Queue()
        self.result_detection: Union[mp.Queue, None] = None
        self.patches_queue: Union[mp.Queue, None] = None
        self.predictions = {}
        self.processed_predictions = set()

        self.grab_thread_handle = None
        self.inspection_pre_process_handle = None

        self.num_inspection_thread_handles = 3
        self.inspection_thread_handles = []
        self.time_to_push_after_disappear = TIME_TO_PUSH_OBJECT  # in second

        self.prev_object_ids = []

        self.defect_event = Event()
        self.good_event = Event()

        self.relay = Relay()
        self.camera_id = -1

        self._preprocessor_params = {
            0: PREPROCESSOR_PARAMS_SIDE_CAMERA,
            1: PREPROCESSOR_PARAMS_TOP_CAMERA
        }

        self._inspector_params = {
            0: INSPECTOR_PARAMS_SIDE_CAMERA,
            1: INSPECTOR_PARAMS_TOP_CAMERA
        }

    @staticmethod
    def draw_result(frame, result):
        frame = resize_image(frame, width=640)

        if len(result) == 0:
            return frame

        for box, object_id in zip(result.boxes.xyxyn, result.boxes.track_id):
            draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

        return frame

    def start(self, camera_id):
        self.camera_id = camera_id
        self.result_detection = mp.Queue()
        self.inspection_pre_process_handle = Thread(target=self.pre_process_inspection, args=(camera_id,))
        self.inspection_pre_process_handle.daemon = True
        self.inspection_pre_process_handle.start()

        self.patches_queue = mp.Queue()
        for i in range(self.num_inspection_thread_handles):
            thread = Thread(target=self.inspect_object, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.inspection_thread_handles.append(thread)

        self.relay.open()
        return self

    def push_object(self):
        if not self.predictions:
            return

        object_ids = list(self.predictions.keys())
        for object_id in object_ids:
            push_at = self.predictions[object_id].get("to_push_at")
            if push_at is not None and time.time() >= push_at:
                obj = self.predictions.pop(object_id)
                if object_id in self.processed_predictions:
                    continue

                defects = obj["data"]["labels"]
                scores = obj["data"]["scores"]
                if _is_object_defect(defects, scores):
                    self.relay.write(self.camera_id)
                    self.defect_event.set()
                    log_relay_push(object_id, obj)
                else:
                    # If the object is deemed good, do not push
                    self.good_event.set()
                    print(f"Object {object_id} is in good condition, not pushing.")

                print("Defects", defects)
                print("Scores", scores)
                print("="*50)

                self.processed_predictions.add(object_id)

    def process(self, frame, result):
        for i, object_id in enumerate(result.boxes.track_id):
            if object_id not in self.predictions:
                self.predictions[object_id] = {
                    "data": {
                        "labels": [],
                        "scores": [],
                    },
                    "start_tracked_at": time.time()
                }

        current_object_ids = result.boxes.track_id
        disappear_object_ids = set(self.prev_object_ids) - set(current_object_ids)
        if len(disappear_object_ids) > 0:
            for object_id in disappear_object_ids:
                if object_id in self.predictions:
                    self.predictions[object_id]["to_push_at"] = time.time() + self.time_to_push_after_disappear
        self.prev_object_ids = current_object_ids

        if self.result_detection is not None:
            self.result_detection.put((frame, result))

    def pre_process_inspection(self, camera_id):
        params = self._preprocessor_params[camera_id]
        pre_processor = DefectPreprocessor(**params)

        print("Start preprocessing defect...")
        while True:
            if self.stopped:
                break

            if self.result_detection is None or self.result_detection.empty():
                continue

            try:
                detection_result = self.result_detection.get_nowait()
            except Exception as e:
                print(e)
            else:
                frame, result = detection_result
                patches = pre_processor.object_post_process(frame, result)
                if self.patches_queue is not None and len(patches) > 0:
                    self.patches_queue.put(patches)

                del detection_result

        del pre_processor
        print("Stop preprocessing defect...")

    def inspect_object(self, camera_id):
        params = self._inspector_params[camera_id]
        model = DefectPredictor(**params).build()

        print("Start inspection...")
        while True:
            if self.stopped:
                break

            if self.patches_queue is None or self.patches_queue.empty():
                continue

            try:
                detections = self.patches_queue.get_nowait()
            except Exception as e:
                pass
            else:
                if detections is not None:
                    centers, object_ids, frames = zip(*detections)
                    # print(object_ids, centers)
                    result = model.predict(frames)
                    if result is not None:
                        scores = result["pred_scores"].cpu().numpy()
                        labels = result["pred_labels"].cpu().numpy()
                        for object_id, score, label in zip(object_ids, scores, labels):
                            if object_id in self.predictions.keys():
                                self.predictions[object_id]["last_inspected_at"] = time.time()
                                self.predictions[object_id]["data"]["scores"].append(score)
                                self.predictions[object_id]["data"]["labels"].append(label)

                del detections

        del model
        print("Stop inspection...")

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.grab_thread_handle is not None:
            # print("before self.grab_thread_handle.join()")
            self.grab_thread_handle.join()
            # print("after self.grab_thread_handle.join()")

        if self.inspection_pre_process_handle is not None:
            # print("before self.inspection_pre_process.join()")
            self.inspection_pre_process_handle.join()
            # print("after self.inspection_pre_process.join()")

        for t in self.inspection_thread_handles:
            t.join()

        if self.result_detection is not None:
            while not self.result_detection.empty():
                try:
                    self.result_detection.get_nowait()
                except Exception as e:
                    print(e)

            self.result_detection.close()
            self.result_detection.join_thread()

        if self.patches_queue is not None:
            while not self.patches_queue.empty():
                try:
                    self.patches_queue.get_nowait()
                except Exception as e:
                    print(e)

            self.patches_queue.close()
            self.patches_queue.join_thread()
