# import the necessary packages
import multiprocessing as mp
import os
import time
from threading import Thread

import numpy as np

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.pre_processing.utils import resize_image, draw_bbox
from config import (
    PREPROCESSOR_PARAMS_TOP_CAMERA,
    PREPROCESSOR_PARAMS_SIDE_CAMERA,
    INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA, TIME_TO_PUSH_OBJECT, NUM_INSPECTION_PROCESS
)
from relay import Relay

relay = Relay()


def draw_result(frame, result):
    frame = resize_image(frame, width=500)

    if len(result) == 0:
        return frame

    for box, object_id in zip(result.boxes.xyxyn.cpu(), result.boxes.track_id):
        draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

    return frame


def clear_queue(q: mp.Queue):
    if q is None:
        return

    while not q.empty():
        try:
            q.get_nowait()
        except Exception as e:
            print(e)
    try:
        q.close()
        q.join_thread()
    except Exception as e:
        print(e)


def log_relay_push(camera_id, obj_id, obj):
    current_time = time.time()
    delay = current_time - obj['to_push_at']

    # Print the messages with camera_id included
    print(f"[Camera {camera_id}] Push object {obj_id} at scheduled time {obj['to_push_at']}")
    print(f"[Camera {camera_id}] Actual push time for object {obj_id}: {current_time}")
    print(f"[Camera {camera_id}] Delay in pushing object {obj_id}: {delay:.2f} seconds")

    # Check and print running time if available
    if "last_inspected_at" in obj and "start_tracked_at" in obj:
        running_time = obj['last_inspected_at'] - obj['start_tracked_at']
        print(f"[Camera {camera_id}] Running time for object {obj_id}: {running_time:.2f} seconds")


def _is_object_defect(defects, scores):
    """
    Determines if an object is defective.

    :param defects: List or array-like structure with boolean values indicating the presence of defects.
    :param scores: List or array-like structure with scores corresponding to the defects (not used).
    :return: bool - True if the object is defective. If there are 3 or more defect indicators,
                     returns True if any defect is present. Otherwise, returns True by default.
    """

    if len(defects) >= 3:
        return np.any(np.array(scores) >= 52)
    return True


class AIEngine:
    def __init__(self, camera_id):
        # initialize the AI Engine along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stopped = mp.Event()

        # initialize the queue used to store frames read from
        # the video file
        self.result_detection_queue = mp.Queue()
        self.patches_queue = mp.Queue()
        self.result_predictor_queue = mp.Queue()

        self.predictions = {}
        self.processed_predictions = set()

        self.num_inspection_processes = NUM_INSPECTION_PROCESS
        self.time_to_push_after_disappear = TIME_TO_PUSH_OBJECT  # in second
        # self.time_warm_start = 5  # in seconds

        self.prev_object_ids = []

        self.defect_event = mp.Event()
        self.good_event = mp.Event()

        self.camera_id = camera_id

        self._preprocessor_params = {
            0: PREPROCESSOR_PARAMS_SIDE_CAMERA,
            1: PREPROCESSOR_PARAMS_TOP_CAMERA
        }

        self._inspector_params = {
            0: INSPECTOR_PARAMS_SIDE_CAMERA,
            1: INSPECTOR_PARAMS_TOP_CAMERA
        }

    def start(self):
        process_result_work_thread = Thread(target=self.process_result_work_thread)
        process_result_work_thread.daemon = True
        process_result_work_thread.start()

        inspection_processes = [
            mp.Process(
                target=self.inspect_object,
                args=(self.patches_queue, self.result_predictor_queue)
            ) for _ in range(self.num_inspection_processes)
        ]
        for process in inspection_processes:
            process.start()

        inspection_pre_process = mp.Process(
            target=self.pre_process_inspection, args=(self.result_detection_queue, self.patches_queue)
        )
        inspection_pre_process.start()

        # time.sleep(self.time_warm_start)
        relay.open()
        return inspection_processes + [process_result_work_thread, inspection_pre_process]

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
                    relay.write(self.camera_id)
                    self.defect_event.set()
                    log_relay_push(self.camera_id, object_id, obj)
                else:
                    # If the object is deemed good, do not push
                    self.good_event.set()
                    print(f"Object {object_id} is in good condition, not pushing.")

                print("Defects", defects)
                print("Scores", scores)
                print("=" * 50)

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

        if self.result_detection_queue is not None:
            self.result_detection_queue.put((frame, result))

    def process_result_work_thread(self):
        while True:
            if self.stopped.is_set() and self.result_predictor_queue.empty():
                break

            try:
                if self.result_predictor_queue.empty():
                    continue

                result = self.result_predictor_queue.get(timeout=0.5)

                if result is None:
                    continue

                object_id, label, score, timestamp = result
                if object_id in self.predictions:
                    self.predictions[object_id]["last_inspected_at"] = timestamp
                    self.predictions[object_id]["data"]["scores"].append(score)
                    self.predictions[object_id]["data"]["labels"].append(label)

                # object_ids, labels, scores, timestamp = result

                # for object_id, label, score in zip(object_ids, labels, scores):
                #     if object_id in self.predictions.keys():
                        # self.predictions[object_id]["last_inspected_at"] = timestamp
                        # self.predictions[object_id]["data"]["scores"].append(score)
                        # self.predictions[object_id]["data"]["labels"].append(label)
            except Exception as e:
                pass
                print("Error:", e)

    def pre_process_inspection(self, result_detection_queue: mp.Queue, patches_queue: mp.Queue):
        print(f"[Process {os.getpid()}] Start preprocessing defect..")
        params = self._preprocessor_params[self.camera_id]
        pre_processor = DefectPreprocessor(**params)
        while True:
            if self.stopped.is_set():
                break

            if result_detection_queue is None or result_detection_queue.empty():
                continue

            try:
                detection_result = result_detection_queue.get(timeout=0.5)
                if detection_result is not None:
                    frame, result = detection_result
                    patches = pre_processor.object_post_process(frame, result)
                    for patch in patches:
                    # if patches_queue is not None and len(patches) > 0:
                        patches_queue.put(patch)

                    del detection_result
            except Exception as e:
                pass
                print("Error:", e)

        del pre_processor
        print(f"[Process {os.getpid()}] Stop preprocessing defect...")

    def inspect_object(self, patches_queue: mp.Queue, result_queue: mp.Queue):
        print(f"[Process {os.getpid()}] Start inspecting object...")
        params = self._inspector_params[self.camera_id]
        model = DefectPredictor(**params).build()

        while True:
            if self.stopped.is_set():
                break

            if patches_queue is None or patches_queue.empty():
                continue

            try:
                detection = patches_queue.get(timeout=0.5)
                if detection is not None:
                    # center, object_ids, frames = zip(*detections)
                    center, object_id, cropped = detection
                    result = model.predict([cropped])
                    if result is not None:
                        scores = result["pred_scores"].cpu().numpy()
                        labels = result["pred_labels"].cpu().numpy()
                        result_queue.put((object_id, labels[0], scores[0], time.time()))
                del detection

            except Exception as e:
                pass
                print("Error:", e)

        del model
        print(f"[Process {os.getpid()}] Stop inspection...")

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped.set()
        clear_queue(self.result_detection_queue)
        clear_queue(self.patches_queue)
        clear_queue(self.result_predictor_queue)
