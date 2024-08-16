# import the necessary packages
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from threading import Thread, Timer

import cv2
import numpy as np

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.pre_processing.utils import resize_image, draw_bbox, get_now_str, generate_unique_filename
from config import (
    PREPROCESSOR_PARAMS_TOP_CAMERA,
    PREPROCESSOR_PARAMS_SIDE_CAMERA,
    INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA, TIME_TO_PUSH_OBJECT, NUM_INSPECTION_PROCESS, DETECTOR_PARAMS_SIDE_CAMERA,
    DETECTOR_PARAMS_TOP_CAMERA, DELAY_SYSTEM, SAVE_IMAGE, RESULT_DIR
)
from relay import Relay

relay = Relay()

logger = logging.getLogger("AutoInspection")


def draw_result(frame, boxes, object_ids):
    frame = resize_image(frame, width=482)

    if len(boxes) == 0:
        return frame

    for box, object_id in zip(boxes, object_ids):
        draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

    return frame


def clear_queue(q: mp.Queue):
    if q is None:
        return

    while not q.empty():
        try:
            q.get_nowait()
        except Exception as e:
            logger.error(e)
    try:
        q.close()
        q.join_thread()
    except Exception as e:
        logger.error(e)


def log_relay_push(camera_id, obj_id, obj):
    current_time = time.perf_counter()
    delay = current_time - obj['to_push_at']

    # Print the messages with camera_id included
    logger.info(f"[Camera {camera_id}] Judge object {obj_id} at scheduled time {obj['to_push_at']}")
    logger.info(f"[Camera {camera_id}] Actual judge time for object {obj_id}: {current_time}")
    logger.info(f"[Camera {camera_id}] Delay in judging object {obj_id}: {delay:.6f} seconds")

    # Check and print running time if available
    if "last_inspected_at" in obj and "start_tracked_at" in obj:
        running_time = obj['last_inspected_at'] - obj['start_tracked_at']
        logger.info(f"[Camera {camera_id}] Running time for object {obj_id}: {running_time:.6f} seconds")


def _is_object_defect(defects, scores, threshold=None):
    """
    Determines if an object is defective.

    :param defects: List or array-like structure with boolean values indicating the presence of defects.
    :param scores: List or array-like structure with scores corresponding to the defects (not used).
    :return: bool - True if the object is defective. If there are 3 or more defect indicators,
                     returns True if any defect is present. Otherwise, returns True by default.
    """

    if len(defects) >= 3:
        if threshold is not None:
            result = (np.array(scores) > threshold).astype(int)
            return np.sum(result) >= 2

        return np.sum(np.array(defects).astype(int)) >= 2

    elif len(defects) == 2 and np.sum(np.array(defects).astype(int)) == 0:
        return False

    return True


def pre_process_inspection(stopped, model_params, result_detection_queue: mp.Queue, patches_queue: mp.Queue, reset_object_ids: mp.Event):
    logger.info(f"[Process {os.getpid()}] Start preprocessing defect..")

    pre_processor = DefectPreprocessor(**model_params)
    while True:
        if stopped.is_set():
            break

        if result_detection_queue is None:
            break

        if reset_object_ids.is_set():
            pre_processor.patches.clear()
            pre_processor.tracker.objects.clear()
            reset_object_ids.clear()

        try:
            detection_result = result_detection_queue.get()
            if detection_result is not None:
                frame, result = detection_result
                patches = pre_processor.object_post_process(frame, result)
                if len(patches) > 0:
                    patches_queue.put(patches)
                # for patch in patches:
                #     patches_queue.put(patch)

                del detection_result
        except Exception as e:
            logger.error(f"Error at pre_process_inspection: {e}")

    del pre_processor
    logger.info(f"[Process {os.getpid()}] Stop preprocessing defect...")


def inspect_object(stopped, model_params, patches_queue: mp.Queue, result_queue: mp.Queue, result_save_queue: mp.Queue):
    logger.info(f"[Process {os.getpid()}] Start inspection...")
    model = DefectPredictor(**model_params).build()

    while True:
        if stopped.is_set():
            break

        if patches_queue is None:
            break

        try:
            detections = patches_queue.get()
            if detections is not None:
                center, object_ids, frames = zip(*detections)
                # center, object_id, cropped = detection
                result = model.predict(frames)
                for object_id, label, score, frame in zip(object_ids, result["labels"], result["scores"], frames):
                    result_queue.put((object_id, label, score, time.perf_counter()))

                    if result_save_queue is not None:
                        result_save_queue.put((object_id, frame, label, score))
                # if result is not None:
                #     scores = result["pred_scores"].cpu().numpy()
                #     labels = result["pred_labels"].cpu().numpy()
                #     result_queue.put((object_id, labels[0], scores[0], time.perf_counter()))
            del detections

        except Exception as e:
            logger.error(f"Error at inspect_object: {e}")

    del model
    logger.info(f"[Process {os.getpid()}] Stop inspection...")


def save_object(output_dir: Path, stopped, result_queue: mp.Queue):
    logger.info(f"[Process {os.getpid()}] Start save object...")

    # original_dir = output_dir / "original"
    # heatmap_dir = output_dir / "heatmap"
    output_dir.mkdir(parents=True, exist_ok=True)
    # heatmap_dir.mkdir(parents=True, exist_ok=True)

    # defect_dir =
    # good_dir = output_dir / "good"
    class_names = {
        0: output_dir / "good",
        1: output_dir / "defect"
    }
    for path in class_names.values():
        path.mkdir(parents=True, exist_ok=True)

    while True:
        if stopped.is_set():
            break

        try:
            result = result_queue.get()
            if result is not None:
                object_id, frame, label, score = result
                output_path, _ = generate_unique_filename(
                    class_names[int(label)] / f"object-{object_id}_{get_now_str()}.jpg")
                cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # ori_h, ori_w = frame.shape[:2]
                # _, h, w = anomaly_map.shape
                # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                # heatmap = superimpose_anomaly_map(anomaly_map, frame, normalize=False)
                # heatmap = cv2.resize(heatmap, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
                # output_path_heatmap = heatmap_dir / output_path.name
                # cv2.imwrite(str(output_path_heatmap), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            del result

        except Exception as e:
            logger.error(f"Error at save_object: {e}")

    logger.info(f"[Process {os.getpid()}] Stop save object...")


def write_video(video_path, stopped, frame_queue: mp.Queue):
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (4024, 3036))
    while True:
        if stopped.is_set():
            break

        if frame_queue is None:
            break

        try:
            frame = frame_queue.get()

            if frame is not None:
                writer.write(frame)

        except Exception as e:
            logger.error(f"Error at write_video: {e}")

    writer.release()


class AIEngine:
    _save_time = None

    def __init__(self, camera_id):
        # initialize the AI Engine along with the boolean
        # used to indicate if the thread should be stopped or not
        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        self.stopped = mp.Event()
        self.reset_object_ids = mp.Event()

        # initialize the queue used to store frames read from
        # the video file
        self.result_detection_queue = mp.Queue()
        self.patches_queue = mp.Queue()
        self.result_predictor_queue = mp.Queue()

        self.predictions = {}
        self.push_schedule = {}
        self.processed_predictions = set()

        self.num_inspection_processes = NUM_INSPECTION_PROCESS
        self.time_to_push_after_disappear = TIME_TO_PUSH_OBJECT  # in second
        self.delay_system = DELAY_SYSTEM
        # self.time_warm_start = 5  # in seconds

        self.prev_object_ids = []

        self.defect_event = mp.Event()
        self.good_event = mp.Event()

        self.push_event = mp.Event()

        self.camera_id = camera_id

        self.save_image = SAVE_IMAGE

        self.result_cropped_queue = mp.Queue() if self.save_image else None
        self.save_dir = RESULT_DIR

        self.save_video = False
        self.save_video_event = mp.Event()
        # self.output_video_dir = Path(r"D:\maftuh\Projects\SME\anomalib\datasets") / f"kamera_{self.camera_id}"
        self.frame_save_queue = mp.Queue()
        self.video_write_process = None

        self._detector_params = {
            0: DETECTOR_PARAMS_SIDE_CAMERA,
            1: DETECTOR_PARAMS_TOP_CAMERA
        }

        self._preprocessor_params = {
            0: PREPROCESSOR_PARAMS_SIDE_CAMERA,
            1: PREPROCESSOR_PARAMS_TOP_CAMERA
        }

        self._inspector_params = {
            0: INSPECTOR_PARAMS_SIDE_CAMERA,
            1: INSPECTOR_PARAMS_TOP_CAMERA
        }

        # If None, we set it adaptive
        self.threshold = {
            0: 80,
            1: None
        }[self.camera_id]

        self.push_object_relay = {
            0: False,
            1: True
        }

    def start(self):
        processes = []
        process_result_work_thread = Thread(target=self.process_result_work_thread)
        process_result_work_thread.daemon = True
        process_result_work_thread.start()
        processes.append(process_result_work_thread)

        process_relay_check_work_thread = Thread(target=self.check_response_work_thread)
        process_relay_check_work_thread.daemon = True
        process_relay_check_work_thread.start()
        processes.append(process_relay_check_work_thread)

        inspection_processes = [
            mp.Process(
                target=inspect_object,
                args=(
                    self.stopped,
                    self._inspector_params[self.camera_id],
                    self.patches_queue,
                    self.result_predictor_queue,
                    self.result_cropped_queue
                )
            ) for _ in range(self.num_inspection_processes)
        ]
        for process in inspection_processes:
            process.start()

        inspection_pre_process = mp.Process(
            target=pre_process_inspection, args=(
                self.stopped,
                self._preprocessor_params[self.camera_id],
                self.result_detection_queue,
                self.patches_queue,
                self.reset_object_ids,
            )
        )
        inspection_pre_process.start()
        processes.append(inspection_pre_process)

        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        save_dir = self.save_dir / "images" / AIEngine._save_time / f"{self._detector_params[self.camera_id]['camera_name']}"

        if self.save_image:
            # save_dir.mkdir(parents=True, exist_ok=True)
            save_object_process = mp.Process(
                target=save_object, args=(
                    save_dir,
                    self.stopped,
                    self.result_cropped_queue
                )
            )
            save_object_process.start()
            processes.append(save_object_process)

        # time.sleep(self.time_warm_start)
        relay.open()
        # if relay.relay_cannot_open:
        #     pass

        return inspection_processes + processes

    def start_save_video(self, class_name=None):

        self.output_video_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{class_name}_" if class_name is not None else ""
        filename += F"{get_now_str(microsecond=False)}.avi"
        video_path = self.output_video_dir / filename
        self.video_write_process = mp.Process(
            target=write_video, args=(
                video_path,
                self.stopped,
                self.frame_save_queue
            )
        )
        self.save_video_event.set()
        self.video_write_process.start()

    def push_object(self, object_id):
        if object_id in self.processed_predictions:
            return

        # obj = self.predictions[object_id]
        # defects = obj["data"]["labels"]
        # scores = obj["data"]["scores"]
        logger.info("=" * 50)
        logger.info(f"Defects {self.predictions[object_id]['data']['labels']}")
        logger.info(f"Scores {self.predictions[object_id]['data']['scores']}")
        if _is_object_defect(
                self.predictions[object_id]["data"]["labels"],
                self.predictions[object_id]["data"]["scores"],
                threshold=self.threshold
        ):
            if self.push_object_relay.get(self.camera_id):
                time_to_push = self.predictions[object_id]["to_push_at"]
                while True:
                    current_time = time.perf_counter()
                    if current_time >= time_to_push:
                        self.defect_event.set()
                        self.push_event.set()
                        logger.info(f"Object {object_id} is in DEFECT condition, pushing it.")
                        log_relay_push(self.camera_id, object_id, self.predictions[object_id])
                        relay.write(self.camera_id, times=10)
                        break
        else:
            # If the object is deemed good, do not push
            self.good_event.set()
            log_relay_push(self.camera_id, object_id, self.predictions[object_id])
            logger.info(f"Object {object_id} is in GOOD condition, not pushing.")

        self.processed_predictions.add(object_id)
        self.predictions.pop(object_id)

    def process(self, frame, result, timestamp):
        # if self.save_video_event.is_set() and self.frame_save_queue is not None:
        #     self.frame_save_queue.put(frame.copy())

        # if len(result["boxes"]) == 0:
        #     return

        for object_id in result["track_ids"]:
            if object_id not in self.predictions:
                self.predictions[object_id] = {
                    "data": {
                        "labels": [],
                        "scores": [],
                    },
                    "start_tracked_at": timestamp
                }

        current_object_ids = result["track_ids"]
        disappear_object_ids = set(self.prev_object_ids) - set(current_object_ids)
        if len(disappear_object_ids) > 0:
            for object_id in disappear_object_ids:
                if object_id in self.predictions:
                    self.predictions[object_id]["to_push_at"] = timestamp + self.time_to_push_after_disappear
                    push_at = self.time_to_push_after_disappear - (time.perf_counter() - timestamp) - self.delay_system
                    # if push_at <= 0:
                    #     self.push_object_v2(object_id)
                    # else:
                    Timer(
                        max(0, push_at),
                        function=self.push_object,
                        args=(object_id,)
                    ).start()
        self.prev_object_ids = current_object_ids

        if self.result_detection_queue is not None and len(result["boxes"]) > 0:
            self.result_detection_queue.put((frame, result))

    def process_result_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            try:
                result = self.result_predictor_queue.get()

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
                logger.error(f"Error process_result_work_thread: {e}")

    # def pre_process_inspection(self, result_detection_queue: mp.Queue, patches_queue: mp.Queue):
    #     logger.info(f"[Process {os.getpid()}] Start preprocessing defect..")
    #     params = self._preprocessor_params[self.camera_id]
    #     pre_processor = DefectPreprocessor(**params)
    #     while True:
    #         if self.stopped.is_set():
    #             break
    #
    #         if result_detection_queue is None or result_detection_queue.empty():
    #             continue
    #
    #         try:
    #             detection_result = result_detection_queue.get(timeout=0.5)
    #             if detection_result is not None:
    #                 frame, result = detection_result
    #                 patches = pre_processor.object_post_process(frame, result)
    #                 for patch in patches:
    #                     # if patches_queue is not None and len(patches) > 0:
    #                     patches_queue.put(patch)
    #
    #                 del detection_result
    #         except Exception as e:
    #             logger.info("Error at pre_process_inspection:", e)
    #
    #     del pre_processor
    #     logger.info(f"[Process {os.getpid()}] Stop preprocessing defect...")

    # def inspect_object(self, patches_queue: mp.Queue, result_queue: mp.Queue):
    #     logger.info(f"[Process {os.getpid()}] Start inspecting object...")
    #     params = self._inspector_params[self.camera_id]
    #     model = DefectPredictor(**params).build()
    #
    #     while True:
    #         if self.stopped.is_set():
    #             break
    #
    #         if patches_queue is None or patches_queue.empty():
    #             continue
    #
    #         try:
    #             detection = patches_queue.get(timeout=0.5)
    #             if detection is not None:
    #                 # center, object_ids, frames = zip(*detections)
    #                 center, object_id, cropped = detection
    #                 result = model.predict([cropped])
    #                 result_queue.put((object_id, result["labels"][0], result["scores"][0], time.perf_counter()))
    #                 # if result is not None:
    #                 #     scores = result["pred_scores"].cpu().numpy()
    #                 #     labels = result["pred_labels"].cpu().numpy()
    #                 #     result_queue.put((object_id, labels[0], scores[0], time.perf_counter()))
    #             del detection
    #
    #         except Exception as e:
    #             logger.info("Error at inspect_object:", e)
    #
    #     del model
    #     logger.info(f"[Process {os.getpid()}] Stop inspection...")

    def check_response_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            if self.push_event.is_set():
                time.sleep(0.01)
                if relay.get_response(self.camera_id).is_set():
                    logger.info(f"Push Camera {self.camera_id} Success")
                else:
                    logger.error(f"Push Camera {self.camera_id} Failed, Try to open relay")
                    relay.close()
                    relay.open()
                    relay.write(self.camera_id, times=10)
                self.push_event.clear()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped.set()
        relay.close()

    def clear(self):
        self.result_detection_queue.put(None)
        self.patches_queue.put(None)
        self.result_predictor_queue.put(None)
        if self.result_cropped_queue is not None:
            self.result_cropped_queue.put(None)

        # clear_queue(self.result_detection_queue)
        # clear_queue(self.patches_queue)
        # clear_queue(self.result_predictor_queue)

    def clear_result(self):
        self.processed_predictions.clear()
        self.prev_object_ids = []
        self.predictions.clear()
        self.reset_object_ids.set()
