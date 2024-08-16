# import the necessary packages
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread, Timer

import cv2

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.models.detection.defect_judgment import PredictionResult
from anomalib.models.detection.object_counter import QualityCounter
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.utils import resize_image, draw_bbox, get_now_str, generate_unique_filename
from config import (
    PREPROCESSOR_PARAMS_TOP_CAMERA,
    PREPROCESSOR_PARAMS_SIDE_CAMERA,
    INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA, TIME_TO_PUSH_OBJECT, NUM_INSPECTION_PROCESS, DETECTOR_PARAMS_SIDE_CAMERA,
    DETECTOR_PARAMS_TOP_CAMERA, DELAY_SYSTEM, SAVE_IMAGE, RESULT_DIR, JUDGMENT_METHOD, IMAGE_THRESHOLD
)

logger = logging.getLogger("AutoInspection")


def start_spawn_method():
    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        logger.error(f"Error start method spawn: {e}")


def draw_result(frame, boxes, object_ids):
    frame = resize_image(frame, width=482)

    if boxes is None or object_ids is None:
        return frame

    if len(boxes) == 0:
        return frame

    for box, object_id in zip(boxes, object_ids):
        draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

    return frame


def clear_queue(q: mp.Queue):
    if q is None:
        return

    try:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception as e:
                logger.error(e)
    except Exception as e:
        logger.error(e)

    try:
        q.close()
        q.join_thread()
    except Exception as e:
        logger.error(e)


def put_queue_none(q: mp.Queue):
    if q is None:
        return

    try:
        q.put(None)
    except Exception as e:
        logger.error(f"Error at putting `None` to queue: {e}")


def log_relay_push(camera_id, obj_id, to_push_at):
    current_time = time.perf_counter()
    delay = current_time - to_push_at

    # Print the messages with camera_id included
    logger.info(f"[Camera {camera_id}] Judge object {obj_id} at scheduled time {to_push_at}")
    logger.info(f"[Camera {camera_id}] Actual judge time for object {obj_id}: {current_time}")
    logger.info(f"[Camera {camera_id}] Delay in pushing object {obj_id}: {delay:.6f} seconds")
    #
    # # Check and print running time if available
    # if "last_inspected_at" in obj and "start_tracked_at" in obj:
    #     running_time = obj['last_inspected_at'] - obj['start_tracked_at']
    #     logger.info(f"[Camera {camera_id}] Running time for object {obj_id}: {running_time:.6f} seconds")


def detect_object(
        stopped,
        model_params,
        frame_queue,
        frame_show_queue,
        result_detection_queue,
        result_predictions,
        push_object_queue
):
    logger.info(f"[Process {os.getpid()}] Start detecting object...")
    detector = SpringMetalDetector(**model_params).build()

    prev_object_ids = set()
    while True:
        if stopped.is_set():
            break

        try:
            # time_frame = None
            # while not frame_queue.empty:
            #     time_frame = frame_queue.get()

            time_frame = frame_queue.get()

            if time_frame is None:
                continue

            timestamp, frame = time_frame
            result = detector.predict(frame)
            current_object_ids = set(result["track_ids"])

            for object_id in current_object_ids:
                if object_id not in result_predictions:
                    result_predictions[object_id] = PredictionResult(
                        method=JUDGMENT_METHOD,
                        timestamp=time.perf_counter(),
                        threshold=None
                    )

            disappear_object_ids = prev_object_ids - current_object_ids
            if len(disappear_object_ids) > 0:
                for object_id in disappear_object_ids:
                    # if object_id in result_predictions:
                    push_in = TIME_TO_PUSH_OBJECT - (time.perf_counter() - timestamp) - DELAY_SYSTEM
                    push_object_queue.put((object_id, timestamp, push_in))
                    # print("push object", object_id, push_in)
                    # result_predictions[object_id]["to_push_at"] = timestamp + TIME_TO_PUSH_OBJECT
                    # print(result_predictions[object_id])

            prev_object_ids = current_object_ids

            # frame_show = draw_result(frame.copy(), result["boxes_n"], result["track_ids"])
            # frame_show_queue.put(frame_show)

            if result_detection_queue is not None and len(result["boxes"]) > 0:
                # print(result)
                result_detection_queue.put((timestamp, frame, result))

        except Exception as e:
            logger.error(f"Error at detect_object: {e}")

    del detector
    logger.info(f"[Process {os.getpid()}] Stop detecting object...")


def pre_process_inspection(
        stopped,
        model_params,
        result_detection_queue: mp.Queue,
        reset_object_ids: mp.Event,
        patches_queues,
):
    logger.info(f"[Process {os.getpid()}] Start preprocessing defect...")

    patch_count = 0
    pre_processor = DefectPreprocessor(**model_params)
    num_queues = len(patches_queues)
    while True:
        if stopped.is_set():
            break

        if result_detection_queue is None:
            break

        if reset_object_ids.is_set():
            pre_processor.patches.clear()
            reset_object_ids.clear()

        try:
            detection_result = result_detection_queue.get()

            if detection_result is None:
                continue

            timestamp, frame, result = detection_result

            patches = pre_processor.object_post_process(frame, result)

            for patch in patches:
                idx_q = patch_count % num_queues
                patches_queues[idx_q].put(patch)
                patch_count += 1

            del detection_result
        except Exception as e:
            logger.error(f"Error at pre_process_inspection: {e}")

    del pre_processor
    logger.info(f"[Process {os.getpid()}] Stop preprocessing defect...")


def get_all_from_queue(q):
    items = []
    while True:
        try:
            item = q.get_nowait()
            if item is None:
                continue
            items.append(item)
        except Exception as e:
            break
    return items


def check_lock_status(lock):
    if lock.acquire(timeout=0):
        lock.release()
        return True
    return False


def inspect_object(
        stopped,
        model_params,
        patches_queue: mp.Queue,
        result_queue: mp.Queue,
        result_save_queue: mp.Queue,
        built_event: mp.Event
):
    logger.info(f"[Process {os.getpid()}] Start inspection...")
    model = DefectPredictor(**model_params).build()
    built_event.set()

    while True:
        if stopped.is_set():
            break

        if patches_queue is None:
            break

        try:
            detections = patches_queue.get()
            if detections is not None:
                center, object_id, frame = detections
                start_time = time.perf_counter()
                result = model.predict([frame])
                logger.info(f"Prediction time for object {object_id}: {time.perf_counter() - start_time:.4f} seconds")

                if result is None:
                    continue

                label, score, mask = result["labels"][0], result["scores"][0], result["heatmaps"][0]
                result_queue.put((object_id, label, score, time.perf_counter()))
                if result_save_queue is not None:
                    result_save_queue.put((object_id, frame, label, score, mask))

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
                object_id, frame, label, score, anomaly_map = result
                output_path, _ = generate_unique_filename(
                    class_names[int(label)] / f"object-{object_id}_{get_now_str()}.jpg"
                )
                cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if label:
                    # save heatmap
                    # print("np.min(anomaly_map)", np.min(anomaly_map))
                    # print("np.max(anomaly_map)", np.max(anomaly_map))
                    # anomaly_map /= anomaly_map.max()
                    # h, w = anomaly_map.shape[-2:]
                    ori_h, ori_w = frame.shape[:2]
                    # image = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

                    # heatmap = superimpose_anomaly_map(anomaly_map, image, normalize=True)
                    heatmap = cv2.resize(anomaly_map, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
                    output_mask_path = output_path.parent / f"{output_path.stem}_heatmap.png"
                    cv2.imwrite(str(output_mask_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            del result

        except Exception as e:
            logger.error(f"Error at save_object: {e}")

    logger.info(f"[Process {os.getpid()}] Stop save object...")


def save_object_v1(output_dir: Path, stopped, result_queue: mp.Queue):
    logger.info(f"[Process {os.getpid()}] Start save object...")

    temp_dir = output_dir / "temp"

    while True:
        if stopped.is_set():
            break

        try:
            result = result_queue.get()
            if result is not None:
                object_id, frame, label, score, anomaly_map = result
                filename = f"object-{object_id}_{get_now_str()}.jpg"
                output_object_dir = temp_dir / f"object_{object_id}"
                output_object_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_object_dir / filename
                output_path, _ = generate_unique_filename(output_path)

                cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                ori_h, ori_w = frame.shape[:2]
                heatmap = cv2.resize(anomaly_map, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
                output_mask_path = output_path.parent / f"{output_path.stem}_heatmap.png"
                cv2.imwrite(str(output_mask_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            del result

        except Exception as e:
            logger.error(f"Error at save_object: {e}")

    logger.info(f"[Process {os.getpid()}] Stop save object...")


def push_object(
        lock,
        relay,
        push_queue: mp.Queue,
        result_predictions: dict,
        push_event: mp.Event,
        defect_event: mp.Event,
        good_event: mp.Event,
        stopped: mp.Event,
        camera_id: int,
        push_relay: bool,
        display_defect_sign: bool,
):
    logger.info(f"[Process {os.getpid()}] Start waiting to push the object...")
    processed_predictions = set()
    # time_clear_processed = time.perf_counter()

    sign = defect_event if display_defect_sign else good_event

    def push_func_():
        push_event.set()
        relay.write(camera_id, times=10)

    push_func = push_func_ if push_relay else lambda: None

    delays = []
    acquire_times = []

    while True:
        if stopped.is_set():
            break

        try:
            queue = push_queue.get()

            # if abs(time.perf_counter() - time_clear_processed) >= 3:
            #     # for key in processed_predictions:
            #     #     result_predictions.pop(key)
            #     if processed_predictions:
            #         processed_predictions.clear()

            if queue is None:
                continue

            time_clear_processed = time.perf_counter()
            object_id, timestamp, push_in = queue

            if object_id in processed_predictions:
                # result_predictions.pop(object_id)
                continue

            current_time = time.perf_counter()
            time_to_push = timestamp + TIME_TO_PUSH_OBJECT
            # wait = TIME_TO_PUSH_OBJECT - (current_time - timestamp) - DELAY_SYSTEM
            # print("waiting", wait)
            # if wait > 0.5:
            #     time.sleep(wait)

            # star_acquire = time.perf_counter()
            with lock:
                defect = result_predictions[object_id].final_labels_status
            # acquire_times.append(time.perf_counter() - star_acquire)
            # print("acquire time", time.perf_counter() - star_acquire)

            if defect:
                while True:
                    current_time = time.perf_counter()
                    remaining_time = time_to_push - current_time
                    if remaining_time <= 0.0001:
                        # start_time = time.perf_counter()
                        sign.set()
                        push_func()
                        # if display_defect_sign:
                        #     defect_event.set()
                        # else:
                        #     good_event.set()
                        # print("if display_defect_sign", time.perf_counter() - start_time)
                        # logger.info(
                        #     f"[Camera {camera_id}] Delay in judging object {object_id}: "
                        #     f"{(time.perf_counter() - time_to_push):.6f} seconds"
                        # )
                        # print("delay", (time.perf_counter() - time_to_push))
                        # if push_relay:
                        #     push_event.set()
                        #     relay.write(camera_id, times=10)
                        # delays.append((time.perf_counter() - time_to_push))
                        log_relay_push(camera_id, object_id, time_to_push)
                        logger.info(f"Object {object_id} is in DEFECT condition, pushing it.")
                        break
                    # print("delay", remaining_time)
                    # time.sleep(0.0001)
            else:
                logger.info(
                    f"[Camera {camera_id}] Delay in judging object {object_id}: "
                    f"{(time.perf_counter() - time_to_push):.6f} seconds"
                )
                good_event.set()
                log_relay_push(camera_id, object_id, time_to_push)
                logger.info(f"Object {object_id} is in GOOD condition, not pushing.")

            # start_time = time.perf_counter()
            logger.info(f"Defects {result_predictions[object_id].labels}")
            logger.info(f"Scores {result_predictions[object_id].scores}")
            logger.info("=" * 50)
            # print("logging time", time.perf_counter() - start_time)

            # start_time = time.perf_counter()
            processed_predictions.add(object_id)
            # print("processed_predictions.add(object_id)", time.perf_counter() - start_time)
            # start_time = time.perf_counter()
            result_predictions.pop(object_id)
            # print("result_predictions.pop(object_id)", time.perf_counter() - start_time)
        except Exception as e:
            logger.error(f"Error at push_object {e}")

    logger.info(f"[Process {os.getpid()}] Stop to push the object...")
    # logger.info(f"Mean Acquire time {np.mean(acquire_times)}")
    # logger.info(f"Std Acquire time {np.std(acquire_times)}", )
    # logger.info(f"Mean Delay {np.mean(delays)}")
    # logger.info(f"Std Delay {np.std(delays)}")
    # print("Acquire times", acquire_times)
    # print("Delays", delays)


class AIEngine:
    _save_time = None

    def __init__(self, camera_id, relay):
        self.camera_id = camera_id
        self.relay = relay

        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        self.stopped = mp.Event()
        self.reset_object_ids = mp.Event()

        # initialize the queue used to store frames read from
        # the video file
        self.result_detection_queue = mp.Queue()

        self.frame_queue = mp.Queue()
        self.frame_show_queue = mp.Queue()
        self.object_patches_queues = []
        self.result_predictor_queues = []
        self.result_predictor_queue = mp.Queue()
        self.push_object_queue = mp.Queue()

        self.lock_prediction = mp.Lock()
        self.manager = mp.Manager()
        self.result_predictions = {}
        self.processed_predictions = set()

        self.num_inspection_processes = NUM_INSPECTION_PROCESS
        self.time_to_push_after_disappear = TIME_TO_PUSH_OBJECT  # in second
        self.delay_system = DELAY_SYSTEM
        # self.time_warm_start = 5  # in seconds

        self.prev_object_ids = set()

        self.defect_event = mp.Event()
        self.good_event = mp.Event()

        self.push_event = mp.Event()
        self._warning_relay_func = None

        self.save_image = SAVE_IMAGE

        self.result_cropped_queue = mp.Queue() if self.save_image else None

        self._detector_params = {
            0: DETECTOR_PARAMS_TOP_CAMERA,
            1: DETECTOR_PARAMS_TOP_CAMERA
        }

        self.save_images_dir = (
            RESULT_DIR / "images" / AIEngine._save_time / f"{self._detector_params[self.camera_id]['camera_name']}-{self.camera_id}"
            if self.save_image else None
        )
        if self.save_images_dir is not None:
            self.save_dest_object_dir = {
                0: self.save_images_dir / "good",
                1: self.save_images_dir / "defect",
            }
            for path in self.save_dest_object_dir.values():
                path.mkdir(parents=True, exist_ok=True)

        self._preprocessor_params = {
            0: PREPROCESSOR_PARAMS_TOP_CAMERA,
            1: PREPROCESSOR_PARAMS_TOP_CAMERA
        }

        self._inspector_params = {
            0: INSPECTOR_PARAMS_TOP_CAMERA,
            1: INSPECTOR_PARAMS_TOP_CAMERA
        }

        # If None, we set it adaptive
        self.threshold = {
            0: 80,
            1: None
        }[self.camera_id]

        self.push_object_relay = {
            0: False,
            1: False
        }
        self.defect_signal_display = {
            0: True,
            1: True
        }

        self.processes = []

        self.model_built_events = []

        self.counter = None
        self._cache_dir = Path("./cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._counter_cache_file = self._cache_dir / f"cache_counter_camera_{self.camera_id}.json"
        self.load_counter()


    def start(self):
        # push_object_process = Thread(
        #     name="Push Object",
        #     target=push_object,
        #     args=(
        #         self.lock_prediction,
        #         self.relay,
        #         self.push_object_queue,
        #         self.result_predictions,
        #         self.push_event,
        #         self.defect_event,
        #         self.good_event,
        #         self.stopped,
        #         self.camera_id,
        #         self.push_object_relay.get(self.camera_id, False),
        #         self.defect_signal_display.get(self.camera_id, False)
        #     )
        # )
        # push_object_process.daemon = True
        # push_object_process.start()
        # self.processes.append(push_object_process)

        # detection_process = Thread(
        #     name="Detection",
        #     target=detect_object,
        #     args=(
        #         self.stopped,
        #         self._detector_params[self.camera_id],
        #         self.frame_queue,
        #         self.frame_show_queue,
        #         self.result_detection_queue,
        #         self.result_predictions,
        #         self.push_object_queue,
        #     )
        # )
        # detection_process.daemon = True
        # detection_process.start()
        # self.processes.append(detection_process)

        process_relay_check_work_thread = Thread(
            name="Check Relay Response",
            target=self.check_response_work_thread
        )
        process_relay_check_work_thread.daemon = True
        process_relay_check_work_thread.start()
        self.processes.append(process_relay_check_work_thread)

        for i in range(self.num_inspection_processes):
            built_event = mp.Event()
            self.model_built_events.append(built_event)
            object_patches = mp.Queue()
            self.object_patches_queues.append(object_patches)
            result_predictor = mp.Queue()
            self.result_predictor_queues.append(result_predictor)

            p = mp.Process(
                name=f"Inspection {i}",
                target=inspect_object,
                args=(
                    self.stopped,
                    self._inspector_params[self.camera_id],
                    self.object_patches_queues[i],
                    self.result_predictor_queue,
                    self.result_cropped_queue,
                    self.model_built_events[i]
                )
            )
            p.start()
            self.processes.append(p)

        inspection_pre_process = mp.Process(
            name="Pre-Process",
            target=pre_process_inspection,
            args=(
                self.stopped,
                self._preprocessor_params[self.camera_id],
                self.result_detection_queue,
                self.reset_object_ids,
                self.object_patches_queues,
            )
        )
        inspection_pre_process.start()
        self.processes.append(inspection_pre_process)

        inspection_post_process = Thread(
            name="Post-Process",
            target=self.post_process_inspection
        )
        inspection_post_process.daemon = True
        inspection_post_process.start()
        self.processes.append(inspection_post_process)

        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        if self.save_images_dir is not None:
            # save_dir.mkdir(parents=True, exist_ok=True)
            save_object_process = mp.Process(
                name="Save Object",
                target=save_object_v1,
                args=(
                    self.save_images_dir,
                    self.stopped,
                    self.result_cropped_queue
                )
            )
            save_object_process.start()
            self.processes.append(save_object_process)

    def process_frame(self, timestamp, frame, result):
        current_object_ids = set(result["track_ids"])

        for object_id in current_object_ids:
            if object_id not in self.result_predictions:
                self.result_predictions[object_id] = PredictionResult(
                    method=JUDGMENT_METHOD,
                    timestamp=time.perf_counter(),
                    threshold=IMAGE_THRESHOLD
                )

        disappear_object_ids = self.prev_object_ids - current_object_ids
        if len(disappear_object_ids) > 0:
            for object_id in disappear_object_ids:
                if object_id in self.result_predictions:
                    self.result_predictions[object_id].set_push_at(timestamp + TIME_TO_PUSH_OBJECT)
                    push_in = TIME_TO_PUSH_OBJECT - (time.perf_counter() - timestamp) - DELAY_SYSTEM
                    Timer(
                        max(0, push_in),
                        function=self.push_object,
                        args=(object_id,)
                    ).start()
                    # self.push_object_queue.put((object_id, timestamp, push_in))
                    # print("push object", object_id, push_in)
                    # # self.result_predictions[object_id]["to_push_at"] = timestamp + TIME_TO_PUSH_OBJECT
                    # print(self.result_predictions[object_id])

        self.prev_object_ids = current_object_ids

        # frame_show = draw_result(frame.copy(), result["boxes_n"], result["track_ids"])
        # frame_show_queue.put(frame_show)

        if self.result_detection_queue is not None and len(result["boxes"]) > 0:
            # print(result)
            self.result_detection_queue.put((timestamp, frame, result))

    def get_frame_show(self):
        return self.frame_show_queue.get()

    def post_process_inspection(self):
        result_predictions_temp = {}

        while True:
            if self.stopped.is_set():
                break

            try:
                result = self.result_predictor_queue.get()
                if result is None:
                    continue

                object_id, label, score, timestamp = result
                if object_id in self.result_predictions:
                    self.result_predictions[object_id].add_prediction(label, score, time.perf_counter())

                # if object_id in self.result_predictions:
                #     self.result_predictions[object_id] = result_predictions_temp[object_id]
                #
                # if object_id in result_predictions_temp and len(result_predictions_temp[object_id].labels) >= 3:
                #     result_predictions_temp.pop(object_id)

            except Exception as e:
                logger.error(f"Error at post_process_inspection: {e}")

    def check_response_work_thread(self):

        num_retry = 0
        max_num_retry = 3

        while True:
            if self.stopped.is_set():
                break

            if self.push_event.is_set():
                time.sleep(0.01)
                if self.relay.get_response(self.camera_id).is_set():
                    logger.info(f"Push Camera {self.camera_id} Success")
                else:
                    logger.error(f"Push Camera {self.camera_id} Failed, Try to open relay")
                    num_retry += 1
                    self.relay.close()
                    self.relay.open()
                    self.relay.write(self.camera_id, times=10)
                self.push_event.clear()

            if num_retry >= max_num_retry:
                if self._warning_relay_func is not None:
                    self._warning_relay_func()
                num_retry = 0

            time.sleep(0.001)

    def _push_relay_func(self):
        self.push_event.set()
        self.relay.write(self.camera_id, times=10)

    def push_object_v1(self):
        logger.info(f"[Process {os.getpid()}] Start waiting to push the object...")
        processed_predictions = set()
        # time_clear_processed = time.perf_counter()

        sign = self.defect_event if self.defect_signal_display[self.camera_id] else self.good_event

        push_func = self._push_relay_func if self.push_object_relay[self.camera_id] else lambda: None

        delays = []
        acquire_times = []

        while True:
            if self.stopped.is_set():
                break

            try:
                queue = self.push_object_queue.get()

                # if abs(time.perf_counter() - time_clear_processed) >= 3:
                #     # for key in processed_predictions:
                #     #     result_predictions.pop(key)
                #     if processed_predictions:
                #         processed_predictions.clear()

                if queue is None:
                    continue

                time_clear_processed = time.perf_counter()
                object_id, timestamp, push_in = queue

                if object_id in processed_predictions:
                    # result_predictions.pop(object_id)
                    continue

                current_time = time.perf_counter()
                time_to_push = timestamp + TIME_TO_PUSH_OBJECT
                # wait = TIME_TO_PUSH_OBJECT - (current_time - timestamp) - DELAY_SYSTEM
                # print("waiting", wait)
                # if wait > 0.5:
                #     time.sleep(wait)

                # star_acquire = time.perf_counter()
                with self.lock_prediction:
                    defect = self.result_predictions[object_id].final_labels_status
                # acquire_times.append(time.perf_counter() - star_acquire)
                # print("acquire time", time.perf_counter() - star_acquire)

                if defect:
                    while True:
                        current_time = time.perf_counter()
                        remaining_time = time_to_push - current_time
                        if remaining_time <= 0.0001:
                            # start_time = time.perf_counter()
                            sign.set()
                            push_func()
                            # if display_defect_sign:
                            #     defect_event.set()
                            # else:
                            #     good_event.set()
                            # print("if display_defect_sign", time.perf_counter() - start_time)
                            # logger.info(
                            #     f"[Camera {camera_id}] Delay in judging object {object_id}: "
                            #     f"{(time.perf_counter() - time_to_push):.6f} seconds"
                            # )
                            # print("delay", (time.perf_counter() - time_to_push))
                            # if push_relay:
                            #     push_event.set()
                            #     relay.write(camera_id, times=10)
                            # delays.append((time.perf_counter() - time_to_push))
                            log_relay_push(self.camera_id, object_id, time_to_push)
                            logger.info(f"Object {object_id} is in DEFECT condition, pushing it.")
                            break
                        # print("delay", remaining_time)
                        # time.sleep(0.0001)
                else:
                    logger.info(
                        f"[Camera {self.camera_id}] Delay in judging object {object_id}: "
                        f"{(time.perf_counter() - time_to_push):.6f} seconds"
                    )
                    self.good_event.set()
                    log_relay_push(self.camera_id, object_id, time_to_push)
                    logger.info(f"Object {object_id} is in GOOD condition, not pushing.")

                # start_time = time.perf_counter()
                logger.info(f"Defects {self.result_predictions[object_id].labels}")
                logger.info(f"Scores {self.result_predictions[object_id].scores}")
                logger.info("=" * 50)
                # print("logging time", time.perf_counter() - start_time)

                # start_time = time.perf_counter()
                processed_predictions.add(object_id)
                # print("processed_predictions.add(object_id)", time.perf_counter() - start_time)
                # start_time = time.perf_counter()
                self.result_predictions.pop(object_id)
                # print("result_predictions.pop(object_id)", time.perf_counter() - start_time)
            except Exception as e:
                logger.error(f"Error at push_object {e}")

        logger.info(f"[Process {os.getpid()}] Stop to push the object...")

    def push_object(self, object_id):
        if object_id in self.processed_predictions:
            return

        with self.lock_prediction:
            result = self.result_predictions[object_id]

        logger.info("=" * 50)
        logger.info(f"Result object {object_id} {result}")
        time_to_push = self.result_predictions[object_id].to_push_at
        # defect = random.choice([True, False])
        defect = result.final_labels_status
        if defect:
            while True:
                current_time = time.perf_counter()
                if current_time >= time_to_push:
                    self.defect_event.set()
                    self.counter.add_defect()
                    logger.info(f"Object {object_id} is in DEFECT condition, pushing it.")
                    log_relay_push(self.camera_id, object_id, time_to_push)
                    if self.push_object_relay.get(self.camera_id):
                        self.push_event.set()
                        self.relay.write(self.camera_id, times=10)
                    break
        else:
            # If the object is deemed good, do not push
            self.good_event.set()
            self.counter.add_good()
            log_relay_push(self.camera_id, object_id, time_to_push)
            logger.info(f"Object {object_id} is in GOOD condition, not pushing.")

        self.processed_predictions.add(object_id)
        with self.lock_prediction:
            self.result_predictions.pop(object_id)

        if self.save_images_dir is not None:
            temp_dir = self.save_images_dir / "temp"
            object_dir = temp_dir / f"object_{object_id}"
            if object_dir.is_dir():
                try:
                    for path in object_dir.iterdir():
                        if not result.final_labels_status and "heatmap" in path.name:
                            continue
                        dest = self.save_dest_object_dir[int(result.final_labels_status)] / path.name
                        # only move defect, otherwise delete
                        if result.final_labels_status:
                            shutil.move(path, dest)
                except Exception as e:
                    logger.error(f"Error at moving the images for object {object_id}: {e}")
                finally:
                    time.sleep(0.5)
                    shutil.rmtree(object_dir)

    def set_show_warning_relay(self, func):
        self._warning_relay_func = func

    def stop(self):
        self.save_counter()
        self.stopped.set()

        queues = ([self.result_detection_queue] + self.object_patches_queues + self.result_predictor_queues +
                  [self.result_cropped_queue, self.frame_queue, self.frame_show_queue, self.push_object_queue,
                   self.result_predictor_queue])

        for q in queues:
            put_queue_none(q)

        for p in self.processes:
            logger.info(f"Stopping {p}")
            if p.is_alive():
                p.join()

        for q in queues:
            if q is None:
                continue
            logger.info(f"Clearing {q}")
            clear_queue(q)

    @property
    def is_model_built(self):
        return all([event.is_set() for event in self.model_built_events])

    @property
    def is_stopped(self):
        return self.stopped.is_set()

    def clear_result(self):
        # self.result_predictions.clear()
        self.processed_predictions.clear()

    def save_counter(self):
        data = {
            "NG": self.counter.num_defect,
            "OK": self.counter.num_good
        }
        with open(self._counter_cache_file, "w") as f:
            json.dump(data, f)

    def load_counter(self):
        data = {}
        if self._counter_cache_file.is_file():
            with open(self._counter_cache_file, 'r') as f:
                data = json.load(f)

        self.counter = QualityCounter(num_good=data.get("OK", 0), num_defect=data.get("NG", 0))

    # def clear_result(self):
    #     self.reset_object_ids.set()
