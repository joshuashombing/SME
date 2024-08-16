# import the necessary packages
import multiprocessing as mp
import os
import time
from pathlib import Path
from threading import Thread

import cv2

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.models.detection.defect_judgment import PredictionResult
from anomalib.pre_processing.utils import resize_image, draw_bbox, get_now_str, generate_unique_filename
from config import (
    PREPROCESSOR_PARAMS_TOP_CAMERA, PREPROCESSOR_PARAMS_SIDE_CAMERA, INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA, TIME_TO_PUSH_OBJECT, NUM_INSPECTION_PROCESS, DETECTOR_PARAMS_SIDE_CAMERA,
    DETECTOR_PARAMS_TOP_CAMERA, DELAY_SYSTEM, SAVE_IMAGE, RESULT_DIR, PUSH_OBJECT, JUDGMENT_METHOD
)
from relay import Relay

relay = Relay()


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
            print(e)
    try:
        q.close()
        q.join_thread()
    except Exception as e:
        print(e)


def log_relay_push(camera_id, obj_id, obj):
    current_time = time.perf_counter()
    # delay = current_time - obj['to_push_at']

    # Print the messages with camera_id included
    print(f"[Camera {camera_id}] Judge object {obj_id} at scheduled time {obj['to_push_at']}")
    print(f"[Camera {camera_id}] Actual judge time for object {obj_id}: {current_time}")
    # print(f"[Camera {camera_id}] Delay in judging object {obj_id}: {delay:.6f} seconds")

    # Check and print running time if available
    if "last_inspected_at" in obj and "start_tracked_at" in obj:
        running_time = obj['last_inspected_at'] - obj['start_tracked_at']
        print(f"[Camera {camera_id}] Running time for object {obj_id}: {running_time:.6f} seconds")


def pre_process_inspection(stopped, model_params, result_detection_queue: mp.Queue, patches_queue: mp.Queue):
    print(f"[Process {os.getpid()}] Start preprocessing defect...")
    pre_processor = DefectPreprocessor(**model_params)
    time_clear_processed = time.perf_counter()
    while True:
        if stopped.is_set():
            break

        try:
            detection_result = result_detection_queue.get()

            if abs(time.perf_counter() - time_clear_processed) >= 3:
                pre_processor.patches.clear()

            if detection_result is not None:
                time_clear_processed = time.perf_counter()
                frame, result = detection_result
                patches = pre_processor.object_post_process(frame, result)
                for patch in patches:
                    patches_queue.put(patch)

                del detection_result
        except Exception as e:
            print("Error at pre_process_inspection:", e)

    del pre_processor
    print(f"[Process {os.getpid()}] Stop preprocessing defect...")


def inspect_object(
        threshold,
        lock,
        stopped,
        model_params,
        patches_queue: mp.Queue,
        result_save_queue: mp.Queue,
        result_predictions,
        build_event: mp.Event
):
    print(f"[Process {os.getpid()}] Start inspecting object...")
    build_event.clear()
    model = DefectPredictor(**model_params).build()
    build_event.set()

    results = {}

    while True:
        if stopped.is_set():
            break

        try:
            detections = patches_queue.get()
            if detections is not None:
                center, object_id, frame = detections
                result = model.predict([frame])
                for object_id, label, score, frame in zip([object_id], result["labels"], result["scores"], [frame]):
                    if object_id not in results:
                        results[object_id] = PredictionResult(
                            labels=[label],
                            scores=[score],
                            threshold=threshold,
                            method=JUDGMENT_METHOD,
                            timestamp=time.perf_counter()
                        )
                    else:
                        results[object_id].add_prediction(label, score, time.perf_counter())

                    lock.acquire()
                    if object_id in result_predictions:
                        result_predictions[object_id]["prediction"] = results[object_id]
                    lock.release()

                    if object_id in results and len(results[object_id].labels) >= 3:
                        results.pop(object_id)

                    if result_save_queue is not None:
                        result_save_queue.put((object_id, frame, label, score))

            del detections

        except Exception as e:
            print("Error at inspect_object:", e)

    del model
    print(f"[Process {os.getpid()}] Stop inspection...")


def push_object(
        lock,
        push_queue: mp.Queue,
        result_predictions: dict,
        defect_event: mp.Event,
        good_event: mp.Event,
        stopped: mp.Event,
        camera_id: int,
        push_relay: bool
):
    print(f"[Process {os.getpid()}] Start waiting to push the object...")
    processed_predictions = set()
    time_clear_processed = time.perf_counter()

    # delays = []
    # acquire_times = []

    while True:
        if stopped.is_set():
            break

        try:
            queue = push_queue.get()

            if abs(time.perf_counter() - time_clear_processed) >= 3:
                for key in processed_predictions:
                    result_predictions.pop(key)
                processed_predictions.clear()

            if queue is None:
                continue

            time_clear_processed = time.perf_counter()
            object_id, timestamp, push_in = queue

            if object_id in processed_predictions:
                # result_predictions.pop(object_id)
                continue

            current_time = time.perf_counter()
            time_to_push = timestamp + TIME_TO_PUSH_OBJECT
            wait = TIME_TO_PUSH_OBJECT - (current_time - timestamp) - DELAY_SYSTEM
            # print("waiting", wait)
            if wait > 0.5:
                time.sleep(wait)

            # star_acquire = time.perf_counter()
            lock.acquire()
            defect = result_predictions[object_id]["prediction"].final_labels_status
            lock.release()
            # acquire_times.append(time.perf_counter() - star_acquire)
            # print("acquire time", time.perf_counter() - star_acquire)

            print("=" * 50)
            print("Defects", result_predictions[object_id]["prediction"].labels)
            print("Scores", result_predictions[object_id]["prediction"].scores)
            if defect:
                while True:
                    current_time = time.perf_counter()
                    # print(current_time - time_to_push)
                    if current_time >= time_to_push:
                        defect_event.set()
                        print(
                            f"[Camera {camera_id}] Delay in judging object {object_id}: "
                            f"{(time.perf_counter() - time_to_push):.6f} seconds"
                        )
                        if push_relay:
                            relay.write(camera_id, times=10)
                        # delays.append((time.perf_counter() - time_to_push))
                        log_relay_push(camera_id, object_id, result_predictions[object_id])
                        print(f"Object {object_id} is in DEFECT condition, pushing it.")
                        break
                    # time.sleep(1e-6)
            else:
                print(
                    f"[Camera {camera_id}] Delay in judging object {object_id}: "
                    f"{(time.perf_counter() - time_to_push):.6f} seconds"
                )
                good_event.set()
                log_relay_push(camera_id, object_id, result_predictions[object_id])
                print(f"Object {object_id} is in GOOD condition, not pushing.")

            processed_predictions.add(object_id)
        except Exception as e:
            print("Error at push_object", e)

    print(f"[Process {os.getpid()}] Stop to push the object...")
    # print("Mean Acquire time", np.mean(acquire_times))
    # print("Std Acquire time", np.std(acquire_times))
    # print("Mean Delay", np.mean(delays))
    # print("Std Delay", np.std(delays))
    # print("Acquire times", acquire_times)
    # print("Delays", delays)


def save_object(output_dir: Path, stopped, result_queue: mp.Queue):
    print(f"[Process {os.getpid()}] Start save object...")

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {
        0: output_dir / "good",
        1: output_dir / "defect"
    }

    for path in class_names.values():
        path.mkdir(parents=True, exist_ok=True)

    while True:
        if stopped.is_set() and result_queue.empty():
            break

        if result_queue.empty():
            continue

        try:
            result = result_queue.get()
            if result is not None:
                object_id, frame, label, score = result
                output_path, _ = generate_unique_filename(
                    class_names[int(label)] / f"object-{object_id}_{get_now_str()}.jpg")
                cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            del result

        except Exception as e:
            print("Error at save_object:", e)

    print(f"[Process {os.getpid()}] Stop save object...")


class AIEngine:
    _save_time = None

    def __init__(self, camera_id):
        # initialize the AI Engine along with the boolean
        # used to indicate if the thread should be stopped or not
        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        self.stopped = mp.Event()
        self.build_event = mp.Event()

        # initialize the queue used to store frames read from
        # the video file
        self.result_detection_queue = mp.Queue()
        self.patches_queue = mp.Queue()
        self.push_queue = mp.Queue()

        self.lock_prediction = mp.Lock()
        self.manager = mp.Manager()
        self.result_predictions = self.manager.dict()

        self.num_inspection_processes = NUM_INSPECTION_PROCESS
        self.time_to_push_after_disappear = TIME_TO_PUSH_OBJECT  # in second
        self.delay_system = DELAY_SYSTEM

        self.prev_object_ids = []

        self.defect_event = mp.Event()
        self.good_event = mp.Event()

        self.camera_id = camera_id
        self.save_image = SAVE_IMAGE
        self.push_relay = PUSH_OBJECT

        self.result_save_queue = mp.Queue() if self.save_image else None
        self.save_dir = RESULT_DIR

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
            0: 47,
            1: None
        }[self.camera_id]

    def start(self):
        processes = []

        push_object_process = Thread(
            target=push_object, args=(
                self.lock_prediction,
                self.push_queue,
                self.result_predictions,
                self.defect_event,
                self.good_event,
                self.stopped,
                self.camera_id,
                self.push_relay
            )
        )
        push_object_process.daemon = True
        push_object_process.start()
        processes.append(push_object_process)

        inspection_processes = [
            mp.Process(
                target=inspect_object,
                args=(
                    self.threshold,
                    self.lock_prediction,
                    self.stopped,
                    self._inspector_params[self.camera_id],
                    self.patches_queue,
                    self.result_save_queue,
                    self.result_predictions,
                    self.build_event
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
            )
        )
        inspection_pre_process.start()
        processes.append(inspection_pre_process)

        if AIEngine._save_time is None:
            AIEngine._save_time = get_now_str(microsecond=False)

        save_dir = self.save_dir / "images" / AIEngine._save_time / f"{self._detector_params[self.camera_id]['camera_name']}"

        if self.save_image:
            save_object_process = mp.Process(
                target=save_object, args=(
                    save_dir,
                    self.stopped,
                    self.result_save_queue
                )
            )
            save_object_process.start()
            processes.append(save_object_process)

        relay.open()
        return inspection_processes + processes

    def process(self, frame, result, timestamp):

        for object_id in result["track_ids"]:
            if object_id not in self.result_predictions:
                self.result_predictions[object_id] = self.manager.dict()
                self.result_predictions[object_id]["prediction"] = PredictionResult(
                    method=JUDGMENT_METHOD,
                    timestamp=time.perf_counter(),
                    threshold=self.threshold
                )

        current_object_ids = result["track_ids"]
        disappear_object_ids = set(self.prev_object_ids) - set(current_object_ids)
        if len(disappear_object_ids) > 0:
            for object_id in disappear_object_ids:
                if object_id in self.result_predictions:
                    push_in = self.time_to_push_after_disappear - (time.perf_counter() - timestamp) - self.delay_system
                    self.push_queue.put((object_id, timestamp, push_in))
                    self.result_predictions[object_id]["to_push_at"] = timestamp + self.time_to_push_after_disappear

        self.prev_object_ids = current_object_ids

        if self.result_detection_queue is not None and len(result["boxes"]) > 0:
            self.result_detection_queue.put((frame, result))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped.set()
        self.result_detection_queue.put(None)
        self.patches_queue.put(None)
        self.push_queue.put(None)
        clear_queue(self.result_detection_queue)
        clear_queue(self.patches_queue)
        clear_queue(self.push_queue)
