import multiprocessing as mp
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from anomalib.models.detection.defect_detection_v1 import DefectPredictor, DefectPreprocessor
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.utils import resize_image, draw_bbox
from config import (
    DETECTOR_PARAMS_TOP_CAMERA,
    DETECTOR_PARAMS_SIDE_CAMERA,
    PREPROCESSOR_PARAMS_TOP_CAMERA,
    PREPROCESSOR_PARAMS_SIDE_CAMERA,
    INSPECTOR_PARAMS_TOP_CAMERA,
    INSPECTOR_PARAMS_SIDE_CAMERA
)


def clear_queue(q: mp.Queue):
    if q is None:
        return

    while not q.empty():
        try:
            q.get_nowait()
        except Exception as e:
            print(e)
    q.close()
    q.join_thread()


def log_relay_push(index, obj):
    current_time = time.time()
    delay = current_time - obj['to_push_at']
    # Format and print the messages
    print(f"Push object {index} at scheduled time {obj['to_push_at']}")
    print(f"Actual push time for object {index}: {current_time}")
    print(f"Delay in pushing object {index}: {delay:.2f} seconds")

    # Check and print running time if available
    if obj.get("last_inspected_at") and obj.get("start_tracked_at"):
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


class CameraOperation:
    def __init__(self, path, camera_id=1):
        self.path = path
        self.stopped = mp.Value('b', False)
        self.camera_id = camera_id

        self.frame_queue = mp.Queue()
        self.detection_queue = mp.Queue()
        self.patches_queue = mp.Queue()
        self.defect_result_queue = mp.Queue()

        self.predictions = {}
        self.processed_predictions = set()

        self.time_to_push_after_disappear = 1.5  # in seconds
        self.time_warm_start = 5  # in seconds
        self.num_inspection_processes = 3

    @staticmethod
    def draw_result(frame, result):
        frame = resize_image(frame, width=640)
        if len(result) == 0:
            return frame

        for box, object_id in zip(result.boxes.xyxyn, result.boxes.track_id):
            draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))
        return frame

    def start(self):
        self.stopped.value = False

        process_result_work_thread = threading.Thread(target=self.process_result_work_thread)
        process_result_work_thread.daemon = True
        process_result_work_thread.start()

        inspection_processes = [
            mp.Process(
                target=self.inspect_object,
                args=(self.patches_queue, self.defect_result_queue)
            ) for _ in range(self.num_inspection_processes)
        ]
        for process in inspection_processes:
            process.start()

        inspection_pre_process = mp.Process(
            target=self.pre_process_inspection, args=(self.detection_queue, self.patches_queue)
        )
        inspection_pre_process.start()

        time.sleep(self.time_warm_start)
        grab_thread = threading.Thread(target=self.grab_work_thread)
        grab_thread.daemon = True
        grab_thread.start()

        return inspection_processes + [inspection_pre_process, grab_thread, process_result_work_thread]

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
                    # self.defect_event.set()
                    log_relay_push(object_id, obj)
                else:
                    # self.good_event.set()
                    print(f"Object {object_id} is in good condition, not pushing.")

                print("Defects", defects)
                print("Scores", scores)
                print("=" * 50)

                self.processed_predictions.add(object_id)

    def grab_work_thread(self):
        params = DETECTOR_PARAMS_TOP_CAMERA if self.camera_id == 1 else DETECTOR_PARAMS_SIDE_CAMERA
        detector = SpringMetalDetector(**params).build()

        print("Start grabbing frame...")
        prev_object_ids = []
        stream = cv2.VideoCapture(self.path)

        while True:
            if self.stopped.value:
                break

            try:
                self.push_object()
                grabbed, frame = stream.read()

                if not grabbed:
                    self.stopped.value = True

                if frame is not None:
                    result = detector.predict(frame)
                    result = detector.track(result)

                    for object_id in result.boxes.track_id:
                        if object_id not in self.predictions:
                            self.predictions[object_id] = {
                                "data": {
                                    "labels": [],
                                    "scores": [],
                                },
                                "start_tracked_at": time.time(),
                                "to_push_at": None,
                                "last_inspected_at": None
                            }

                    current_object_ids = result.boxes.track_id
                    disappear_object_ids = set(prev_object_ids) - set(current_object_ids)
                    for object_id in disappear_object_ids:
                        if object_id in self.predictions:
                            self.predictions[object_id]["to_push_at"] = time.time() + self.time_to_push_after_disappear
                        print(self.predictions.get(object_id))
                    prev_object_ids = current_object_ids

                    self.detection_queue.put((frame, result))
                    self.frame_queue.put(self.draw_result(frame, result))
            except Exception as e:
                pass

        stream.release()
        print("Stop grabbing frame...")

    def process_result_work_thread(self):
        while True:
            if self.stopped.value and self.defect_result_queue.empty():
                break

            try:
                if self.defect_result_queue.empty():
                    continue

                result = self.defect_result_queue.get_nowait()

                if result is None:
                    continue

                object_ids, labels, scores, timestamp = result

                for object_id, score, label in zip(object_ids, scores, labels):
                    if object_id in self.predictions.keys():
                        self.predictions[object_id]["last_inspected_at"] = timestamp
                        self.predictions[object_id]["data"]["scores"].append(score)
                        self.predictions[object_id]["data"]["labels"].append(label)
            except Exception as e:
                pass

    def read(self):
        try:
            return self.frame_queue.get(timeout=1)
        except Exception as e:
            pass

    def pre_process_inspection(self, result_detection: mp.Queue, patches_queue: mp.Queue):
        params = PREPROCESSOR_PARAMS_TOP_CAMERA if self.camera_id == 1 else PREPROCESSOR_PARAMS_SIDE_CAMERA
        pre_processor = DefectPreprocessor(**params)

        print("Start preprocessing defect...")
        while True:
            if self.stopped.value:
                break

            if result_detection.empty():
                continue

            try:
                detection_result = result_detection.get_nowait()
            except Exception as e:
                print(e)
            else:
                frame, result = detection_result
                patches = pre_processor.object_post_process(frame, result)
                if len(patches) > 0:
                    patches_queue.put(patches)

        print("Stop preprocessing defect...")

    def inspect_object(self, patches_queue: mp.Queue, result_queue: mp.Queue):
        params = INSPECTOR_PARAMS_TOP_CAMERA if self.camera_id == 1 else INSPECTOR_PARAMS_SIDE_CAMERA
        model = DefectPredictor(**params).build()
        print("Start inspecting object...")
        while True:
            if self.stopped.value:
                break

            if patches_queue.empty():
                continue

            try:
                detections = patches_queue.get_nowait()
                if detections is None:
                    continue
                centers, object_ids, frames = zip(*detections)
                result = model.predict(frames)
                if result is not None:
                    scores = result["pred_scores"].cpu().numpy()
                    labels = result["pred_labels"].cpu().numpy()
                    result_queue.put((object_ids, labels, scores, time.time()))
                    # for object_id, score, label in zip(object_ids, scores, labels):
                    #     if object_id in self.predictions.keys():
                    #         self.predictions[object_id]["last_inspected_at"] = time.time()
                    #         self.predictions[object_id]["data"]["scores"].append(score)
                    #         self.predictions[object_id]["data"]["labels"].append(label)
                    #     print(self.predictions.get(object_id))
            except Exception as e:
                pass

        print("Stop inspecting object...")

    def running(self):
        return self.more() or not self.stopped.value

    def more(self):
        tries = 0
        while self.frame_queue.qsize() == 0 and not self.stopped.value and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.frame_queue.qsize() > 0

    def stop(self):
        self.stopped.value = True

        clear_queue(self.frame_queue)
        clear_queue(self.detection_queue)
        clear_queue(self.patches_queue)
        clear_queue(self.defect_result_queue)


def extract_video_filename(video_path: Path):
    camera_name, class_name, timestamp = video_path.stem.split("_")
    _, side = camera_name.split("-")
    return side, camera_name, class_name, timestamp


def stream_video(video_path, camera_id):
    print("Processing", video_path)

    streamer = CameraOperation(
        str(video_path),
        camera_id=camera_id
    )

    processes = streamer.start()
    while True:
        if streamer.stopped.value:
            break

        frame = streamer.read()
        if frame is not None:
            cv2.imshow(video_path.name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    streamer.stop()

    for p in processes:
        p.join()

    cv2.destroyAllWindows()


def main():
    data_dir = Path("D:\\maftuh\\DATA\\2024-05-15_13-00")
    for video_path in data_dir.iterdir():
        side, _, class_name, _ = extract_video_filename(video_path)

        if side == "atas":
            if class_name == "berr":
                class_name = "good"
        else:
            if class_name == "dented":
                continue

        camera_ids = {
            "atas": 1,
            "samping": 0
        }

        if side not in camera_ids:
            continue

        stream_video(video_path, camera_id=camera_ids[side])


if __name__ == "__main__":
    main()
