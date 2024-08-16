# import the necessary packages
import multiprocessing as mp
import time
from pathlib import Path
from threading import Thread, Event
from typing import Union

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


class CameraOperation:
    def __init__(self, path, camera_id=1):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.camera_id = camera_id

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
        self.time_to_push_after_disappear = 1.5  # in second
        self.defect_event = Event()
        self.good_event = Event()

    @staticmethod
    def draw_result(frame, result):
        frame = resize_image(frame, width=640)

        if len(result) == 0:
            return frame

        for box, object_id in zip(result.boxes.xyxyn, result.boxes.track_id):
            draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

        return frame

    def start(self):
        # start a thread to read frames from the file video stream

        self.grab_thread_handle = Thread(target=self.grab_work_thread, args=())
        self.grab_thread_handle.daemon = True
        self.grab_thread_handle.start()

        self.result_detection = mp.Queue()
        self.inspection_pre_process_handle = Thread(target=self.pre_process_inspection, args=())
        self.inspection_pre_process_handle.daemon = True
        self.inspection_pre_process_handle.start()

        self.patches_queue = mp.Queue()
        for i in range(self.num_inspection_thread_handles):
            thread = Thread(target=self.inspect_object, args=())
            thread.daemon = True
            thread.start()
            self.inspection_thread_handles.append(thread)

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
                    # self.relay.write(self.camera_id)
                    self.defect_event.set()
                    log_relay_push(object_id, obj)
                else:
                    # If the object is deemed good, do not push
                    self.good_event.set()
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

        while True:
            if self.stopped:
                break

            self.push_object()
            (grabbed, frame) = self.stream.read()
            # time.sleep(0.05)

            if not grabbed:
                self.stopped = True

            if frame is not None:

                result = detector.predict(frame)
                result = detector.track(result)

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
                disappear_object_ids = set(prev_object_ids) - set(current_object_ids)
                if len(disappear_object_ids) > 0:
                    for object_id in disappear_object_ids:
                        if object_id in self.predictions:
                            self.predictions[object_id]["to_push_at"] = time.time() + self.time_to_push_after_disappear
                prev_object_ids = current_object_ids

                if self.result_detection is not None:
                    self.result_detection.put((frame, result))

                self.frame_queue.put(self.draw_result(frame, result))

        self.stream.release()
        print("Stop grabbing frame...")

    def read(self):
        # return next frame in the queue
        try:
            return self.frame_queue.get(timeout=1)
        except Exception as e:
            print(e)

    def pre_process_inspection(self):
        params = PREPROCESSOR_PARAMS_TOP_CAMERA if self.camera_id == 1 else PREPROCESSOR_PARAMS_SIDE_CAMERA
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

    def inspect_object(self):
        params = INSPECTOR_PARAMS_TOP_CAMERA if self.camera_id == 1 else INSPECTOR_PARAMS_SIDE_CAMERA
        # params = INSPECTOR_PARAMS_TOP_CAMERA
        model = DefectPredictor(**params).build()

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

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.frame_queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.frame_queue.qsize() > 0

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

    streamer.start()

    while True:
        if streamer.stopped:
            break

        # print("streamer.defect_event.is_set()", streamer.defect_event.is_set())
        # print("streamer.good_event.is_set()", streamer.good_event.is_set())

        frame = streamer.read()
        if frame is not None:
            cv2.imshow(video_path.name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    streamer.stop()
    cv2.destroyAllWindows()


def main():
    data_dir = Path("D:\\maftuh\\DATA\\2024-05-15_13-00")
    # data_dir = Path("D:\\maftuh\\DATA\\datasets-130524")
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

        # if side == "atas":
        #     continue

        stream_video(video_path, camera_id=camera_ids[side])


def main2():
    data_dir = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)\\DENTED")
    for video_path in data_dir.iterdir():
        stream_video(video_path, camera_id=1)


if __name__ == "__main__":
    main()
    # main2()
