# import the necessary packages
import queue
import time
from queue import Queue
from threading import Thread
from dataclasses import dataclass, field
from typing import Any

import cv2

from anomalib.models.detection.defect_detection_v1 import DefectPredictor
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.utils import resize_image
from fps import FPS


@dataclass
class TimeBundle:
    obj: Any
    timestamp: float = field(init=False)

    def __post_init__(self):
        self.timestamp = time.time()


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.frame_queue = Queue(maxsize=queue_size)
        self.detection_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.grab_thread = None
        self.detect_thread = None
        self.inspection_thread = None

    def start(self):
        # start a thread to read frames from the file video stream
        self.grab_thread = Thread(target=self.update, args=())
        self.grab_thread.daemon = True
        self.grab_thread.start()

        self.detect_thread = Thread(target=self.detect_object, args=())
        self.detect_thread.daemon = True
        self.detect_thread.start()

        self.inspection_thread = Thread(target=self.inspect_object, args=())
        self.inspection_thread.daemon = True
        self.inspection_thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.frame_queue.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                time.sleep(0.1)
                frame_bundle = TimeBundle(frame)

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.frame_queue.put(frame_bundle)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.result_queue.get()

    def detect_object(self):
        detector = SpringMetalDetector(
            path="runs/segment/train/weights/best.pt",
            distance_thresholds=(0.4, 0.5),
            pre_processor=lambda x: resize_image(x, width=640),
        )
        detector.build()

        while True:
            if self.stopped:
                break

            try:
                frame_bundle = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                frame = frame_bundle.obj
                result = detector.predict(frame)
                result_bundle = TimeBundle(result)
                result = detector.track(result)
                patches = detector.object_post_process(frame, result)
                patches_bundle = TimeBundle(patches)
                self.detection_queue.put((result_bundle, patches_bundle))
                self.result_queue.put(result)
                del frame_bundle
                objects_copy = detector.patches.keys().copy()
                for object_id, all_exists in objects_copy.items():
                    if all(all_exists):
                        detector.patches.pop(object_id)


        del detector

    def inspect_object(self):
        model = DefectPredictor(
            config_path="results/patchcore/mvtec/spring_sheet_metal/run.2024-05-15_01-26-33/config.yaml",
        )
        model.build()

        while True:
            if self.stopped:
                break

            try:
                detections_bundle = self.detection_queue.get(timeout=1)
            except queue.Empty:
                pass
            else:
                (result_bundle, patches_bundle) = detections_bundle
                patches = patches_bundle.obj
                patches = [patch for _, patch in patches.values()]
                result = model.predict(patches)

                if result is not None:
                    scores = result["pred_scores"].cpu().numpy()
                    labels = result["pred_labels"].cpu().long().numpy()
                    print(list(patches_bundle.obj.keys()), scores, labels)

                del detections_bundle

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
        self.grab_thread.join()
        self.detect_thread.join()
        self.inspection_thread.join()


if __name__ == "__main__":
    streamer = FileVideoStream(
        "sample/dented/atas/Video_20240420173419630.avi",
        queue_size=128,
    )
    streamer.start()
    fps = FPS().start()

    while True:

        if streamer.stopped:
            break

        result = streamer.read()

        # if len(result.boxes.track_id):
        #     print("object id per frame", result.boxes.track_id)

        frame = result.plot()
        frame = resize_image(frame, width=1024)
        fps.update()
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if fps.elapsed() >= 1:
            print(f"Processing time: {fps.fps()} FPS")
            fps.start()

    streamer.stop()
    cv2.destroyAllWindows()
