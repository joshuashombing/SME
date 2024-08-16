import cv2
import time
from threading import Thread

from queue import Queue

import torch
import torch.multiprocessing as mp

from anomalib.models.detection.defect_detection import DefectDetector
from anomalib.models.detection.utils import resize_image

from fps import FPS
from model import SpringMetalInspector, transform_bbox


class FileVideoStream:
    def __init__(self, path, batch_size=1, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform
        self.batch_size = batch_size

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # initialize the queue used to store predictions
        self.detection_results = Queue(maxsize=queue_size)
        # self.inspection_results = mp.Queue(maxsize=queue_size)
        # self.predict_process = mp.Process(target=self.inspect_objects, args=())
        # self.predict_process.start()

        # initialize threads
        self.thread_read = Thread(target=self.read_frames, args=())
        self.thread_detect = Thread(target=self.detect_objects, args=())
        # self.thread_inspect = Thread(target=self.inspect_objects, args=())
        self.thread_read.daemon = True
        self.thread_detect.daemon = True
        # self.thread_inspect.daemon = True

        self.model = DefectDetector(
            config_path="configs/kamera-atas.yaml",
            yolo_path="runs/segment/train/weights/best.pt",
        )
        self.model.build()

    def start(self):
        # start threads to read frames and perform prediction
        self.thread_read.start()
        self.thread_detect.start()
        # self.thread_inspect.start()
        return self

    def read_frames(self):
        # keep looping infinitely
        while not self.stopped:
            # ensure the queue has room in it
            if not self.Q.full():
                # read the next batch of frames from the file
                # frames = []
                # for _ in range(self.batch_size):
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if grabbed:
                    if self.transform:
                        frame = self.transform(frame)

                        # frames.append(frame)

                    # add the batch of frames to the queue
                    self.Q.put(frame)
            # else:
                # time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def detect_objects(self):
        # keep looping infinitely
        while not self.stopped:
            # ensure there are frames in the queue
            if not self.Q.empty():
                frames = self.Q.get()
                # print("len frames", len(frames))
                predictions = self.model.detector.track(frames, persist=True, stream=True, verbose=True, tracker="bytetrack.yaml")
                # batches = self.model.predict_defect(frames, predictions)
                del frames

                # for i, batch in enumerate(batches):
                #     self.inspection_results.put(batch)

                for i, result in enumerate(predictions):
                    self.detection_results.put(result.plot())

                del predictions

    # def inspect_objects(self, model):
    #     batch
    #     with torch.no_grad():
    #         defect_result = model.predict_step(batch, 0)
    #
    #     self.inspection_results.put(defect_result)

    def read(self):
        # return next batch of frames in the queue
        return self.detection_results.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return not self.stopped

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until streams are released
        self.thread_read.join()
        self.thread_detect.join()


# model = SpringMetalInspector()
# model.build()
#
#
# def transform(frame):
#     return model.predict(frame, "kamera-atas")


if __name__ == "__main__":
    streamer = FileVideoStream(
        "sample/good/atas/Video_20240420185538514.avi",
        # transform=transform,
        queue_size=128,
        batch_size=4
    )
    streamer.start()
    fps = FPS().start()

    while True:

        if streamer.stopped:
            break

        frame = streamer.read()
        # time.sleep(0.1)
        # print(frame.shape)
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
