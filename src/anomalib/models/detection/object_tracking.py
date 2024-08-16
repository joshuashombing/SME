import json
import math
import time
from collections import OrderedDict
from enum import Enum

import cv2

from anomalib.pre_processing.utils import resize_image


class Direction(int, Enum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1


class ObjectTracker:
    def __init__(self, direction=Direction.LEFT_TO_RIGHT):
        self.prev_center_points = []
        # self.curr_center_points = []

        self.objects = OrderedDict()
        self.distances = []
        self._frame_count = 0

        self.frame_shape = (680, 560)
        self.track_id = 0

        self.threshold_distance_ratio = 0.15

        self.data = []
        # self.distance_range = (-5, 200)  # right to left
        self.distance_range = (-250, 5)  # left to right
        # self.lines = [0.4, 0.5, 0.6]
        self.direction_multiplier = 1 if direction == Direction.LEFT_TO_RIGHT else -1

    @staticmethod
    def calculate_distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # def line_distance(self, center_x, idx):
    #     return self.lines[idx] - center_x

    @staticmethod
    def get_center(box):
        (x1, y1, x2, y2) = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def update(self, boxes):
        self._frame_count += 1

        boxes = sorted(boxes, key=lambda box: (box[0] + box[2]) / 2)

        curr_center_points = [self.get_center(box) for box in boxes]

        # for box in boxes:
        #     center = self.get_center(box)
        #     if center[0] not in {c[0] for c in curr_center_points}:
        #         curr_center_points.append(center)

        if self._frame_count <= 2:
            for pt in curr_center_points:
                for pt2 in self.prev_center_points:
                    # r = self.calculate_distance(pt2, pt) / max(self.frame_shape)
                    if self.direction_multiplier * (pt2[0] - pt[0]) < 5:
                        self.objects[self.track_id] = pt
                        self.track_id += 1
                        break

        else:
            objects_copy = self.objects.copy()
            curr_center_points_copy = curr_center_points.copy()

            for object_id, pt2 in objects_copy.items():
                object_exists = False
                for pt in curr_center_points_copy:
                    # r = self.calculate_distance(pt2, pt) / max(self.frame_shape)
                    # print("distance", pt2[0] - pt[0])

                    # Update IDs position
                    if self.direction_multiplier * (pt2[0] - pt[0]) < 5:
                        self.objects[object_id] = pt
                        object_exists = True
                        if pt[0] in {c[0] for c in curr_center_points}:
                            curr_center_points.remove(pt)
                        break

                if not object_exists:
                    self.objects.pop(object_id)

        # Add new IDs found
        for pt in curr_center_points:
            if pt[0] not in {c[0] for c in self.objects.values()}:
                self.objects[self.track_id] = pt
                self.track_id += 1

        self.prev_center_points = curr_center_points.copy()


def main():
    from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
    # Initialize Object Detection
    model = SpringMetalDetector(
        path="runs/segment/train/weights/best.pt",
        transform=lambda x: resize_image(x, width=640)
    ).build()

    tracker = ObjectTracker()

    # path = "D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)\\GOOD\\Video_20240326152018928.avi"
    # path = "datasets/kamera-atas_dented_1714378296-9803188.avi"
    path = r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_10-03-00\kamera-atas.avi"
    cap = cv2.VideoCapture(path)

    distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time.sleep(0.05)

        # Point current frame
        result = model.predict(frame)
        frame = result.plot()

        # Detect objects on frame
        (class_ids, scores, boxes) = result.boxes.cls.numpy(), result.boxes.conf.numpy(), result.boxes.xyxy.numpy()

        tracker.update(boxes)
        if len(tracker.objects) > 0:
            print(tracker.objects)

        for object_id, pt in tracker.objects.items():
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # with open("sample.json", "w") as outfile:
    #     json.dump(tracker.data, outfile, indent=4)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
