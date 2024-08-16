import math

import cv2

from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.utils import resize_image

# Initialize Object Detection
model = SpringMetalDetector(
    path="runs/segment/train/weights/best.pt",
    pre_processor=lambda x: resize_image(x, width=640)
).build()
# path = "D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)\\GOOD\\Video_20240326152018928.avi"
# path = "datasets/kamera-atas_dented_1714378296-9803188.avi"
path = "sample/dented/atas/Video_20240420173419630.avi"
cap = cv2.VideoCapture(path)

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0
object_shape = (680, 560)
ori_max_distance = max(object_shape)

distances = []

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    # time.sleep(0.2)

    # Point current frame
    center_points_cur_frame = []
    result = model.predict(frame)
    frame = result.plot()
    ratio_max_distance = ori_max_distance / max(frame.shape[:2])
    # Detect objects on frame
    (class_ids, scores, boxes) = result.boxes.cls.numpy(), result.boxes.conf.numpy(), result.boxes.xyxy.numpy()
    for box in boxes:
        (x1, y1, x2, y2) = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                r = distance / max(frame.shape[:2])
                if r < 0.2:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                r = distance / max(frame.shape[:2])
                print(r)

                # Update IDs position
                if r < 0.2:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)

    print("Tracking objects")
    print(tracking_objects)

    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

# print(np.mean(distances), np.std(distances))

cap.release()
cv2.destroyAllWindows()
