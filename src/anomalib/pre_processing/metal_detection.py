import time

import cv2
import numpy as np

from anomalib.models.detection.utils import resize_image


def segment_patch(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 10, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    box = np.array([], dtype=int)

    if len(contours) > 0:
        contours = np.vstack(contours)
        contours = cv2.convexHull(contours)
        approx = cv2.approxPolyDP(contours, 0.02 * cv2.arcLength(contours, True), True)

        polygon = approx.squeeze()
        # Find the rotated rectangle that encloses the polygon
        rect = cv2.minAreaRect(polygon)
        # Get the four vertices of the rectangle
        box = cv2.boxPoints(rect)
        box = np.intp(box)

    return box


def segment_metal(image, bbox):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = int(x1 * width), int(y1 * height)
    x2, y2 = int(x2 * width), int(y2 * height)

    cropped_image = image[y1:y2, x1:x2].copy()
    box = segment_patch(cropped_image)
    if len(box) > 0:
        box += np.array([x1, y1])
    return box


def detect_metal(image, min_area=0.001, draw_bbox=False, verbose=True):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours, sort for largest contour, draw contour
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    bboxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        wn, hn = w / width, h / height
        area_n = wn * hn
        ratio = w / h
        if area_n < min_area:
            if verbose:
                print(f"[FAILED] Invalid values: ratio={ratio}, normalize area={area_n:.6f}")
            continue

        if verbose:
            print(f"[SUCCESS] Valid values: ratio={ratio}, normalize area={area_n:.6f}")

        bboxes.append([x / width, y / height, (x + w) / width, (y + h) / height])

        if draw_bbox:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, thresh, bboxes


def group_bboxes(bboxes, threshold_distance=0.1):
    if len(bboxes) <= 1:
        return [bboxes]

    def is_close(bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # if the value minus there is no intersection
        # and if big minus the distance is very far
        dist_x = min(x1_max, x2_max) - max(x1_min, x2_min)
        dist_y = min(y1_max, y2_max) - max(y1_min, y2_min)

        return dist_x > -threshold_distance and dist_y > -threshold_distance

    # Group bounding boxes
    groups = []
    for bbox in bboxes:
        assigned = False
        for group in groups:
            if any(is_close(bbox, existing_bbox) for existing_bbox in group):
                group.append(bbox)
                assigned = True
                break
        if not assigned:
            groups.append([bbox])

    return groups


def merge_bbox(grouped_bboxes):
    final_bboxes = []
    for bboxes in grouped_bboxes:
        bboxes = np.array(bboxes)
        min_x, min_y = np.min(bboxes[:, :2], axis=0)
        max_x, max_y = np.max(bboxes[:, 2:], axis=0)
        final_bboxes.append([min_x, min_y, max_x, max_y])

    return final_bboxes


def filter_bbox(bboxes, min_area=0.01):
    return list(filter(lambda b: (b[2] - b[0]) * (b[3] - b[1]) > min_area, bboxes))


def _test(path):
    cap = cv2.VideoCapture(str(path))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        time.sleep(0.25)
        frame = resize_image(frame, width=640)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image, thresh, bboxes = detect_metal(frame, draw_bbox=False, verbose=False)
        height, width = frame.shape[:2]

        if len(bboxes) > 0:
            grouped_bboxes = group_bboxes(bboxes)
            merged_bboxes = merge_bbox(grouped_bboxes)
            final_bboxes = filter_bbox(merged_bboxes)
            for i, bbox in enumerate(final_bboxes):
                x1, y1, x2, y2 = bbox
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)

                box = segment_metal(image, bbox)
                if len(box) > 0:
                    cv2.drawContours(image, [box], -1, (0, 0, 255), 1)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image = resize_image(image, width=640)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        thresh = resize_image(thresh, width=640)
        cv2.imshow("segmentation", image)
        cv2.imshow("mask", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "D:\\maftuh\\DATA\\SME-VIDEO-Dataset\\good\\samping\\Video_20240420185552637.avi"
    # path = "D:\\maftuh\\DATA\\16Apr24-sme-spring sheet\\good-kamera atas (1).avi"
    _test(path)

    # path = "D:\\maftuh\\DATA\\ObjectDetectionDataset20042024\\atas\\berr-kamera atas (1)_frame_1713632679_717765.jpg"
    # img = cv2.imread(path)
    # print(cls(img))
