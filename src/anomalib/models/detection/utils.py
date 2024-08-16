import math
from typing import Union

import cv2
import numpy as np
from scipy.spatial import distance as dist
# from supervision.detection.utils import extract_ultralytics_masks, mask_non_max_suppression, box_non_max_suppression


def resize_shape(source_shape, width=None, height=None):
    (h, w) = source_shape

    if width is None and height is None:
        return h, w

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        shape = (height, int(w * r))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        shape = (int(h * r), width)

    return shape


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA) -> np.ndarray:
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    h, w = resize_shape(source_shape=image.shape[:2], width=width, height=height)

    # resize the image
    resized = cv2.resize(image, (w, h), interpolation=inter)

    # return the resized image
    return resized


def is_bbox_center(bbox, threshold: Union[list, tuple, int] = 0.1):
    center = 0.5

    if isinstance(threshold, int):
        threshold = [threshold, threshold]

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    is_center = abs(center_x - center) <= threshold[0] and abs(center_y - center) <= threshold[1]
    distance = np.sqrt((center_x - center) ** 2 + (center_y - center) ** 2)
    return is_center, distance


def expand_bbox(bbox, percentage=0.01):
    if isinstance(percentage, float):
        percentage = [percentage, percentage]

    is_normalized = np.all(np.array(bbox) <= 1)

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate the expansion amount
    width_expansion = width * percentage[0]
    height_expansion = height * percentage[1]

    new_x1 = x1 - width_expansion
    new_x2 = x2 + width_expansion
    new_y1 = y1 - height_expansion
    new_y2 = y2 + height_expansion

    if not is_normalized:
        new_x1, new_y1 = int(new_x1), int(new_y1)
        new_x2, new_y2 = int(new_x2), int(new_y2)

    return new_x1, new_y1, new_x2, new_y2


def create_centered_bbox(image_shape, center_box_size):
    """
    Create a centered bounding box with the specified size in an image.

    Parameters:
        image_shape (tuple): Tuple containing (image_width, image_height) of the image.
        center_box_size (tuple): Tuple containing (box_width_ratio, box_height_ratio) of the box relative to the image size.

    Returns:
        tuple: Tuple containing (x1, y1, x2, y2) of the centered bounding box.
    """

    image_height, image_width = image_shape[:2]
    box_width_ratio, box_height_ratio = center_box_size

    # Calculate box width and height based on ratios
    box_width = image_width * box_width_ratio
    box_height = image_height * box_height_ratio

    # Calculate box coordinates to center it in the image
    x_center = image_width // 2
    y_center = image_height // 2
    x1 = int(x_center - (box_width / 2))
    y1 = int(y_center - (box_height / 2))
    x2 = int(x_center + (box_width / 2))
    y2 = int(y_center + (box_height / 2))

    return x1, y1, x2, y2


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def draw_bbox(frame, result, target_names=None, expand_ratio=0.05):
    if result is None:
        return frame

    height, width = frame.shape[:2]

    cls, conf, xyxyn = result
    if target_names is not None:
        cls = target_names[cls]

    x1, y1, x2, y2 = xyxyn

    if expand_ratio > 0:
        x1, y1, x2, y2 = expand_bbox([x1, y1, x2, y2], expand_ratio)

    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    color = (0, 255, 0) if cls == 0 else (0, 0, 255)

    thickness = 10
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{cls}: {conf:.2f}"
    cv2.putText(frame, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness)

    return frame


def make_bbox_square(bbox):
    # Extract coordinates of the bounding box
    x1, y1, x2, y2 = bbox

    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Determine the larger dimension
    max_dim = max(width, height)

    # Calculate new coordinates to form a square
    new_x1 = x1 + (width - max_dim) / 2
    new_y1 = y1 + (height - max_dim) / 2
    new_x2 = new_x1 + max_dim
    new_y2 = new_y1 + max_dim

    return new_x1, new_y1, new_x2, new_y2


def unormalize_bbox(bbox, src_shape, transcend=False):
    x1, y1, x2, y2 = bbox
    height, width = src_shape[:2]

    x1, y1 = int(x1 * width), int(y1 * height)
    x2, y2 = int(x2 * width), int(y2 * height)

    if not transcend:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

    return x1, y1, x2, y2


def select_closest_bbox(result, conf_threshold=0.5, distance_thresholds=None):
    if len(result) == 0:
        return result

    if distance_thresholds is None:
        distance_thresholds = [0.5, 0.5]

    indices = np.array([False for _ in range(len(result))])
    for i, (box, conf) in enumerate(zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy())):
        if conf < conf_threshold:
            continue

        is_center, distance = is_bbox_center(box, distance_thresholds)

        if is_center:
            indices[i] = True

    return result[indices]


def transform_mask_to_rect(points: np.ndarray) -> list:
    contour = points.reshape((-1, 1, 2)).astype(np.int32)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    if approx is None:
        return []

    polygon = approx.squeeze()
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    return box.tolist()


def get_rect_angle_rotation(rect):
    sorted_rect_v = sorted(rect, key=lambda p: p[1])
    p12, p34 = sorted_rect_v[:2], sorted_rect_v[2:]

    (x1, y1), (x2, y2) = sorted(p12, key=lambda p: p[0])
    # (x3, y3), (x4, y4) = sorted(p34, key=lambda p: p[0], reverse=True)

    # x1, y1 = p1
    # x2, y2 = p2

    # Calculate the differences in y (dy) and x (dx) coordinates between the two points
    dy = y2 - y1
    dx = x2 - x1

    # Calculate the arctangent of dy/dx (angle in radians)
    radian = math.atan2(dy, dx)

    # Convert angle from radians to degrees
    angle = math.degrees(radian)

    return angle


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


# @timeit
def non_max_suppression(result, threshold=0.5, class_agnostic=True):
    if len(result) == 0:
        return result

    xyxy = result.boxes.xyxy.numpy()
    confidence = result.boxes.conf.numpy()
    class_id = result.boxes.cls.numpy().astype(int)
    mask = extract_ultralytics_masks(result)

    if class_agnostic:
        predictions = np.hstack((xyxy, confidence.reshape(-1, 1)))
    else:
        assert class_id is not None, (
            "Detections class_id must be given for NMS to be executed. If you"
            " intended to perform class agnostic NMS set class_agnostic=True."
        )
        predictions = np.hstack(
            (
                xyxy,
                confidence.reshape(-1, 1),
                class_id.reshape(-1, 1),
            )
        )

    if mask is not None:
        indices = mask_non_max_suppression(
            predictions=predictions, masks=mask, iou_threshold=threshold
        )
    else:
        indices = box_non_max_suppression(
            predictions=predictions, iou_threshold=threshold
        )

    return result[indices]


def center_crop(image, new_width, new_height):
    # Get the dimensions of the rotated_image
    rotated_image_height, rotated_image_width = image.shape[:2]

    # Get center image
    center_x, center_y = rotated_image_width / 2, rotated_image_height / 2

    # Calculate top-left corner coordinates for cropping
    x_min = int(center_x - new_width / 2)
    y_min = int(center_y - new_height / 2)

    # Ensure coordinates are within the frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_min + new_width)
    y_max = min(image.shape[0], y_min + new_height)

    # Crop the frame using the calculated coordinates
    cropped_frame = image[y_min:y_max, x_min:x_max]

    return cropped_frame


def remove_close_centers(result, min_distance=10):
    if len(result) <= 1:
        return result

    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy()])

    # Calculate pairwise distances
    distances = dist.cdist(centers, centers)

    # Mask to keep centers
    keep = np.ones(len(centers), dtype=bool)

    for i in range(len(centers)):
        if keep[i]:
            # Exclude distances to self and already excluded centers
            mask = (distances[i] < min_distance) & keep
            mask[i] = False
            keep[mask] = False

    return result[keep]
