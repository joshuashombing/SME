import os
import shutil
from typing import Union

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .metal_detection import detect_metal, group_bboxes, filter_bbox, merge_bbox, segment_metal
from .utils import validate_path, get_now_str, generate_unique_filename, resize_image
from ..models.detection.utils import transform_mask_to_rect


def normalize_coordinate(src_shape, point):
    return point[0] / src_shape[1], point[1] / src_shape[0]


def write_yolo_label(path: Union[Path, str], values: list, mode="a"):
    with open(path, mode) as f:
        f.write(" ".join([str(v) for v in values]) + "\n")


def create_yolo_data(image, output_dir: Union[Path, str], class_id=0, filename=None, width=640, height=None):
    image = resize_image(image, width=width, height=height)
    _, _, bboxes = detect_metal(image, draw_bbox=False, verbose=False)

    if len(bboxes) == 0:
        return

    path = validate_path(output_dir)
    images_dir = path / "images"
    labels_dir = path / "labels"
    bboxed_dir = path / "bboxed"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    bboxed_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{get_now_str()}.jpg"

    image_path = images_dir / filename
    image_path, _ = generate_unique_filename(image_path)
    label_path = labels_dir / f"{image_path.stem}.txt"
    bbox_path = bboxed_dir / image_path.name

    grouped_bboxes = group_bboxes(bboxes)
    merged_bboxes = merge_bbox(grouped_bboxes)
    final_bboxes = filter_bbox(merged_bboxes)

    if len(final_bboxes) == 0:
        return

    bbox_image = image.copy()
    for bbox in final_bboxes:
        rect_polygon = segment_metal(image, bbox)

        if len(rect_polygon) > 0:
            cv2.drawContours(bbox_image, [rect_polygon], -1, (0, 255, 0), 2)

        rect_polygon_normalized = np.array([
            list(normalize_coordinate(image.shape, (max(0, x), max(0, y))))
            for x, y in rect_polygon
        ])
        rect_polygon_normalized = rect_polygon_normalized.flatten().tolist()
        write_yolo_label(label_path, [class_id] + rect_polygon_normalized)

    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(bbox_path), cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))


yolo_model = None


def create_yolo_data_2(image, output_dir: Union[Path, str], class_id=0, filename=None, width=640, height=None,
                       ckpt_path="tools/yolov8.pt", device="cuda:0"):
    global yolo_model
    if yolo_model is None or yolo_model.ckpt_path != ckpt_path:
        from ultralytics import YOLO
        yolo_model = YOLO(ckpt_path)

    image = resize_image(image, width=width, height=height)
    result = yolo_model.predict(image, verbose=False, device=device)[0].cpu()

    if len(result.boxes) == 0:
        return

    path = validate_path(output_dir)
    images_dir = path / "images"
    labels_dir = path / "labels"
    bboxed_dir = path / "bboxed"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    bboxed_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{get_now_str()}.jpg"
    else:
        filename = f"{Path(filename).stem}_{get_now_str()}{Path(filename).suffix}"

    image_path = images_dir / filename
    image_path, _ = generate_unique_filename(image_path)
    label_path = labels_dir / f"{image_path.stem}.txt"
    bbox_path = bboxed_dir / image_path.name

    if result.masks is not None:
        bbox_image = image.copy()
        for xy in result.masks.xy:
            rect_polygon = transform_mask_to_rect(xy)
            if len(rect_polygon) > 0:
                cv2.drawContours(bbox_image, [np.array(rect_polygon).reshape((-1, 1, 2))], -1, (0, 255, 0), 2)

            rect_polygon_normalized = np.array([
                list(normalize_coordinate(image.shape, (max(0, x), max(0, y))))
                for x, y in rect_polygon
            ])
            rect_polygon_normalized = rect_polygon_normalized.flatten().tolist()
            write_yolo_label(label_path, [class_id] + rect_polygon_normalized)

        cv2.imwrite(str(bbox_path), bbox_image)
    else:
        # result.boxes.data[:, -1] = torch.tensor([class_id for _ in range(len(result.boxes.data[:, -1]))])
        result.save(bbox_path)
        result.save_txt(label_path)

    cv2.imwrite(str(image_path), image)


def create_yolo_data_class_pcb_counter(image, output_dir: Union[Path, str], class_id=0, filename=None, width=640,
                                       height=None):
    image = resize_image(image, width=width, height=height)

    bbox_classification = None  # buat function

    path = validate_path(output_dir)
    images_dir = path / "images"
    labels_dir = path / "labels"
    bboxed_dir = path / "bboxed"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    bboxed_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{get_now_str()}.jpg"
    else:
        filename = f"{Path(filename).stem}_{get_now_str()}{Path(filename).suffix}"

    image_path = images_dir / filename
    image_path, _ = generate_unique_filename(image_path)
    label_path = labels_dir / f"{image_path.stem}.txt"
    bbox_path = bboxed_dir / image_path.name

    if bbox_classification is not None:
        bbox_image = image.copy()

        for bbox in bbox_classification:
            x1, y1, x2, y2 = bbox
            x1_n, y1_n = normalize_coordinate(bbox_image.shape, (x1, y1))
            x2_n, y2_n = normalize_coordinate(bbox_image.shape, (x2, y2))

            center_x = (x1_n + x2_n) / 2
            center_y = (y1_n + y2_n) / 2
            w, h = abs(x2_n - x1_n), abs(y2_n - y1_n)

            write_yolo_label(label_path, [class_id, center_x, center_y, w, h])

        cv2.imwrite(str(bbox_path), bbox_image)

    cv2.imwrite(str(image_path), image)


def clear_missing(data_dir: Union[Path, str]):
    print("Removing missing bbox image", data_dir)
    data_dir = validate_path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    bboxed_dir = data_dir / "bboxed"

    missing_bbox = set(os.listdir(images_dir)) - set(os.listdir(bboxed_dir))
    print("Missing found", len(missing_bbox))
    for filename in missing_bbox:
        img_path = images_dir / filename
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        os.remove(img_path)
        os.remove(label_path)


def copy_data(data_dir, list_filename, split="train", dest_dir=None):
    data_dir = validate_path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if dest_dir is not None:
        output_dir = validate_path(dest_dir) / split
    else:
        output_dir = data_dir.parent / f"{data_dir.name}_split" / split

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying data from {data_dir} to {output_dir}")
    for filename in tqdm(list_filename):
        img_path = images_dir / filename
        label_path = labels_dir / f"{img_path.stem}.txt"

        dest_img_path = output_images_dir / img_path.name
        dest_label_path = output_labels_dir / label_path.name
        shutil.copy(img_path, dest_img_path)
        shutil.copy(label_path, dest_label_path)

